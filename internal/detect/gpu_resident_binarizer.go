//go:build jabcode_gpu

package detect

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
)

//go:embed shaders/histogram_rgb.wgsl
var histogramRGBWGSL string

//go:embed shaders/histogram_bounds.wgsl
var histogramBoundsWGSL string

//go:embed shaders/balance_rgb.wgsl
var balanceRGBWGSL string

//go:embed shaders/block_thresholds.wgsl
var blockThresholdsWGSL string

const (
	gpuRGBHistogramBytes = 3 * 256 * 4
	gpuRGBBoundsBytes    = 8 * 4
)

type gpuResidentInputBindings struct {
	histogram *vulki.BindingSet
	balance   *vulki.BindingSet
}

// gpuResidentBinarizer consumes an image buffer that already belongs to its
// borrowed device. Histogram balancing, scale-adaptive block statistics and
// the fused classifier/filter/packer remain on that device; only packed masks
// cross back to the host.
type gpuResidentBinarizer struct {
	mu     sync.Mutex
	closed bool

	device    *vulki.Device
	binarizer *gpuBinarizer

	histogram *vulki.Buffer
	bounds    *vulki.Buffer
	balanced  *vulki.Buffer

	histogramKernel *vulki.Kernel
	boundsKernel    *vulki.Kernel
	balanceKernel   *vulki.Kernel
	blocksKernel    *vulki.Kernel

	boundsBindings     *vulki.BindingSet
	blocksBindings     *vulki.BindingSet
	classifierBindings *vulki.BindingSet
	inputBindings      map[*vulki.Buffer]gpuResidentInputBindings
}

func newGPUResidentBinarizerWithDevice(
	device *vulki.Device,
	maxWidth, maxHeight int,
) (*gpuResidentBinarizer, error) {
	binarizer, err := newGPUBinarizerPipelineWithDevice(device, maxWidth, maxHeight, false)
	if err != nil {
		return nil, err
	}
	resident := &gpuResidentBinarizer{
		device:        device,
		binarizer:     binarizer,
		inputBindings: make(map[*vulki.Buffer]gpuResidentInputBindings),
	}
	if err := resident.initialize(); err != nil {
		_ = resident.closeResources()
		return nil, err
	}
	return resident, nil
}

func (resident *gpuResidentBinarizer) initialize() error {
	area := uint64(resident.binarizer.maxWidth) * uint64(resident.binarizer.maxHeight)
	var err error
	resident.histogram, err = resident.device.NewBuffer(gpuRGBHistogramBytes)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU RGB histogram: %w", err)
	}
	resident.bounds, err = resident.device.NewBuffer(gpuRGBBoundsBytes)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU RGB bounds: %w", err)
	}
	resident.balanced, err = resident.device.NewBuffer(area * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU balanced image: %w", err)
	}

	resident.histogramKernel, err = resident.device.NewKernel(vulki.KernelOptions{
		WGSL: histogramRGBWGSL,
		Bindings: []vulki.BindingLayout{
			{Binding: 0, Access: vulki.BufferReadOnly},
			{Binding: 1, Access: vulki.BufferReadWrite},
			{Binding: 2, Access: vulki.BufferReadOnly},
		},
	})
	if err != nil {
		return fmt.Errorf("jabcode: create resident GPU RGB histogram kernel: %w", err)
	}
	resident.boundsKernel, err = resident.device.NewKernel(vulki.KernelOptions{
		WGSL: histogramBoundsWGSL,
		Bindings: []vulki.BindingLayout{
			{Binding: 0, Access: vulki.BufferReadWrite},
			{Binding: 1, Access: vulki.BufferReadWrite},
		},
	})
	if err != nil {
		return fmt.Errorf("jabcode: create resident GPU histogram-bounds kernel: %w", err)
	}
	resident.balanceKernel, err = resident.device.NewKernel(vulki.KernelOptions{
		WGSL: balanceRGBWGSL,
		Bindings: []vulki.BindingLayout{
			{Binding: 0, Access: vulki.BufferReadOnly},
			{Binding: 1, Access: vulki.BufferReadWrite},
			{Binding: 2, Access: vulki.BufferReadOnly},
			{Binding: 3, Access: vulki.BufferReadOnly},
		},
	})
	if err != nil {
		return fmt.Errorf("jabcode: create resident GPU RGB balance kernel: %w", err)
	}
	resident.blocksKernel, err = resident.device.NewKernel(vulki.KernelOptions{
		WGSL: blockThresholdsWGSL,
		Bindings: []vulki.BindingLayout{
			{Binding: 0, Access: vulki.BufferReadOnly},
			{Binding: 1, Access: vulki.BufferReadWrite},
			{Binding: 2, Access: vulki.BufferReadOnly},
		},
	})
	if err != nil {
		return fmt.Errorf("jabcode: create resident GPU block-threshold kernel: %w", err)
	}

	resident.boundsBindings, err = resident.boundsKernel.NewBindings(
		vulki.BindBuffer(0, resident.histogram),
		vulki.BindBuffer(1, resident.bounds),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU histogram bounds: %w", err)
	}
	resident.blocksBindings, err = resident.blocksKernel.NewBindings(
		vulki.BindBuffer(0, resident.balanced),
		vulki.BindBuffer(1, resident.binarizer.thresholds),
		vulki.BindBuffer(2, resident.binarizer.params),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU block thresholds: %w", err)
	}
	resident.classifierBindings, err = resident.binarizer.classify.kernel.NewBindings(
		vulki.BindBuffer(0, resident.balanced),
		vulki.BindBuffer(1, resident.binarizer.thresholds),
		vulki.BindBuffer(2, resident.binarizer.rawMasks),
		vulki.BindBuffer(3, resident.binarizer.params),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU RGB classifier: %w", err)
	}
	return nil
}

func (resident *gpuResidentBinarizer) bindingsFor(
	input *vulki.Buffer,
) (gpuResidentInputBindings, error) {
	if bindings, ok := resident.inputBindings[input]; ok {
		return bindings, nil
	}
	var bindings gpuResidentInputBindings
	var err error
	bindings.histogram, err = resident.histogramKernel.NewBindings(
		vulki.BindBuffer(0, input),
		vulki.BindBuffer(1, resident.histogram),
		vulki.BindBuffer(2, resident.binarizer.params),
	)
	if err != nil {
		return bindings, fmt.Errorf("jabcode: bind resident GPU RGB histogram input: %w", err)
	}
	bindings.balance, err = resident.balanceKernel.NewBindings(
		vulki.BindBuffer(0, input),
		vulki.BindBuffer(1, resident.balanced),
		vulki.BindBuffer(2, resident.bounds),
		vulki.BindBuffer(3, resident.binarizer.params),
	)
	if err != nil {
		_ = bindings.histogram.Close()
		return gpuResidentInputBindings{}, fmt.Errorf("jabcode: bind resident GPU RGB balance input: %w", err)
	}
	resident.inputBindings[input] = bindings
	return bindings, nil
}

func (resident *gpuResidentBinarizer) Binarize(
	input *vulki.Buffer,
	width, height int,
	blkThs []float32,
	printLevels bool,
) ([3]*core.Bitmap, error) {
	var empty [3]*core.Bitmap
	if resident == nil {
		return empty, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	resident.mu.Lock()
	defer resident.mu.Unlock()
	if resident.closed || resident.device == nil || resident.device.Closed() || resident.binarizer == nil {
		return empty, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	if width <= 0 || height <= 0 || width > resident.binarizer.maxWidth || height > resident.binarizer.maxHeight {
		return empty, fmt.Errorf(
			"jabcode: resident GPU image %dx%d exceeds configured maximum %dx%d",
			width, height, resident.binarizer.maxWidth, resident.binarizer.maxHeight,
		)
	}
	pixelCount := width * height
	if input == nil || input.Size() < uint64(pixelCount)*4 {
		return empty, fmt.Errorf("jabcode: resident GPU input buffer is too small")
	}
	if blkThs != nil && len(blkThs) < 3 {
		return empty, fmt.Errorf("jabcode: resident GPU binarizer needs three fixed thresholds")
	}
	bindings, err := resident.bindingsFor(input)
	if err != nil {
		return empty, err
	}
	params, blocksX, blocksY := gpuResidentBinarizerParams(width, height, blkThs, printLevels)
	packedMasks := resident.binarizer.hostMasks[:((pixelCount+7)/8)*4]
	var zeroHistogram [gpuRGBHistogramBytes]byte
	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return empty, fmt.Errorf("jabcode: create resident GPU binarizer recorder: %w", err)
	}
	defer recorder.Abort()
	if err := recorder.Update(resident.histogram, 0, zeroHistogram[:]); err != nil {
		return empty, fmt.Errorf("jabcode: clear resident GPU RGB histogram: %w", err)
	}
	if err := recorder.Update(resident.binarizer.params, 0, params[:]); err != nil {
		return empty, fmt.Errorf("jabcode: update resident GPU binarizer parameters: %w", err)
	}
	pixelGroups := gpuCanvasWorkgroups(width, height)
	if err := recorder.Dispatch(resident.histogramKernel, bindings.histogram, pixelGroups); err != nil {
		return empty, fmt.Errorf("jabcode: dispatch resident GPU RGB histogram: %w", err)
	}
	if err := recorder.Barrier(resident.histogram); err != nil {
		return empty, fmt.Errorf("jabcode: synchronize resident GPU RGB histogram: %w", err)
	}
	if err := recorder.Dispatch(
		resident.boundsKernel,
		resident.boundsBindings,
		vulki.Workgroups{X: 1, Y: 1, Z: 1},
	); err != nil {
		return empty, fmt.Errorf("jabcode: dispatch resident GPU histogram bounds: %w", err)
	}
	if err := recorder.Barrier(resident.bounds); err != nil {
		return empty, fmt.Errorf("jabcode: synchronize resident GPU histogram bounds: %w", err)
	}
	if err := recorder.Dispatch(resident.balanceKernel, bindings.balance, pixelGroups); err != nil {
		return empty, fmt.Errorf("jabcode: dispatch resident GPU RGB balance: %w", err)
	}
	if err := recorder.Barrier(resident.balanced); err != nil {
		return empty, fmt.Errorf("jabcode: synchronize resident GPU RGB balance: %w", err)
	}
	if blkThs == nil {
		if err := recorder.Dispatch(
			resident.blocksKernel,
			resident.blocksBindings,
			vulki.Workgroups{X: uint32(blocksX), Y: uint32(blocksY), Z: 1},
		); err != nil {
			return empty, fmt.Errorf("jabcode: dispatch resident GPU block thresholds: %w", err)
		}
		if err := recorder.Barrier(resident.binarizer.thresholds); err != nil {
			return empty, fmt.Errorf("jabcode: synchronize resident GPU block thresholds: %w", err)
		}
	}
	if err := resident.binarizer.recordComputeWithClassifier(
		recorder,
		resident.classifierBindings,
		width,
		height,
	); err != nil {
		return empty, err
	}
	if err := recorder.Download(resident.binarizer.packedMasks, 0, packedMasks); err != nil {
		return empty, fmt.Errorf("jabcode: download resident GPU binarizer masks: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return empty, fmt.Errorf("jabcode: run resident GPU binarizer: %w", err)
	}
	shape := core.Bitmap{Width: width, Height: height}
	return unpackGPUBinarizerMasks(&shape, packedMasks), nil
}

func gpuResidentBinarizerParams(
	width, height int,
	blkThs []float32,
	printLevels bool,
) (params [gpuBinarizerParamsSize]byte, blocksX, blocksY int) {
	binary.LittleEndian.PutUint32(params[0:], uint32(width))
	binary.LittleEndian.PutUint32(params[4:], uint32(height))
	flags := uint32(0)
	if blkThs == nil {
		blockSize := capInt(min(width, height)/binThresholdDivisor, binMinBlock, binMaxBlock)
		blocksX = (width + blockSize - 1) / blockSize
		blocksY = (height + blockSize - 1) / blockSize
		binary.LittleEndian.PutUint32(params[8:], uint32(blockSize))
		binary.LittleEndian.PutUint32(params[12:], uint32(blocksX))
		binary.LittleEndian.PutUint32(params[16:], uint32(blocksY))
	} else {
		flags |= 1
		blocksX, blocksY = 1, 1
		binary.LittleEndian.PutUint32(params[8:], 1)
		binary.LittleEndian.PutUint32(params[12:], 1)
		binary.LittleEndian.PutUint32(params[16:], 1)
		binary.LittleEndian.PutUint32(params[24:], math.Float32bits(blkThs[0]))
		binary.LittleEndian.PutUint32(params[28:], math.Float32bits(blkThs[1]))
		binary.LittleEndian.PutUint32(params[32:], math.Float32bits(blkThs[2]))
	}
	if printLevels {
		flags |= 2
	}
	binary.LittleEndian.PutUint32(params[20:], flags)
	return params, blocksX, blocksY
}

func (resident *gpuResidentBinarizer) Close() error {
	if resident == nil {
		return nil
	}
	resident.mu.Lock()
	defer resident.mu.Unlock()
	if resident.closed {
		return nil
	}
	resident.closed = true
	return resident.closeResources()
}

func (resident *gpuResidentBinarizer) closeResources() error {
	var closeErrors []error
	for input, bindings := range resident.inputBindings {
		closeErrors = append(closeErrors, bindings.balance.Close(), bindings.histogram.Close())
		delete(resident.inputBindings, input)
	}
	for _, bindings := range []*vulki.BindingSet{
		resident.classifierBindings,
		resident.blocksBindings,
		resident.boundsBindings,
	} {
		if bindings != nil {
			closeErrors = append(closeErrors, bindings.Close())
		}
	}
	resident.classifierBindings = nil
	resident.blocksBindings = nil
	resident.boundsBindings = nil
	for _, kernel := range []*vulki.Kernel{
		resident.blocksKernel,
		resident.balanceKernel,
		resident.boundsKernel,
		resident.histogramKernel,
	} {
		if kernel != nil {
			closeErrors = append(closeErrors, kernel.Close())
		}
	}
	resident.blocksKernel = nil
	resident.balanceKernel = nil
	resident.boundsKernel = nil
	resident.histogramKernel = nil
	for _, buffer := range []*vulki.Buffer{resident.balanced, resident.bounds, resident.histogram} {
		if buffer != nil {
			closeErrors = append(closeErrors, buffer.Close())
		}
	}
	resident.balanced = nil
	resident.bounds = nil
	resident.histogram = nil
	if resident.binarizer != nil {
		closeErrors = append(closeErrors, resident.binarizer.Close())
		resident.binarizer = nil
	}
	resident.device = nil
	return errors.Join(closeErrors...)
}
