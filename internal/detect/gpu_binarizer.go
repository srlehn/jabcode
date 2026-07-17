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

//go:embed shaders/binarize_rgb.wgsl
var binarizeRGBWGSL string

//go:embed shaders/filter_binary.wgsl
var filterBinaryWGSL string

//go:embed shaders/pack_binary_masks.wgsl
var packBinaryMasksWGSL string

//go:embed shaders/finder_row_scan.wgsl
var finderRowScanWGSL string

const (
	gpuBinarizerWorkgroupWidth  = 8
	gpuBinarizerWorkgroupHeight = 8
	gpuPackWorkgroupSize        = 64
	gpuBinarizerParamsSize      = 48
	gpuThresholdCellSize        = 32

	// gpuFinderScanCapacity bounds the raw-hit records one scan pass may
	// return. Measured passes seed hundreds to a few thousand raw hits even
	// on print retries; a pass overflowing this generous bound falls back to
	// the CPU row walk, so the cap risks throughput, never correctness.
	gpuFinderScanCapacity    = 65536
	gpuFinderScanRecordWords = 8
	gpuFinderScanHeaderBytes = 16
	gpuFinderScanParamsSize  = 16
	gpuFinderScanBufferBytes = gpuFinderScanHeaderBytes +
		gpuFinderScanCapacity*gpuFinderScanRecordWords*4
	gpuFinderScanWorkgroupSize = 64
)

type gpuBinarizerStage struct {
	kernel   *vulki.Kernel
	bindings *vulki.BindingSet
}

// gpuBinarizer is a measurement surface for the fused RGB classification and
// binary-filter chain. Its buffers and bindings are reused across calls; the
// compute kernels come from a shared per-device set so concurrent route
// contexts do not recompile WGSL. It is deliberately internal until parity
// and transfer measurements establish a useful integration boundary.
type gpuBinarizer struct {
	mu sync.Mutex

	device      *vulki.Device
	kernels     *gpuDecodeKernels
	ownsKernels bool
	ownsDevice  bool
	closed      bool
	maxWidth    int
	maxHeight   int

	input       *vulki.Buffer
	thresholds  *vulki.Buffer
	rawMasks    *vulki.Buffer
	finalMasks  *vulki.Buffer
	packedMasks *vulki.Buffer
	params      *vulki.Buffer
	hostMasks   []byte

	scanRecords     *vulki.Buffer
	scanParams      *vulki.Buffer
	hostScanRecords []byte

	classify gpuBinarizerStage
	filter   gpuBinarizerStage
	pack     gpuBinarizerStage
	scan     gpuBinarizerStage
}

func newGPUBinarizer(maxWidth, maxHeight int) (*gpuBinarizer, error) {
	device, err := vulki.Open()
	if err != nil {
		return nil, fmt.Errorf("jabcode: open GPU binarizer device: %w", err)
	}
	kernels := newGPUDecodeKernels(device)
	binarizer, err := newGPUBinarizerPipelineWithDevice(device, kernels, maxWidth, maxHeight, true)
	if err != nil {
		_ = kernels.Close()
		_ = device.Close()
		return nil, err
	}
	binarizer.ownsDevice = true
	binarizer.ownsKernels = true
	return binarizer, nil
}

func newGPUBinarizerPipelineWithDevice(
	device *vulki.Device,
	kernels *gpuDecodeKernels,
	maxWidth, maxHeight int,
	hostInput bool,
) (*gpuBinarizer, error) {
	if device == nil || device.Closed() {
		return nil, fmt.Errorf("jabcode: GPU binarizer device is closed")
	}
	if maxWidth <= 0 || maxHeight <= 0 {
		return nil, fmt.Errorf("jabcode: GPU binarizer dimensions must be positive")
	}
	area := uint64(maxWidth) * uint64(maxHeight)
	if area > uint64(^uint32(0)) || area > uint64(int(^uint(0)>>1)) {
		return nil, fmt.Errorf("jabcode: GPU binarizer image area exceeds shader limits")
	}

	b := &gpuBinarizer{device: device, kernels: kernels, maxWidth: maxWidth, maxHeight: maxHeight}
	if err := b.initialize(hostInput); err != nil {
		_ = b.closeResources()
		return nil, err
	}
	return b, nil
}

func (b *gpuBinarizer) initialize(hostInput bool) error {
	area := uint64(b.maxWidth) * uint64(b.maxHeight)
	maxBlocksX := (b.maxWidth + binMinBlock - 1) / binMinBlock
	maxBlocksY := (b.maxHeight + binMinBlock - 1) / binMinBlock
	thresholdBytes := uint64(maxBlocksX) * uint64(maxBlocksY) * gpuThresholdCellSize

	var err error
	if hostInput {
		b.input, err = b.device.NewBuffer(area * 4)
		if err != nil {
			return fmt.Errorf("jabcode: allocate GPU input: %w", err)
		}
	}
	b.thresholds, err = b.device.NewBuffer(thresholdBytes)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU thresholds: %w", err)
	}
	b.rawMasks, err = b.device.NewBuffer(area * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU raw masks: %w", err)
	}
	b.finalMasks, err = b.device.NewBuffer(area * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU final masks: %w", err)
	}
	packedWords := (area + 7) / 8
	b.packedMasks, err = b.device.NewBuffer(packedWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU packed masks: %w", err)
	}
	b.hostMasks = make([]byte, packedWords*4)
	b.params, err = b.device.NewBuffer(gpuBinarizerParamsSize)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU parameters: %w", err)
	}

	b.classify.kernel, err = b.kernels.classifyRGB()
	if err != nil {
		return err
	}
	if hostInput {
		b.classify.bindings, err = b.classify.kernel.NewBindings(
			vulki.BindBuffer(0, b.input),
			vulki.BindBuffer(1, b.thresholds),
			vulki.BindBuffer(2, b.rawMasks),
			vulki.BindBuffer(3, b.params),
		)
		if err != nil {
			return fmt.Errorf("jabcode: bind GPU RGB classifier: %w", err)
		}
	}
	b.filter, err = b.newStage(
		b.kernels.filterBinary,
		vulki.BindBuffer(0, b.rawMasks),
		vulki.BindBuffer(1, b.finalMasks),
		vulki.BindBuffer(2, b.params),
	)
	if err != nil {
		return fmt.Errorf("jabcode: create GPU binary filter: %w", err)
	}
	b.pack, err = b.newStage(
		b.kernels.packMasks,
		vulki.BindBuffer(0, b.finalMasks),
		vulki.BindBuffer(1, b.packedMasks),
		vulki.BindBuffer(2, b.params),
	)
	if err != nil {
		return fmt.Errorf("jabcode: create GPU mask packer: %w", err)
	}
	b.scanRecords, err = b.device.NewBuffer(gpuFinderScanBufferBytes)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU finder scan records: %w", err)
	}
	b.hostScanRecords = make([]byte, gpuFinderScanBufferBytes)
	b.scanParams, err = b.device.NewBuffer(gpuFinderScanParamsSize)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU finder scan parameters: %w", err)
	}
	b.scan, err = b.newStage(
		b.kernels.finderRowScan,
		vulki.BindBuffer(0, b.packedMasks),
		vulki.BindBuffer(1, b.scanRecords),
		vulki.BindBuffer(2, b.scanParams),
	)
	if err != nil {
		return fmt.Errorf("jabcode: create GPU finder row scan: %w", err)
	}
	return nil
}

// recordFinderScan appends the packed-mask row scan for the requested channel
// mask to a recording whose mask packer already ran, and downloads the
// compact hit records. The caller parses them with parseFinderScanRecords.
func (b *gpuBinarizer) recordFinderScan(
	recorder *vulki.Recorder,
	width, height int,
	channelMask uint32,
) error {
	var params [gpuFinderScanParamsSize]byte
	binary.LittleEndian.PutUint32(params[0:], uint32(width))
	binary.LittleEndian.PutUint32(params[4:], uint32(height))
	binary.LittleEndian.PutUint32(params[8:], channelMask)
	binary.LittleEndian.PutUint32(params[12:], gpuFinderScanCapacity)
	if err := recorder.Update(b.scanParams, 0, params[:]); err != nil {
		return fmt.Errorf("jabcode: update GPU finder scan parameters: %w", err)
	}
	var header [gpuFinderScanHeaderBytes]byte
	if err := recorder.Update(b.scanRecords, 0, header[:]); err != nil {
		return fmt.Errorf("jabcode: clear GPU finder scan counter: %w", err)
	}
	if err := recorder.Barrier(b.packedMasks); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU packed masks for the finder scan: %w", err)
	}
	groups := vulki.Workgroups{
		X: uint32((height + gpuFinderScanWorkgroupSize - 1) / gpuFinderScanWorkgroupSize),
		Y: 1,
		Z: 1,
	}
	if err := recorder.Dispatch(b.scan.kernel, b.scan.bindings, groups); err != nil {
		return fmt.Errorf("jabcode: dispatch GPU finder row scan: %w", err)
	}
	if err := recorder.Barrier(b.scanRecords); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU finder scan records: %w", err)
	}
	if err := recorder.Download(b.scanRecords, 0, b.hostScanRecords); err != nil {
		return fmt.Errorf("jabcode: download GPU finder scan records: %w", err)
	}
	return nil
}

func (b *gpuBinarizer) newStage(
	kernel func() (*vulki.Kernel, error),
	buffers ...vulki.BufferBinding,
) (gpuBinarizerStage, error) {
	shared, err := kernel()
	if err != nil {
		return gpuBinarizerStage{}, err
	}
	bindings, err := shared.NewBindings(buffers...)
	if err != nil {
		return gpuBinarizerStage{}, err
	}
	return gpuBinarizerStage{kernel: shared, bindings: bindings}, nil
}

func (b *gpuBinarizer) Binarize(bm *core.Bitmap, blkThs []float32, printLevels bool) ([3]*core.Bitmap, error) {
	var empty [3]*core.Bitmap
	if bm == nil || bm.Width <= 0 || bm.Height <= 0 || bm.Channels != 4 {
		return empty, fmt.Errorf("jabcode: GPU binarizer requires a non-empty packed RGBA bitmap")
	}
	if bm.Width > b.maxWidth || bm.Height > b.maxHeight {
		return empty, fmt.Errorf("jabcode: GPU binarizer image %dx%d exceeds configured maximum %dx%d", bm.Width, bm.Height, b.maxWidth, b.maxHeight)
	}
	pixelCount := bm.Width * bm.Height
	if len(bm.Pix) != pixelCount*4 {
		return empty, fmt.Errorf("jabcode: GPU binarizer requires a non-empty packed RGBA bitmap")
	}
	if blkThs != nil && len(blkThs) < 3 {
		return empty, fmt.Errorf("jabcode: GPU binarizer needs three fixed thresholds")
	}

	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed || b.device == nil || b.device.Closed() {
		return empty, fmt.Errorf("jabcode: GPU binarizer is closed")
	}
	if b.input == nil || b.classify.bindings == nil {
		return empty, fmt.Errorf("jabcode: GPU binarizer has no host-input path")
	}

	params, thresholdData := gpuBinarizerInputs(bm, blkThs, printLevels)
	packedMasks := b.hostMasks[:((pixelCount+7)/8)*4]
	recorder, err := b.device.NewRecorder()
	if err != nil {
		return empty, fmt.Errorf("jabcode: create GPU binarizer recorder: %w", err)
	}
	defer recorder.Abort()
	if err := recorder.Upload(b.input, 0, bm.Pix); err != nil {
		return empty, fmt.Errorf("jabcode: upload GPU binarizer image: %w", err)
	}
	if err := recorder.Upload(b.thresholds, 0, thresholdData); err != nil {
		return empty, fmt.Errorf("jabcode: upload GPU binarizer thresholds: %w", err)
	}
	if err := recorder.Update(b.params, 0, params); err != nil {
		return empty, fmt.Errorf("jabcode: update GPU binarizer parameters: %w", err)
	}
	if err := b.recordCompute(recorder, bm.Width, bm.Height); err != nil {
		return empty, err
	}
	if err := recorder.Download(b.packedMasks, 0, packedMasks); err != nil {
		return empty, fmt.Errorf("jabcode: download GPU binarizer masks: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return empty, fmt.Errorf("jabcode: run GPU binarizer: %w", err)
	}
	return unpackGPUBinarizerMasks(bm, packedMasks), nil
}

func unpackGPUBinarizerMasks(bm *core.Bitmap, packedMasks []byte) [3]*core.Bitmap {
	pixelCount := bm.Width * bm.Height
	var rgb [3]*core.Bitmap
	for channel := range rgb {
		rgb[channel] = newBinary(bm)
	}
	wordCount := (pixelCount + 7) / 8
	core.ParallelChunks(wordCount, 1024, func(lo, hi int) {
		pixel := lo * 8
		for word := lo; word < hi; word++ {
			packed := binary.LittleEndian.Uint32(packedMasks[word*4:])
			for lane := 0; lane < 8 && pixel < pixelCount; lane++ {
				mask := packed & 7
				rgb[0].Pix[pixel] = b2byte(mask&1 != 0)
				rgb[1].Pix[pixel] = b2byte(mask&2 != 0)
				rgb[2].Pix[pixel] = b2byte(mask&4 != 0)
				packed >>= 3
				pixel++
			}
		}
	})
	return rgb
}

func (b *gpuBinarizer) recordCompute(recorder *vulki.Recorder, width, height int) error {
	return b.recordComputeWithClassifier(recorder, b.classify.bindings, width, height)
}

func (b *gpuBinarizer) recordComputeWithClassifier(
	recorder *vulki.Recorder,
	classifier *vulki.BindingSet,
	width, height int,
) error {
	pixelCount := width * height
	pixelGroups := vulki.Workgroups{
		X: uint32((width + gpuBinarizerWorkgroupWidth - 1) / gpuBinarizerWorkgroupWidth),
		Y: uint32((height + gpuBinarizerWorkgroupHeight - 1) / gpuBinarizerWorkgroupHeight),
		Z: 1,
	}
	if err := recorder.Dispatch(b.classify.kernel, classifier, pixelGroups); err != nil {
		return fmt.Errorf("jabcode: dispatch GPU RGB classifier: %w", err)
	}
	if err := recorder.Barrier(b.rawMasks); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU RGB classifier: %w", err)
	}
	if err := recorder.Dispatch(b.filter.kernel, b.filter.bindings, pixelGroups); err != nil {
		return fmt.Errorf("jabcode: dispatch GPU binary filter: %w", err)
	}
	if err := recorder.Barrier(b.finalMasks); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU binary filter: %w", err)
	}
	packGroups := vulki.Workgroups{
		X: uint32(((pixelCount+7)/8 + gpuPackWorkgroupSize - 1) / gpuPackWorkgroupSize),
		Y: 1,
		Z: 1,
	}
	if err := recorder.Dispatch(b.pack.kernel, b.pack.bindings, packGroups); err != nil {
		return fmt.Errorf("jabcode: dispatch GPU mask packer: %w", err)
	}
	return nil
}

func gpuBinarizerInputs(bm *core.Bitmap, blkThs []float32, printLevels bool) (params, thresholds []byte) {
	params = make([]byte, gpuBinarizerParamsSize)
	binary.LittleEndian.PutUint32(params[0:], uint32(bm.Width))
	binary.LittleEndian.PutUint32(params[4:], uint32(bm.Height))
	flags := uint32(0)
	if blkThs != nil {
		flags |= 1
		binary.LittleEndian.PutUint32(params[24:], math.Float32bits(blkThs[0]))
		binary.LittleEndian.PutUint32(params[28:], math.Float32bits(blkThs[1]))
		binary.LittleEndian.PutUint32(params[32:], math.Float32bits(blkThs[2]))
		binary.LittleEndian.PutUint32(params[8:], 1)
		binary.LittleEndian.PutUint32(params[12:], 1)
		binary.LittleEndian.PutUint32(params[16:], 1)
		thresholds = make([]byte, gpuThresholdCellSize)
	} else {
		bs := capInt(min(bm.Width, bm.Height)/binThresholdDivisor, binMinBlock, binMaxBlock)
		anchors, means, blocksX, blocksY := blockThresholds(bm, bs)
		binary.LittleEndian.PutUint32(params[8:], uint32(bs))
		binary.LittleEndian.PutUint32(params[12:], uint32(blocksX))
		binary.LittleEndian.PutUint32(params[16:], uint32(blocksY))
		thresholds = make([]byte, len(means)*gpuThresholdCellSize)
		for index := range means {
			for channel := range 3 {
				binary.LittleEndian.PutUint32(thresholds[index*gpuThresholdCellSize+channel*4:], math.Float32bits(float32(anchors[index][channel])))
				binary.LittleEndian.PutUint32(thresholds[index*gpuThresholdCellSize+16+channel*4:], math.Float32bits(float32(means[index][channel])))
			}
		}
	}
	if printLevels {
		flags |= 2
	}
	binary.LittleEndian.PutUint32(params[20:], flags)
	return params, thresholds
}

func (b *gpuBinarizer) Close() error {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil
	}
	b.closed = true
	return b.closeResources()
}

func (b *gpuBinarizer) closeResources() error {
	var closeErrors []error
	// The stage kernels belong to the shared per-device set; only the binding
	// sets are this instance's to close.
	for _, stage := range []*gpuBinarizerStage{&b.scan, &b.pack, &b.filter, &b.classify} {
		if stage.bindings != nil {
			closeErrors = append(closeErrors, stage.bindings.Close())
			stage.bindings = nil
		}
		stage.kernel = nil
	}
	for _, buffer := range []*vulki.Buffer{
		b.scanParams, b.scanRecords,
		b.params, b.packedMasks, b.finalMasks, b.rawMasks, b.thresholds, b.input,
	} {
		if buffer != nil {
			closeErrors = append(closeErrors, buffer.Close())
		}
	}
	b.scanParams = nil
	b.scanRecords = nil
	b.hostScanRecords = nil
	b.params = nil
	b.packedMasks = nil
	b.finalMasks = nil
	b.rawMasks = nil
	b.thresholds = nil
	b.input = nil
	b.hostMasks = nil
	if b.ownsKernels {
		closeErrors = append(closeErrors, b.kernels.Close())
	}
	b.kernels = nil
	if b.ownsDevice && b.device != nil {
		closeErrors = append(closeErrors, b.device.Close())
	}
	b.device = nil
	return errors.Join(closeErrors...)
}
