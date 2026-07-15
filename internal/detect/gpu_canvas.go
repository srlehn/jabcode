//go:build jabcode_gpu

package detect

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"math"
	"sync"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
)

//go:embed shaders/halve_nrgba.wgsl
var halveNRGBAWGSL string

//go:embed shaders/rotate_nrgba.wgsl
var rotateNRGBAWGSL string

const gpuCanvasParamsSize = 48

type gpuCanvasLevel struct {
	width  int
	height int
	buffer *vulki.Buffer
}

// gpuCanvasLadder is the resident-image measurement surface for the decode
// ladder. One upload builds every requested half-resolution level. Rotations
// and ROI rotations then read those retained levels and reuse one grow-only
// route buffer, so no prepared canvas crosses the host boundary unless a test
// or diagnostic explicitly downloads it.
type gpuCanvasLadder struct {
	mu sync.Mutex

	device     *vulki.Device
	ownsDevice bool
	closed     bool

	params         *vulki.Buffer
	levels         []gpuCanvasLevel
	halveKernel    *vulki.Kernel
	halveBindings  []*vulki.BindingSet
	rotateKernel   *vulki.Kernel
	rotateBindings []*vulki.BindingSet
	route          *vulki.Buffer
	routeCapacity  uint64
	routeWidth     int
	routeHeight    int
}

func newGPUCanvasLadder(width, height, levelCount int) (*gpuCanvasLadder, error) {
	device, err := vulki.Open()
	if err != nil {
		return nil, fmt.Errorf("jabcode: open GPU canvas device: %w", err)
	}
	ladder, err := newGPUCanvasLadderWithDevice(device, width, height, levelCount)
	if err != nil {
		_ = device.Close()
		return nil, err
	}
	ladder.ownsDevice = true
	return ladder, nil
}

func newGPUCanvasLadderWithDevice(
	device *vulki.Device,
	width, height, levelCount int,
) (*gpuCanvasLadder, error) {
	if device == nil || device.Closed() {
		return nil, fmt.Errorf("jabcode: GPU canvas device is closed")
	}
	if levelCount <= 0 {
		return nil, fmt.Errorf("jabcode: GPU canvas ladder needs at least one level")
	}
	if _, err := gpuCanvasArea(width, height); err != nil {
		return nil, err
	}

	ladder := &gpuCanvasLadder{device: device}
	ladder.levels = make([]gpuCanvasLevel, levelCount)
	w, h := width, height
	for index := range ladder.levels {
		ladder.levels[index].width = w
		ladder.levels[index].height = h
		if index+1 < levelCount {
			w = max((w+1)/2, 1)
			h = max((h+1)/2, 1)
		}
	}
	if err := ladder.initialize(); err != nil {
		_ = ladder.closeResources()
		return nil, err
	}
	return ladder, nil
}

func gpuCanvasArea(width, height int) (uint64, error) {
	if width <= 0 || height <= 0 {
		return 0, fmt.Errorf("jabcode: GPU canvas dimensions must be positive")
	}
	w, h := uint64(width), uint64(height)
	if w > math.MaxUint32 || h > math.MaxUint32 || w*h > math.MaxUint32 {
		return 0, fmt.Errorf("jabcode: GPU canvas area exceeds shader limits")
	}
	if w*h > uint64(int(^uint(0)>>1)) {
		return 0, fmt.Errorf("jabcode: GPU canvas area exceeds platform limits")
	}
	return w * h, nil
}

func (ladder *gpuCanvasLadder) initialize() error {
	var err error
	ladder.params, err = ladder.device.NewBuffer(gpuCanvasParamsSize)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU canvas parameters: %w", err)
	}
	for index := range ladder.levels {
		level := &ladder.levels[index]
		area, _ := gpuCanvasArea(level.width, level.height)
		level.buffer, err = ladder.device.NewBuffer(area * 4)
		if err != nil {
			return fmt.Errorf("jabcode: allocate GPU canvas level %d: %w", index, err)
		}
	}

	ladder.halveKernel, err = ladder.device.NewKernel(vulki.KernelOptions{
		WGSL: halveNRGBAWGSL,
		Bindings: []vulki.BindingLayout{
			{Binding: 0, Access: vulki.BufferReadOnly},
			{Binding: 1, Access: vulki.BufferReadWrite},
			{Binding: 2, Access: vulki.BufferReadOnly},
		},
	})
	if err != nil {
		return fmt.Errorf("jabcode: create GPU canvas half-scale kernel: %w", err)
	}
	ladder.halveBindings = make([]*vulki.BindingSet, len(ladder.levels)-1)
	for index := range ladder.halveBindings {
		ladder.halveBindings[index], err = ladder.halveKernel.NewBindings(
			vulki.BindBuffer(0, ladder.levels[index].buffer),
			vulki.BindBuffer(1, ladder.levels[index+1].buffer),
			vulki.BindBuffer(2, ladder.params),
		)
		if err != nil {
			return fmt.Errorf("jabcode: bind GPU canvas half-scale level %d: %w", index, err)
		}
	}

	ladder.rotateKernel, err = ladder.device.NewKernel(vulki.KernelOptions{
		WGSL: rotateNRGBAWGSL,
		Bindings: []vulki.BindingLayout{
			{Binding: 0, Access: vulki.BufferReadOnly},
			{Binding: 1, Access: vulki.BufferReadWrite},
			{Binding: 2, Access: vulki.BufferReadOnly},
		},
	})
	if err != nil {
		return fmt.Errorf("jabcode: create GPU canvas rotation kernel: %w", err)
	}
	return nil
}

func (ladder *gpuCanvasLadder) UploadAndBuild(bm *core.Bitmap) error {
	if ladder == nil {
		return fmt.Errorf("jabcode: GPU canvas ladder is closed")
	}
	ladder.mu.Lock()
	defer ladder.mu.Unlock()
	if ladder.closed || ladder.device == nil || ladder.device.Closed() {
		return fmt.Errorf("jabcode: GPU canvas ladder is closed")
	}
	base := ladder.levels[0]
	if bm == nil || bm.Width != base.width || bm.Height != base.height || bm.Channels != 4 ||
		len(bm.Pix) != bm.Width*bm.Height*4 {
		return fmt.Errorf("jabcode: GPU canvas base does not match the configured packed RGBA level")
	}

	recorder, err := ladder.device.NewRecorder()
	if err != nil {
		return fmt.Errorf("jabcode: create GPU canvas build recorder: %w", err)
	}
	defer recorder.Abort()
	if err := recorder.Upload(base.buffer, 0, bm.Pix); err != nil {
		return fmt.Errorf("jabcode: upload GPU canvas base: %w", err)
	}
	for index, bindings := range ladder.halveBindings {
		source := ladder.levels[index]
		destination := ladder.levels[index+1]
		params := gpuHalveParams(source.width, source.height, destination.width, destination.height)
		if err := recorder.Update(ladder.params, 0, params[:]); err != nil {
			return fmt.Errorf("jabcode: update GPU half-scale parameters for level %d: %w", index, err)
		}
		groups := gpuCanvasWorkgroups(destination.width, destination.height)
		if err := recorder.Dispatch(ladder.halveKernel, bindings, groups); err != nil {
			return fmt.Errorf("jabcode: dispatch GPU half-scale level %d: %w", index, err)
		}
		if err := recorder.Barrier(destination.buffer); err != nil {
			return fmt.Errorf("jabcode: synchronize GPU half-scale level %d: %w", index, err)
		}
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return fmt.Errorf("jabcode: build GPU canvas ladder: %w", err)
	}
	return nil
}

func gpuHalveParams(sourceWidth, sourceHeight, destinationWidth, destinationHeight int) [gpuCanvasParamsSize]byte {
	var params [gpuCanvasParamsSize]byte
	binary.LittleEndian.PutUint32(params[0:], uint32(sourceWidth))
	binary.LittleEndian.PutUint32(params[4:], uint32(sourceHeight))
	binary.LittleEndian.PutUint32(params[8:], uint32(destinationWidth))
	binary.LittleEndian.PutUint32(params[12:], uint32(destinationHeight))
	return params
}

func gpuCanvasWorkgroups(width, height int) vulki.Workgroups {
	return vulki.Workgroups{
		X: uint32((width + gpuBinarizerWorkgroupWidth - 1) / gpuBinarizerWorkgroupWidth),
		Y: uint32((height + gpuBinarizerWorkgroupHeight - 1) / gpuBinarizerWorkgroupHeight),
		Z: 1,
	}
}

func (ladder *gpuCanvasLadder) DownloadLevel(index int) (*core.Bitmap, error) {
	if ladder == nil {
		return nil, fmt.Errorf("jabcode: GPU canvas ladder is closed")
	}
	ladder.mu.Lock()
	defer ladder.mu.Unlock()
	if ladder.closed || index < 0 || index >= len(ladder.levels) {
		return nil, fmt.Errorf("jabcode: invalid GPU canvas level %d", index)
	}
	level := ladder.levels[index]
	bm := core.NewBitmap(level.width, level.height, 4)
	if err := level.buffer.Download(bm.Pix); err != nil {
		return nil, fmt.Errorf("jabcode: download GPU canvas level %d: %w", index, err)
	}
	return bm, nil
}

func (ladder *gpuCanvasLadder) Rotate(levelIndex int, crop image.Rectangle, angle float64) (image.Point, error) {
	if ladder == nil {
		return image.Point{}, fmt.Errorf("jabcode: GPU canvas ladder is closed")
	}
	ladder.mu.Lock()
	defer ladder.mu.Unlock()
	if ladder.closed || levelIndex < 0 || levelIndex >= len(ladder.levels) {
		return image.Point{}, fmt.Errorf("jabcode: invalid GPU canvas level %d", levelIndex)
	}
	level := ladder.levels[levelIndex]
	bounds := image.Rect(0, 0, level.width, level.height)
	if crop.Empty() || crop.Intersect(bounds) != crop {
		return image.Point{}, fmt.Errorf("jabcode: GPU canvas crop %v exceeds level bounds %v", crop, bounds)
	}

	rad := angle * math.Pi / 180
	cosine, sine := math.Cos(rad), math.Sin(rad)
	width, height := crop.Dx(), crop.Dy()
	rotatedWidth := int(math.Ceil(math.Abs(float64(width)*cosine) + math.Abs(float64(height)*sine)))
	rotatedHeight := int(math.Ceil(math.Abs(float64(width)*sine) + math.Abs(float64(height)*cosine)))
	area, err := gpuCanvasArea(rotatedWidth, rotatedHeight)
	if err != nil {
		return image.Point{}, err
	}
	if err := ladder.ensureRouteLocked(area); err != nil {
		return image.Point{}, err
	}

	params := gpuRotateParams(level, crop, rotatedWidth, rotatedHeight, cosine, sine)
	recorder, err := ladder.device.NewRecorder()
	if err != nil {
		return image.Point{}, fmt.Errorf("jabcode: create GPU canvas rotation recorder: %w", err)
	}
	defer recorder.Abort()
	if err := recorder.Update(ladder.params, 0, params[:]); err != nil {
		return image.Point{}, fmt.Errorf("jabcode: update GPU canvas rotation parameters: %w", err)
	}
	if err := recorder.Dispatch(
		ladder.rotateKernel,
		ladder.rotateBindings[levelIndex],
		gpuCanvasWorkgroups(rotatedWidth, rotatedHeight),
	); err != nil {
		return image.Point{}, fmt.Errorf("jabcode: dispatch GPU canvas rotation: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return image.Point{}, fmt.Errorf("jabcode: rotate GPU canvas: %w", err)
	}
	ladder.routeWidth = rotatedWidth
	ladder.routeHeight = rotatedHeight
	return image.Pt(rotatedWidth, rotatedHeight), nil
}

func gpuRotateParams(
	level gpuCanvasLevel,
	crop image.Rectangle,
	rotatedWidth, rotatedHeight int,
	cosine, sine float64,
) [gpuCanvasParamsSize]byte {
	var params [gpuCanvasParamsSize]byte
	binary.LittleEndian.PutUint32(params[0:], uint32(level.width))
	binary.LittleEndian.PutUint32(params[4:], uint32(level.height))
	binary.LittleEndian.PutUint32(params[8:], uint32(crop.Min.X))
	binary.LittleEndian.PutUint32(params[12:], uint32(crop.Min.Y))
	binary.LittleEndian.PutUint32(params[16:], uint32(crop.Dx()))
	binary.LittleEndian.PutUint32(params[20:], uint32(crop.Dy()))
	binary.LittleEndian.PutUint32(params[24:], uint32(rotatedWidth))
	binary.LittleEndian.PutUint32(params[28:], uint32(rotatedHeight))
	binary.LittleEndian.PutUint32(params[32:], math.Float32bits(float32(cosine)))
	binary.LittleEndian.PutUint32(params[36:], math.Float32bits(float32(sine)))
	return params
}

func (ladder *gpuCanvasLadder) ensureRouteLocked(area uint64) error {
	if ladder.route != nil && ladder.routeCapacity >= area {
		return nil
	}
	newRoute, err := ladder.device.NewBuffer(area * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU route canvas: %w", err)
	}
	newBindings := make([]*vulki.BindingSet, len(ladder.levels))
	for index := range newBindings {
		newBindings[index], err = ladder.rotateKernel.NewBindings(
			vulki.BindBuffer(0, ladder.levels[index].buffer),
			vulki.BindBuffer(1, newRoute),
			vulki.BindBuffer(2, ladder.params),
		)
		if err != nil {
			for closeIndex := index - 1; closeIndex >= 0; closeIndex-- {
				_ = newBindings[closeIndex].Close()
			}
			_ = newRoute.Close()
			return fmt.Errorf("jabcode: bind GPU route canvas level %d: %w", index, err)
		}
	}
	if err := ladder.closeRouteLocked(); err != nil {
		for index := len(newBindings) - 1; index >= 0; index-- {
			_ = newBindings[index].Close()
		}
		_ = newRoute.Close()
		return err
	}
	ladder.route = newRoute
	ladder.routeCapacity = area
	ladder.rotateBindings = newBindings
	return nil
}

func (ladder *gpuCanvasLadder) DownloadRoute() (*core.Bitmap, error) {
	if ladder == nil {
		return nil, fmt.Errorf("jabcode: GPU canvas ladder is closed")
	}
	ladder.mu.Lock()
	defer ladder.mu.Unlock()
	if ladder.closed || ladder.route == nil || ladder.routeWidth <= 0 || ladder.routeHeight <= 0 {
		return nil, fmt.Errorf("jabcode: GPU route canvas is unavailable")
	}
	bm := core.NewBitmap(ladder.routeWidth, ladder.routeHeight, 4)
	if err := ladder.route.Download(bm.Pix); err != nil {
		return nil, fmt.Errorf("jabcode: download GPU route canvas: %w", err)
	}
	return bm, nil
}

func (ladder *gpuCanvasLadder) Close() error {
	if ladder == nil {
		return nil
	}
	ladder.mu.Lock()
	defer ladder.mu.Unlock()
	if ladder.closed {
		return nil
	}
	ladder.closed = true
	return ladder.closeResources()
}

func (ladder *gpuCanvasLadder) closeRouteLocked() error {
	var closeErrors []error
	for index := len(ladder.rotateBindings) - 1; index >= 0; index-- {
		if ladder.rotateBindings[index] != nil {
			closeErrors = append(closeErrors, ladder.rotateBindings[index].Close())
		}
	}
	ladder.rotateBindings = nil
	if ladder.route != nil {
		closeErrors = append(closeErrors, ladder.route.Close())
		ladder.route = nil
	}
	ladder.routeCapacity = 0
	ladder.routeWidth = 0
	ladder.routeHeight = 0
	return errors.Join(closeErrors...)
}

func (ladder *gpuCanvasLadder) closeResources() error {
	var closeErrors []error
	closeErrors = append(closeErrors, ladder.closeRouteLocked())
	for index := len(ladder.halveBindings) - 1; index >= 0; index-- {
		if ladder.halveBindings[index] != nil {
			closeErrors = append(closeErrors, ladder.halveBindings[index].Close())
		}
	}
	ladder.halveBindings = nil
	if ladder.rotateKernel != nil {
		closeErrors = append(closeErrors, ladder.rotateKernel.Close())
		ladder.rotateKernel = nil
	}
	if ladder.halveKernel != nil {
		closeErrors = append(closeErrors, ladder.halveKernel.Close())
		ladder.halveKernel = nil
	}
	for index := len(ladder.levels) - 1; index >= 0; index-- {
		if ladder.levels[index].buffer != nil {
			closeErrors = append(closeErrors, ladder.levels[index].buffer.Close())
			ladder.levels[index].buffer = nil
		}
	}
	ladder.levels = nil
	if ladder.params != nil {
		closeErrors = append(closeErrors, ladder.params.Close())
		ladder.params = nil
	}
	if ladder.ownsDevice && ladder.device != nil {
		closeErrors = append(closeErrors, ladder.device.Close())
	}
	ladder.device = nil
	return errors.Join(closeErrors...)
}
