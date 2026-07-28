//go:build !js

package detect

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
)

const gpuCanvasParamsSize = 48

type gpuCanvasLevel struct {
	width  int
	height int
	buffer *vulki.Buffer
}

// gpuCanvasLadder is the resident-image measurement surface for the decode
// ladder. One upload builds every requested half-resolution level; the levels
// are read-only afterwards, so concurrent route canvases rotate from them
// without coordination. UploadAndBuild remains exclusive with route work -
// the session quiesces its route contexts before rebuilding.
type gpuCanvasLadder struct {
	mu sync.Mutex

	device      *vulki.Device
	kernels     *gpuDecodeKernels
	ownsDevice  bool
	ownsKernels bool
	closed      bool

	params        *vulki.Buffer
	levels        []gpuCanvasLevel
	halveKernel   *vulki.Kernel
	halveBindings []*vulki.BindingSet
}

func newGPUCanvasLadder(width, height, levelCount int) (*gpuCanvasLadder, error) {
	device, err := vulki.Open()
	if err != nil {
		return nil, fmt.Errorf("jabcode: open GPU canvas device: %w", err)
	}
	kernels := newGPUDecodeKernels(device)
	ladder, err := newGPUCanvasLadderWithDevice(device, kernels, width, height, levelCount)
	if err != nil {
		_ = kernels.Close()
		_ = device.Close()
		return nil, err
	}
	ladder.ownsDevice = true
	ladder.ownsKernels = true
	return ladder, nil
}

func newGPUCanvasLadderWithDevice(
	device *vulki.Device,
	kernels *gpuDecodeKernels,
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

	ladder := &gpuCanvasLadder{device: device, kernels: kernels}
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
	if w*h > uint64(math.MaxInt) {
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

	ladder.halveKernel, err = ladder.kernels.halve()
	if err != nil {
		return err
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

// gpuRouteCanvas is one rotation target over a ladder's retained levels. Each
// route context owns one, so concurrent routes never share rotation
// parameters, output buffers or binding sets. It serves one route at a time
// and is not safe for concurrent use; the context pool enforces exclusivity.
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

func (ladder *gpuCanvasLadder) closeResources() error {
	var closeErrors []error
	for index := len(ladder.halveBindings) - 1; index >= 0; index-- {
		if ladder.halveBindings[index] != nil {
			closeErrors = append(closeErrors, ladder.halveBindings[index].Close())
		}
	}
	ladder.halveBindings = nil
	ladder.halveKernel = nil
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
	if ladder.ownsKernels {
		closeErrors = append(closeErrors, ladder.kernels.Close())
	}
	ladder.kernels = nil
	if ladder.ownsDevice && ladder.device != nil {
		closeErrors = append(closeErrors, ladder.device.Close())
	}
	ladder.device = nil
	return errors.Join(closeErrors...)
}
