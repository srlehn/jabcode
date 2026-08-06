//go:build !js

package detect

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"image"
	"math"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/phaseprobe"
)

//go:embed shaders/sample_symbol.wgsl
var sampleSymbolWGSL string

// gpuSampleMaxSide bounds the module grid the resident sampler can hold. A JAB
// side is at most SideSize's upper bound, so one allocation covers every
// symbol the decoder will ever be asked to sample.
const gpuSampleMaxSide = 145

// Parameter word indices, matching sample_symbol.wgsl.
const (
	gpuSampleParamWidth     = 0
	gpuSampleParamHeight    = 1
	gpuSampleParamSideX     = 2
	gpuSampleParamSideY     = 3
	gpuSampleParamRegime    = 4
	gpuSampleParamKX        = 5
	gpuSampleParamKY        = 6
	gpuSampleParamUseDelta  = 7
	gpuSampleParamTransform = 8
	gpuSampleParamDelta     = 17
	gpuSampleParamWords     = 24

	gpuSampleRegimeCentre    = 0
	gpuSampleRegimeFootprint = 1
)

// gpuSampleResultWords is the module grid plus the reject flag that shares its
// buffer, in words.
const gpuSampleResultWords = 1 + gpuSampleMaxSide*gpuSampleMaxSide

// initializeSampler allocates the module grid and compiles the sampler. It runs
// with the rest of the resident stage set rather than on first sample, so the
// compile lands in warm-up where there is idle time for it instead of on the
// decode call.
func (resident *gpuResidentBinarizer) initializeSampler() error {
	var err error
	resident.sampleResult, err = resident.device.NewBuffer(gpuSampleResultWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU module grid: %w", err)
	}
	resident.sampleParams, err = resident.device.NewBuffer(gpuSampleParamWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU sampler parameters: %w", err)
	}
	resident.sampleKernel, err = resident.kernels.sampleSymbol()
	if err != nil {
		return err
	}
	resident.sampleBindings, err = resident.sampleKernel.NewBindings(
		vulki.BindBuffer(0, resident.balanced),
		vulki.BindBuffer(1, resident.sampleResult),
		vulki.BindBuffer(2, resident.sampleParams),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU symbol sampler: %w", err)
	}
	return nil
}

// SampleSymbol reads the module grid straight off the resident balanced image,
// so the host receives the roughly 27 KB it decodes from instead of the whole
// prepared frame. It returns nil, nil when a module maps too far outside the
// image, which is the host sampler's own failure and not an error.
func (resident *gpuResidentBinarizer) SampleSymbol(
	width, height int,
	pt core.Perspective,
	side image.Point,
	delta [3]core.PointF,
) (*core.Bitmap, error) {
	if resident == nil || resident.closed || resident.sampleBindings == nil {
		return nil, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	if side.X <= 0 || side.Y <= 0 || side.X > gpuSampleMaxSide || side.Y > gpuSampleMaxSide {
		return nil, fmt.Errorf("jabcode: GPU sampler side %dx%d is out of range", side.X, side.Y)
	}
	if width <= 0 || height <= 0 ||
		width > resident.binarizer.maxWidth || height > resident.binarizer.maxHeight {
		return nil, fmt.Errorf("jabcode: GPU sampler dimensions are unavailable")
	}

	// The grid buffer is about to be overwritten, so whatever the payload chain
	// was allowed to classify stops being valid here rather than on success.
	resident.forgetSampledGrid()

	params := gpuSampleParams(width, height, pt, side, delta)
	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return nil, fmt.Errorf("jabcode: create GPU sampler recorder: %w", err)
	}
	defer recorder.Abort()
	if err := recorder.Update(resident.sampleParams, 0, params[:]); err != nil {
		return nil, fmt.Errorf("jabcode: update GPU sampler parameters: %w", err)
	}
	// The reject flag is only ever raised, never lowered, so it has to start
	// clear for this grid rather than carry the previous symbol's verdict.
	if err := recorder.Update(resident.sampleResult, 0, make([]byte, 4)); err != nil {
		return nil, fmt.Errorf("jabcode: clear GPU sampler reject flag: %w", err)
	}
	if err := recorder.Barrier(resident.sampleResult); err != nil {
		return nil, fmt.Errorf("jabcode: synchronize GPU sampler reset: %w", err)
	}
	modules := side.X * side.Y
	if err := recorder.Dispatch(
		resident.sampleKernel,
		resident.sampleBindings,
		vulki.Workgroups{X: uint32((modules + 63) / 64), Y: 1, Z: 1},
	); err != nil {
		return nil, fmt.Errorf("jabcode: dispatch GPU sampler: %w", err)
	}
	if err := recorder.Barrier(resident.sampleResult); err != nil {
		return nil, fmt.Errorf("jabcode: synchronize GPU module grid: %w", err)
	}
	result := make([]byte, (1+modules)*4)
	phaseprobe.Count("download.module_grid", len(result))
	if err := recorder.Download(resident.sampleResult, 0, result); err != nil {
		return nil, fmt.Errorf("jabcode: download GPU module grid: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return nil, fmt.Errorf("jabcode: run GPU sampler: %w", err)
	}
	if binary.LittleEndian.Uint32(result) != 0 {
		return nil, nil
	}
	// The grid words are packed in the balanced image's own channel order, so
	// the tail is already the bitmap's pixel buffer.
	grid := &core.Bitmap{Width: side.X, Height: side.Y, Channels: 4, Pix: result[4:]}
	// The device copy of this grid is what the payload chain classifies, so the
	// chain has to be able to prove it is being asked about this sample and not
	// a later one that overwrote the buffer.
	resident.mu.Lock()
	resident.sampledGrid = grid
	resident.mu.Unlock()
	return grid, nil
}

func (resident *gpuResidentBinarizer) forgetSampledGrid() {
	resident.mu.Lock()
	resident.sampledGrid = nil
	resident.mu.Unlock()
}

func gpuSampleParams(
	width, height int,
	pt core.Perspective,
	side image.Point,
	delta [3]core.PointF,
) [gpuSampleParamWords * 4]byte {
	var params [gpuSampleParamWords * 4]byte
	put := func(index int, value uint32) {
		binary.LittleEndian.PutUint32(params[index*4:], value)
	}
	putFloat := func(index int, value float64) {
		put(index, math.Float32bits(float32(value)))
	}
	put(gpuSampleParamWidth, uint32(width))
	put(gpuSampleParamHeight, uint32(height))
	put(gpuSampleParamSideX, uint32(side.X))
	put(gpuSampleParamSideY, uint32(side.Y))

	modW, modH := moduleExtent(pt, side)
	if min(modW, modH) < legacySampleBelowPx {
		put(gpuSampleParamRegime, gpuSampleRegimeCentre)
	} else {
		put(gpuSampleParamRegime, gpuSampleRegimeFootprint)
		kx, ky := sampleGridSize(modW, modH)
		put(gpuSampleParamKX, uint32(kx))
		put(gpuSampleParamKY, uint32(ky))
	}
	if delta != ([3]core.PointF{}) {
		put(gpuSampleParamUseDelta, 1)
	}
	for i, coefficient := range pt.Coefficients() {
		putFloat(gpuSampleParamTransform+i, coefficient)
	}
	for i, d := range delta {
		putFloat(gpuSampleParamDelta+i*2, d.X)
		putFloat(gpuSampleParamDelta+i*2+1, d.Y)
	}
	return params
}
