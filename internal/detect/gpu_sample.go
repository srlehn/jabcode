//go:build !js

package detect

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"image"
	"math"
	"sync/atomic"

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
	gpuSampleParamWidth      = 0
	gpuSampleParamHeight     = 1
	gpuSampleParamSideX      = 2
	gpuSampleParamSideY      = 3
	gpuSampleParamRegime     = 4
	gpuSampleParamKX         = 5
	gpuSampleParamKY         = 6
	gpuSampleParamUseDelta   = 7
	gpuSampleParamTransform  = 8
	gpuSampleParamDelta      = 17
	gpuSampleParamDestX      = 23
	gpuSampleParamDestY      = 24
	gpuSampleParamDestWidth  = 25
	gpuSampleParamDestHeight = 26
	// Metadata reuses the completed sampler control after the grid dispatch.
	// Keeping the wire profile in its spare tail word lets the device build the
	// metadata table without another command-buffer upload.
	gpuSampleParamMetadataProfile = 27
	gpuSampleParamWords           = 28

	gpuSampleRegimeCentre    = 0
	gpuSampleRegimeFootprint = 1
)

// gpuSampleResultWords is the module grid in words, after the reserved word
// every consumer of the buffer addresses modules from.
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

// SampleSymbol reads the module grid straight off the resident balanced image
// and leaves it there, returning the grid's shape. Nothing a successful device
// read does with the grid happens on the host, so the modules cross only when
// a fallback asks for them through MaterializeGrid.
//
// It returns nil, nil when a module maps too far outside the image, which is
// the host sampler's own failure and not an error.
func (resident *gpuResidentBinarizer) SampleSymbol(
	width, height int,
	pt core.Perspective,
	side image.Point,
	delta [3]core.PointF,
) (*core.Bitmap, error) {
	whole := []AlignmentBlock{{Transform: pt, Size: side}}
	return resident.sampleBlocks(width, height, side, whole, delta)
}

// SampleBlocks assembles a whole alignment resample in the resident grid: every
// block is scattered into its own region there and the assembled grid stays.
// Sampling block by block instead cost one download each and left the assembled
// matrix a host object the payload chain could not recognize.
func (resident *gpuResidentBinarizer) SampleBlocks(
	width, height int,
	side image.Point,
	blocks []AlignmentBlock,
) (*core.Bitmap, error) {
	return resident.sampleBlocks(width, height, side, blocks, [3]core.PointF{})
}

func (resident *gpuResidentBinarizer) sampleBlocks(
	width, height int,
	side image.Point,
	blocks []AlignmentBlock,
	delta [3]core.PointF,
) (*core.Bitmap, error) {
	if resident == nil || resident.closed || resident.sampleBindings == nil {
		return nil, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	if side.X <= 0 || side.Y <= 0 || side.X > gpuSampleMaxSide || side.Y > gpuSampleMaxSide {
		return nil, fmt.Errorf("jabcode: GPU sampler side %dx%d is out of range", side.X, side.Y)
	}
	if len(blocks) == 0 {
		return nil, fmt.Errorf("jabcode: GPU sampler was given no blocks")
	}
	for _, block := range blocks {
		if block.Size.X <= 0 || block.Size.Y <= 0 ||
			block.Size.X > gpuSampleMaxSide || block.Size.Y > gpuSampleMaxSide {
			return nil, fmt.Errorf("jabcode: GPU sampler block %dx%d is out of range",
				block.Size.X, block.Size.Y)
		}
		// The shader addresses the destination in unsigned words, so a negative
		// origin would wrap into another block's modules rather than clip.
		if block.Origin.X < 0 || block.Origin.Y < 0 {
			return nil, fmt.Errorf("jabcode: GPU sampler block origin %d,%d is negative",
				block.Origin.X, block.Origin.Y)
		}
	}
	if width <= 0 || height <= 0 ||
		width > resident.binarizer.maxWidth || height > resident.binarizer.maxHeight {
		return nil, fmt.Errorf("jabcode: GPU sampler dimensions are unavailable")
	}

	// Whether a module centre lands on the image is a property of the
	// transform, so it is settled here rather than dispatched for and read
	// back. A rejected grid costs no submission at all.
	for _, block := range blocks {
		if !sampleGridFits(block.Transform, block.Size, width, height) {
			return nil, nil
		}
	}

	// The grid buffer is about to be overwritten, so whatever the payload chain
	// was allowed to classify stops being valid here rather than on success.
	resident.forgetSampledGrid()

	modules := side.X * side.Y
	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return nil, fmt.Errorf("jabcode: create GPU sampler recorder: %w", err)
	}
	defer recorder.Abort()
	// Modules no block covers stay zero, as they do in the host's freshly
	// allocated matrix, rather than carrying whatever the previous grid left.
	if err := recorder.Fill(resident.sampleResult, 0, uint64((1+modules)*4), 0); err != nil {
		return nil, fmt.Errorf("jabcode: clear GPU module grid: %w", err)
	}
	// The host fit check above proves this grid until a sampling lane says
	// otherwise. Resident geometry sets the same word itself before its indirect
	// dispatch, so metadata has one validity contract for both entry points.
	if err := recorder.Fill(resident.sampleResult, 0, 4, 1); err != nil {
		return nil, fmt.Errorf("jabcode: admit GPU module grid: %w", err)
	}
	if err := recorder.Barrier(resident.sampleResult); err != nil {
		return nil, fmt.Errorf("jabcode: synchronize GPU sampler reset: %w", err)
	}
	for _, block := range blocks {
		params := gpuSampleParams(width, height, block, side, delta)
		if err := recordGPUUpdate(
			recorder, "upload.sample_params", resident.sampleParams, 0, params[:],
		); err != nil {
			return nil, fmt.Errorf("jabcode: update GPU sampler parameters: %w", err)
		}
		if err := recorder.Barrier(resident.sampleParams); err != nil {
			return nil, fmt.Errorf("jabcode: synchronize GPU sampler parameters: %w", err)
		}
		count := block.Size.X * block.Size.Y
		if err := recorder.Dispatch(
			resident.sampleKernel,
			resident.sampleBindings,
			vulki.Workgroups{X: uint32((count + 63) / 64), Y: 1, Z: 1},
		); err != nil {
			return nil, fmt.Errorf("jabcode: dispatch GPU sampler: %w", err)
		}
		// Blocks overlap, and the selection orders them widest first so the
		// tighter one wins. They therefore have to land in sequence.
		if err := recorder.Barrier(resident.sampleResult); err != nil {
			return nil, fmt.Errorf("jabcode: synchronize GPU module grid: %w", err)
		}
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return nil, fmt.Errorf("jabcode: run GPU sampler: %w", err)
	}
	// The grid stays where it was written. Everything a successful device read
	// does with it - the metadata walk, classification, unmasking, correction -
	// happens on this side, so the modules cross only when a host fallback
	// genuinely reads one, through MaterializeGrid.
	grid := &core.Bitmap{Width: side.X, Height: side.Y, Channels: 4}
	// The device copy of this grid is what the payload chain classifies, so the
	// chain has to be able to prove it is being asked about this sample and not
	// a later one that overwrote the buffer.
	resident.mu.Lock()
	resident.sampledGrid = grid
	resident.payloadControlReady = false
	resident.mu.Unlock()
	return grid, nil
}

// gpuGridMaterializer holds the fill to the route context that produced the
// grid, as the payload and metadata adapters do.
type gpuGridMaterializer struct {
	resident *gpuResidentBinarizer
	epoch    *atomic.Uint64
	lease    uint64
}

func (materializer gpuGridMaterializer) MaterializeGrid(matrix *core.Bitmap) bool {
	if materializer.epoch.Load() != materializer.lease {
		return false
	}
	return materializer.resident.MaterializeGrid(matrix)
}

// MaterializeGrid fills a sampled grid's module data from the device that
// produced it, for the host stages that genuinely read modules.
//
// It refuses any grid but the one the sampler currently holds. The resident
// buffer carries a single sample, so filling a bitmap from a later sample would
// hand a host stage another symbol's modules under this one's metadata, which
// is the failure hard LDPC cannot report.
func (resident *gpuResidentBinarizer) MaterializeGrid(matrix *core.Bitmap) bool {
	if resident == nil || matrix == nil {
		return false
	}
	if matrix.HasPixels() {
		return true
	}
	resident.mu.Lock()
	defer resident.mu.Unlock()
	if resident.closed || resident.sampleBindings == nil || matrix != resident.sampledGrid {
		return false
	}

	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return false
	}
	defer recorder.Abort()
	result := make([]byte, (1+matrix.Width*matrix.Height)*4)
	phaseprobe.Count("download.module_grid", len(result))
	if err := recorder.Download(resident.sampleResult, 0, result); err != nil {
		return false
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return false
	}
	// The grid words are packed in the balanced image's own channel order, so
	// the tail is already the bitmap's pixel buffer.
	matrix.Pix = result[4:]
	return true
}

// sampleGridFits reports whether every module centre of one block warps close
// enough to the frame for the sampler to place it.
//
// This is the sampler's own accept/reject rule, and both samplers apply it
// identically: a centre one pixel outside clamps to the edge and anything
// further out abandons the whole grid. Evaluating it here rather than reading a
// flag the device raised also collapses two predicates into one, so the two
// routes can no longer disagree about a centre that lands on the boundary in
// f64 and just outside it in f32.
func sampleGridFits(pt core.Perspective, side image.Point, width, height int) bool {
	for y := range side.Y {
		for x := range side.X {
			p := pt.Warp(core.Pt(float64(x)+0.5, float64(y)+0.5))
			mx, my := int(p.X), int(p.Y)
			if mx < -1 || mx > width || my < -1 || my > height {
				return false
			}
		}
	}
	return true
}

func (resident *gpuResidentBinarizer) forgetSampledGrid() {
	resident.mu.Lock()
	resident.sampledGrid = nil
	resident.payloadControlReady = false
	resident.mu.Unlock()
}

func gpuSampleParams(
	width, height int,
	block AlignmentBlock,
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
	pt := block.Transform
	put(gpuSampleParamWidth, uint32(width))
	put(gpuSampleParamHeight, uint32(height))
	put(gpuSampleParamSideX, uint32(block.Size.X))
	put(gpuSampleParamSideY, uint32(block.Size.Y))
	put(gpuSampleParamDestX, uint32(block.Origin.X))
	put(gpuSampleParamDestY, uint32(block.Origin.Y))
	put(gpuSampleParamDestWidth, uint32(side.X))
	put(gpuSampleParamDestHeight, uint32(side.Y))

	modW, modH := moduleExtent(pt, block.Size)
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
