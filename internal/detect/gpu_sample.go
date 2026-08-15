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

// gpuSampleRetainSlots is how many displaced samples stay on the device, and it
// is derived from what a route can hold at once rather than chosen.
//
// One physical sample is decoded once per compiled wire variant, so the shared
// sample stays live across the loop; the variant being decoded may resample at
// the metadata version; and the alignment cache may still hold the resample the
// previous variant produced. That is three live grids beside the one being
// sampled. A grid displaced past that is dropped rather than fetched: crossing
// the bus to preserve it would put a transfer nobody asked for into a route
// whose whole contract is one upload and one download.
const gpuSampleRetainSlots = 3

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
	for slot := range resident.sampleRetain {
		resident.sampleRetain[slot], err = resident.device.NewBuffer(gpuSampleResultWords * 4)
		if err != nil {
			return fmt.Errorf("jabcode: allocate resident GPU retained module grid: %w", err)
		}
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

	modules := side.X * side.Y
	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return nil, fmt.Errorf("jabcode: create GPU sampler recorder: %w", err)
	}
	defer recorder.Abort()
	// The grid buffer is about to be overwritten, so whatever the payload chain
	// was allowed to classify stops being valid here rather than on success. The
	// modules themselves move to a retained slot first, because the host may
	// still be handed the displaced grid.
	retained, err := resident.retainSampledGrid(recorder)
	if err != nil {
		return nil, err
	}
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
	resident.commitRetainedSample(retained)
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

// retainedSample is a recorded retention waiting on its submission: the slot
// the modules are being copied into, and the grid that slot is about to hold.
type retainedSample struct {
	slot    int
	current *core.Bitmap
}

// retainSampledGrid records a copy of the current sample out of the working
// buffer before the caller's recording overwrites it, so a grid the host still
// holds stays readable without having crossed the bus for it.
//
// The copy is only recorded here. Nothing is published until the caller commits
// the result, because a retained identity installed before its submission would
// point a materialization at whatever the slot held before, and a failed
// submission would leave it pointing there permanently.
func (resident *gpuResidentBinarizer) retainSampledGrid(
	recorder *vulki.Recorder,
) (retainedSample, error) {
	resident.mu.Lock()
	defer resident.mu.Unlock()
	current := resident.sampledGrid
	resident.sampledGrid = nil
	resident.payloadControlReady = false
	if current == nil || current.HasPixels() {
		return retainedSample{}, nil
	}
	slot := resident.sampleRetainNext
	retain := resident.sampleRetain[slot]
	if retain == nil {
		return retainedSample{}, nil
	}
	words := uint64((1 + current.Width*current.Height) * 4)
	if err := recorder.Copy(retain, 0, resident.sampleResult, 0, words); err != nil {
		return retainedSample{}, fmt.Errorf("jabcode: retain the GPU module grid: %w", err)
	}
	if err := recorder.Barrier(retain, resident.sampleResult); err != nil {
		return retainedSample{}, fmt.Errorf("jabcode: synchronize the retained GPU module grid: %w", err)
	}
	return retainedSample{slot: slot, current: current}, nil
}

// commitRetainedSample publishes a retention whose copy has run. The grid the
// slot held before this point stops being readable here, which is where the
// derived capacity of the ring is spent.
func (resident *gpuResidentBinarizer) commitRetainedSample(retained retainedSample) {
	if retained.current == nil {
		return
	}
	resident.mu.Lock()
	defer resident.mu.Unlock()
	resident.sampleRetained[retained.slot] = retained.current
	resident.sampleRetainNext = (retained.slot + 1) % len(resident.sampleRetain)
}

// MaterializeGrid fills a sampled grid's module data from the device that
// produced it, for the host stages that genuinely read modules.
//
// It refuses any grid the device is not still holding, by identity. A sample
// lives in one buffer at a time, so filling a bitmap from a later sample would
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
	if resident.closed || resident.sampleBindings == nil {
		return false
	}
	source := resident.sampleResult
	if matrix != resident.sampledGrid {
		source = nil
		for slot, held := range resident.sampleRetained {
			if held == matrix {
				source = resident.sampleRetain[slot]
				break
			}
		}
	}
	if source == nil {
		return false
	}

	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return false
	}
	defer recorder.Abort()
	result := make([]byte, (1+matrix.Width*matrix.Height)*4)
	phaseprobe.Count("download.module_grid", len(result))
	if err := recorder.Download(source, 0, result); err != nil {
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
