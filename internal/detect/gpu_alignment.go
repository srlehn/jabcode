//go:build !js

package detect

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/phaseprobe"
	"github.com/srlehn/jabcode/internal/tables"
	"github.com/srlehn/jabcode/internal/wire"
)

//go:embed shaders/alignment_search.wgsl
var alignmentSearchWGSL string

//go:embed shaders/alignment_prepare.wgsl
var alignmentPrepareWGSL string

//go:embed shaders/alignment_confirm.wgsl
var alignmentConfirmWGSL string

//go:embed shaders/alignment_rects.wgsl
var alignmentRectsWGSL string

//go:embed shaders/alignment_sample.wgsl
var alignmentSampleWGSL string

// Parameter word indices, matching alignment_search.wgsl.
const (
	gpuAlignParamWidth    = 0
	gpuAlignParamHeight   = 1
	gpuAlignParamNAPX     = 2
	gpuAlignParamNAPY     = 3
	gpuAlignParamMode     = 4
	gpuAlignParamCoreR    = 5
	gpuAlignParamCoreG    = 6
	gpuAlignParamCoreB    = 7
	gpuAlignParamSideX    = 8
	gpuAlignParamSideY    = 9
	gpuAlignParamModuleMx = 10
	gpuAlignParamQuad     = 11
	gpuAlignParamAPPosX   = 19
	gpuAlignParamAPPosY   = 28
	gpuAlignParamXform    = 37
	gpuAlignParamCornerMS = 46
	gpuAlignParamExplicit = 50
	gpuAlignParamPosition = 51

	gpuAlignModePredict = 0
	gpuAlignModeRefine  = 1
	gpuAlignModeReduce  = 2
)

// One explicit candidate's record, matching POSITION_WORDS in
// alignment_search.wgsl.
const (
	gpuAlignPositionWords = 8

	// gpuAlignMaxPositions bounds one explicit search. The version walk visits
	// five candidates, so this leaves room without making the parameter block
	// worth splitting into a buffer of its own.
	gpuAlignMaxPositions = 16
)

const gpuAlignParamWords = gpuAlignParamPosition + gpuAlignMaxPositions*gpuAlignPositionWords

// One resident rectangle records the tightest four-pattern transform chosen
// for an AP cell. The module sampler scans this small fixed table rather than
// making the host sort and upload a variable block list.
const (
	gpuAlignRectWords = 18
	gpuAlignMaxRects  = 8 * 8
)

// The first two indirect commands deliberately match the sampler and
// one-workgroup offsets used by the rest of the resident primary chain. The
// confirmation pairs are the two ordered default-mode side searches; the grid
// pairs are the ordinary AP prediction and refinement passes.
const (
	gpuAlignIndirectSample       = 0
	gpuAlignIndirectAttempt      = 3 * 4
	gpuAlignIndirectConfirmX     = 6 * 4
	gpuAlignIndirectConfirmXFold = 9 * 4
	gpuAlignIndirectConfirmY     = 12 * 4
	gpuAlignIndirectConfirmYFold = 15 * 4
	gpuAlignIndirectGridPredict  = 18 * 4
	gpuAlignIndirectGridReduce1  = 21 * 4
	gpuAlignIndirectGridRefine   = 24 * 4
	gpuAlignIndirectGridReduce2  = 27 * 4
	gpuAlignIndirectRects        = 30 * 4
	gpuAlignIndirectState        = 33
	gpuAlignIndirectInitialX     = 34
	gpuAlignIndirectInitialY     = 35
	gpuAlignIndirectConfirmedX   = 36
	gpuAlignIndirectConfirmedY   = 37
	gpuAlignIndirectWords        = 38
)

// The alignment-pattern table the resident chain reads: one pattern count per
// side version, then that version's nine positions.
//
// It is a device buffer rather than a constant in the shaders. A WGSL const
// array indexed by a runtime value compiles to zero here, and a version is only
// known at dispatch time, so the tables the shaders used to carry were read as
// zeros: the preparation derived an empty cell grid and the confirmation
// underflowed its module distance. Reading them from a buffer also leaves
// internal/tables as the one place they are written down.
const (
	gpuAlignTableVersions = 32
	gpuAlignTableStride   = 9
	gpuAlignTablePos      = gpuAlignTableVersions
	gpuAlignTableWords    = gpuAlignTablePos + gpuAlignTableVersions*gpuAlignTableStride
)

// The device layout is the host tables' own shape, so it cannot drift from them
// silently. The buffer itself is the kernel set's; see alignmentTables.
var (
	_ [len(tables.APNum) - gpuAlignTableVersions]struct{}
	_ [len(tables.APPos) - gpuAlignTableVersions]struct{}
	_ [len(tables.APPos[0]) - gpuAlignTableStride]struct{}
)

const gpuAlignRetryRetainedBytes = gpuAlignMaxRects*gpuAlignRectWords*4 +
	gpuAlignIndirectWords*4

// gpuAlignmentTableBytes lays the host tables out for the device in the order
// the shaders index them.
func gpuAlignmentTableBytes() []byte {
	out := make([]byte, gpuAlignTableWords*4)
	for version, count := range tables.APNum {
		binary.LittleEndian.PutUint32(out[version*4:], uint32(count))
	}
	for version, positions := range tables.APPos {
		base := gpuAlignTablePos + version*gpuAlignTableStride
		for at, position := range positions {
			binary.LittleEndian.PutUint32(out[(base+at)*4:], uint32(position))
		}
	}
	return out
}

// gpuAlignTiles must match TILES in alignment_search.wgsl: how many workgroups
// share one cell's candidate window. The cell count is fixed by the symbol and
// is far too small to fill a device on its own, so the window is the only axis
// with enough parallelism in it.
const gpuAlignTiles = 32

// The per-cell record the kernel carries between diagonals and hands back.
const (
	gpuAlignCellWords  = 6
	gpuAlignCellFound  = 0
	gpuAlignCellCX     = 1
	gpuAlignCellCY     = 2
	gpuAlignCellModule = 3
	gpuAlignCellDir    = 4
)

// gpuAlignMaxCells bounds the alignment grid. APNum tops out at nine per axis,
// so one allocation covers every symbol version.
const gpuAlignMaxCells = 9 * 9

// initializeAlignment allocates the cell table and compiles the search. Like
// the sampler it runs with the rest of the resident stage set, so the compile
// lands in warm-up rather than on the decode call.
func (resident *gpuResidentBinarizer) initializeAlignment() error {
	var err error
	resident.alignCells, err = resident.device.NewBuffer(gpuAlignMaxCells * gpuAlignCellWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU alignment cells: %w", err)
	}
	resident.alignParams, err = resident.device.NewBuffer(gpuAlignParamWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU alignment parameters: %w", err)
	}
	resident.alignTiles, err = resident.device.NewBuffer(
		gpuAlignMaxCells * gpuAlignTiles * gpuAlignCellWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU alignment tiles: %w", err)
	}
	resident.alignKernel, err = resident.kernels.alignmentSearch()
	if err != nil {
		return err
	}
	resident.alignBindings, err = resident.alignKernel.NewBindings(
		vulki.BindBuffer(0, resident.binarizer.packedMasks),
		vulki.BindBuffer(1, resident.alignCells),
		vulki.BindBuffer(2, resident.alignParams),
		vulki.BindBuffer(3, resident.alignTiles),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU alignment search: %w", err)
	}
	return nil
}

// initializeResidentAlignmentRetry binds the alignment stages that derive a
// paired candidate from a fixed direct-result slot. It runs after metadata and
// result initialization because the preparation kernel reads the direct
// attempt in place; the older diagnostic alignment API remains usable without
// this resident chain.
func (resident *gpuResidentBinarizer) initializeResidentAlignmentRetry() error {
	var err error
	resident.alignRects, err = resident.device.NewBuffer(gpuAlignMaxRects * gpuAlignRectWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU alignment rectangles: %w", err)
	}
	resident.alignIndirect, err = resident.device.NewBuffer(gpuAlignIndirectWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU alignment dispatches: %w", err)
	}
	// The tables belong to the device, not to this context: they never change,
	// and holding them per context uploaded the same 1.3 KB once per pyramid
	// level on every image, which is a census line for nothing.
	resident.alignTable, err = resident.kernels.alignmentTables()
	if err != nil {
		return err
	}
	resident.alignPrepareKernel, err = resident.kernels.alignmentPrepare()
	if err != nil {
		return err
	}
	resident.alignConfirmKernel, err = resident.kernels.alignmentConfirm()
	if err != nil {
		return err
	}
	resident.alignRectsKernel, err = resident.kernels.alignmentRects()
	if err != nil {
		return err
	}
	resident.alignSampleKernel, err = resident.kernels.alignmentSample()
	if err != nil {
		return err
	}
	resident.alignPrepareBindings, err = resident.alignPrepareKernel.NewBindings(
		vulki.BindBuffer(0, resident.primaryResult),
		vulki.BindBuffer(1, resident.sampleParams),
		vulki.BindBuffer(2, resident.primaryResultControl),
		vulki.BindBuffer(3, resident.alignParams),
		vulki.BindBuffer(4, resident.alignCells),
		vulki.BindBuffer(5, resident.alignIndirect),
		vulki.BindBuffer(6, resident.sampleResult),
		vulki.BindBuffer(7, resident.alignTable),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU alignment preparation: %w", err)
	}
	resident.alignConfirmBindings, err = resident.alignConfirmKernel.NewBindings(
		vulki.BindBuffer(0, resident.alignCells),
		vulki.BindBuffer(1, resident.alignParams),
		vulki.BindBuffer(2, resident.alignIndirect),
		vulki.BindBuffer(3, resident.alignTable),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU alignment side confirmation: %w", err)
	}
	resident.alignRectsBindings, err = resident.alignRectsKernel.NewBindings(
		vulki.BindBuffer(0, resident.alignCells),
		vulki.BindBuffer(1, resident.alignParams),
		vulki.BindBuffer(2, resident.alignRects),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU alignment rectangles: %w", err)
	}
	resident.alignSampleBindings, err = resident.alignSampleKernel.NewBindings(
		vulki.BindBuffer(0, resident.balanced),
		vulki.BindBuffer(1, resident.sampleResult),
		vulki.BindBuffer(2, resident.alignParams),
		vulki.BindBuffer(3, resident.alignRects),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU alignment sampler: %w", err)
	}
	return nil
}

// recordResidentAlignmentRetry records one AP result beside its direct result.
// The host cannot know whether the direct message parses until after the joined
// download, so both candidates are completed while all geometry stays in
// resident buffers, including the four search passes and the compact rectangle
// table consumed by the module-parallel sampler.
func (resident *gpuResidentBinarizer) recordResidentAlignmentRetry(
	recorder *vulki.Recorder,
	variant wire.Variant,
	directSlot int,
) error {
	if resident.alignPrepareBindings == nil || resident.alignConfirmBindings == nil ||
		resident.alignRectsBindings == nil ||
		resident.alignSampleBindings == nil || resident.alignIndirect == nil {
		return fmt.Errorf("jabcode: resident GPU alignment retry is unavailable")
	}
	if directSlot < 0 || directSlot >= gpuPrimaryResultDirectSlots {
		return fmt.Errorf("jabcode: GPU direct result slot %d is out of range", directSlot)
	}
	control := gpuMetadataProfile(variant)&gpuMetadataProfileMask |
		uint32(directSlot)<<gpuMetadataResultSlotShift | gpuMetadataResultBatchFlag
	if err := recorder.Fill(
		resident.sampleParams,
		uint64(gpuSampleParamMetadataProfile*4),
		4,
		control,
	); err != nil {
		return fmt.Errorf("jabcode: select resident GPU alignment slot: %w", err)
	}
	if err := recorder.Fill(
		resident.alignIndirect,
		uint64(gpuAlignIndirectState*4),
		4,
		0,
	); err != nil {
		return fmt.Errorf("jabcode: reset resident GPU alignment control: %w", err)
	}
	if err := recorder.Barrier(
		resident.sampleParams, resident.primaryResult, resident.alignIndirect,
	); err != nil {
		return fmt.Errorf("jabcode: synchronize resident GPU alignment result: %w", err)
	}
	prepare := func() error {
		if err := recorder.Dispatch(
			resident.alignPrepareKernel,
			resident.alignPrepareBindings,
			vulki.Workgroups{X: 1, Y: 1, Z: 1},
		); err != nil {
			return fmt.Errorf("jabcode: prepare resident GPU alignment retry: %w", err)
		}
		if err := recorder.Barrier(
			resident.alignParams,
			resident.alignCells,
			resident.alignIndirect,
			resident.sampleParams,
			resident.sampleResult,
		); err != nil {
			return fmt.Errorf("jabcode: synchronize resident GPU alignment preparation: %w", err)
		}
		return nil
	}
	pass := func(mode uint32, offset uint64) error {
		if err := recorder.Fill(
			resident.alignParams,
			uint64(gpuAlignParamMode*4),
			4,
			mode,
		); err != nil {
			return fmt.Errorf("jabcode: select resident GPU alignment pass: %w", err)
		}
		if err := recorder.Barrier(resident.alignParams); err != nil {
			return fmt.Errorf("jabcode: synchronize resident GPU alignment pass: %w", err)
		}
		if err := recorder.DispatchIndirect(
			resident.alignKernel,
			resident.alignBindings,
			resident.alignIndirect,
			offset,
		); err != nil {
			return fmt.Errorf("jabcode: dispatch resident GPU alignment pass: %w", err)
		}
		if err := recorder.Barrier(resident.alignCells, resident.alignTiles); err != nil {
			return fmt.Errorf("jabcode: synchronize resident GPU alignment pass result: %w", err)
		}
		return nil
	}
	confirm := func() error {
		if err := recorder.Dispatch(
			resident.alignConfirmKernel,
			resident.alignConfirmBindings,
			vulki.Workgroups{X: 1, Y: 1, Z: 1},
		); err != nil {
			return fmt.Errorf("jabcode: confirm resident GPU alignment side: %w", err)
		}
		if err := recorder.Barrier(
			resident.alignCells, resident.alignParams, resident.alignIndirect,
		); err != nil {
			return fmt.Errorf("jabcode: synchronize resident GPU alignment side: %w", err)
		}
		return nil
	}

	if err := prepare(); err != nil {
		return err
	}
	confirmation := [...]struct {
		mode   uint32
		offset uint64
	}{
		{gpuAlignModePredict, gpuAlignIndirectConfirmX},
		{gpuAlignModeReduce, gpuAlignIndirectConfirmXFold},
	}
	for _, step := range confirmation {
		if err := pass(step.mode, step.offset); err != nil {
			return err
		}
	}
	if err := confirm(); err != nil {
		return err
	}
	confirmation = [...]struct {
		mode   uint32
		offset uint64
	}{
		{gpuAlignModePredict, gpuAlignIndirectConfirmY},
		{gpuAlignModeReduce, gpuAlignIndirectConfirmYFold},
	}
	for _, step := range confirmation {
		if err := pass(step.mode, step.offset); err != nil {
			return err
		}
	}
	if err := confirm(); err != nil {
		return err
	}
	if err := prepare(); err != nil {
		return err
	}
	grid := [...]struct {
		mode   uint32
		offset uint64
	}{
		{gpuAlignModePredict, gpuAlignIndirectGridPredict},
		{gpuAlignModeReduce, gpuAlignIndirectGridReduce1},
		{gpuAlignModeRefine, gpuAlignIndirectGridRefine},
		{gpuAlignModeReduce, gpuAlignIndirectGridReduce2},
	}
	for _, step := range grid {
		if err := pass(step.mode, step.offset); err != nil {
			return err
		}
	}
	if err := recorder.DispatchIndirect(
		resident.alignRectsKernel,
		resident.alignRectsBindings,
		resident.alignIndirect,
		gpuAlignIndirectRects,
	); err != nil {
		return fmt.Errorf("jabcode: dispatch resident GPU alignment rectangles: %w", err)
	}
	if err := recorder.Barrier(resident.alignRects); err != nil {
		return fmt.Errorf("jabcode: synchronize resident GPU alignment rectangles: %w", err)
	}
	if err := recorder.DispatchIndirect(
		resident.alignSampleKernel,
		resident.alignSampleBindings,
		resident.alignIndirect,
		gpuAlignIndirectSample,
	); err != nil {
		return fmt.Errorf("jabcode: dispatch resident GPU alignment sampler: %w", err)
	}
	if err := recorder.Barrier(resident.sampleResult); err != nil {
		return fmt.Errorf("jabcode: synchronize resident GPU alignment sample: %w", err)
	}
	return nil
}

// SearchAlignment locates every interior alignment pattern against the masks
// the pass left resident, and hands back the cell table.
//
// The grid costs one submission and two dispatches, both fully parallel. The
// first predicts every cell from the perspective the four finder centres define
// and searches them all at once; the second revisits only what the first left
// unfound, correcting each prediction by the residual its located neighbours
// measured. The host sees neither pass, and the masks it would otherwise have
// needed never leave the device.
func (resident *gpuResidentBinarizer) SearchAlignment(
	width, height int,
	grid alignmentGrid,
) ([]FinderPattern, error) {
	if resident == nil || resident.closed || resident.alignBindings == nil {
		return nil, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	cells := grid.nApX * grid.nApY
	if grid.nApX <= 0 || grid.nApY <= 0 || cells > gpuAlignMaxCells {
		return nil, fmt.Errorf("jabcode: GPU alignment grid %dx%d is out of range", grid.nApX, grid.nApY)
	}
	if len(grid.posX) < grid.nApX || len(grid.posY) < grid.nApY {
		return nil, fmt.Errorf("jabcode: GPU alignment grid is missing its module positions")
	}
	if width <= 0 || height <= 0 ||
		width > resident.binarizer.maxWidth || height > resident.binarizer.maxHeight {
		return nil, fmt.Errorf("jabcode: GPU alignment dimensions are unavailable")
	}
	transform, ok := gpuAlignmentTransform(grid)
	if !ok {
		return nil, nil
	}

	resident.mu.Lock()
	defer resident.mu.Unlock()
	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return nil, fmt.Errorf("jabcode: create GPU alignment recorder: %w", err)
	}
	defer recorder.Abort()

	if err := recordGPUUpdate(
		recorder, "upload.alignment_seed", resident.alignCells, 0, gpuAlignmentSeed(grid),
	); err != nil {
		return nil, fmt.Errorf("jabcode: seed GPU alignment cells: %w", err)
	}
	if err := recorder.Barrier(resident.binarizer.packedMasks, resident.alignCells); err != nil {
		return nil, fmt.Errorf("jabcode: synchronize GPU alignment inputs: %w", err)
	}
	// Each search pass is followed by the fold that turns its per-tile winners
	// into one result per cell, so a pass is two dispatches: the wide one that
	// fills the device, and a narrow one over the tiles it wrote.
	passes := [4]int{
		gpuAlignModePredict, gpuAlignModeReduce,
		gpuAlignModeRefine, gpuAlignModeReduce,
	}
	for at, mode := range passes {
		params := gpuAlignmentParams(width, height, grid, transform, mode)
		if err := recordGPUUpdate(
			recorder, "upload.alignment_params", resident.alignParams, 0, params[:],
		); err != nil {
			return nil, fmt.Errorf("jabcode: update GPU alignment parameters: %w", err)
		}
		if err := recorder.Barrier(resident.alignParams); err != nil {
			return nil, fmt.Errorf("jabcode: synchronize GPU alignment parameters: %w", err)
		}
		groups := uint32(cells)
		if mode != gpuAlignModeReduce {
			groups *= gpuAlignTiles
		}
		if err := recorder.Dispatch(
			resident.alignKernel,
			resident.alignBindings,
			vulki.Workgroups{X: groups, Y: 1, Z: 1},
		); err != nil {
			return nil, fmt.Errorf("jabcode: dispatch GPU alignment pass %d: %w", at, err)
		}
		if err := recorder.Barrier(resident.alignCells, resident.alignTiles); err != nil {
			return nil, fmt.Errorf("jabcode: synchronize GPU alignment pass %d: %w", at, err)
		}
	}
	table := make([]byte, cells*gpuAlignCellWords*4)
	phaseprobe.Count("download.alignment_cells", len(table))
	if err := recorder.Download(resident.alignCells, 0, table); err != nil {
		return nil, fmt.Errorf("jabcode: record GPU alignment download: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return nil, fmt.Errorf("jabcode: run GPU alignment search: %w", err)
	}
	return parseAlignmentCells(table, cells, grid.apType), nil
}

// SearchAlignmentPositions answers a list of explicit predicted positions
// against the masks the pass left resident, one accepted pattern per candidate
// or none.
//
// This is what the side-version confirmation needs. It predicts where the first
// alignment pattern would sit under each candidate version and asks which
// prediction is borne out, so its candidates are image positions with their own
// module axes rather than cells of a grid. Asking once per candidate would be a
// submission and a stall per version for a few hundred bytes each; the
// candidates are independent, so they go in one dispatch and the caller keeps
// its own order to pick the first hit.
func (resident *gpuResidentBinarizer) SearchAlignmentPositions(
	width, height int,
	apType int,
	candidates []alignmentCandidate,
) ([]FinderPattern, error) {
	if resident == nil || resident.closed || resident.alignBindings == nil {
		return nil, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	if len(candidates) == 0 || len(candidates) > gpuAlignMaxPositions {
		return nil, fmt.Errorf("jabcode: GPU alignment position count %d is out of range", len(candidates))
	}
	if width <= 0 || height <= 0 ||
		width > resident.binarizer.maxWidth || height > resident.binarizer.maxHeight {
		return nil, fmt.Errorf("jabcode: GPU alignment dimensions are unavailable")
	}

	resident.mu.Lock()
	defer resident.mu.Unlock()
	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return nil, fmt.Errorf("jabcode: create GPU alignment position recorder: %w", err)
	}
	defer recorder.Abort()

	// Nothing is seeded found here: every candidate is a question, and the grid
	// search's corner seeding has no counterpart in a plain list.
	if err := recorder.Fill(
		resident.alignCells, 0, uint64(len(candidates)*gpuAlignCellWords*4), 0,
	); err != nil {
		return nil, fmt.Errorf("jabcode: clear GPU alignment position cells: %w", err)
	}
	if err := recorder.Barrier(resident.binarizer.packedMasks, resident.alignCells); err != nil {
		return nil, fmt.Errorf("jabcode: synchronize GPU alignment position inputs: %w", err)
	}
	params := gpuAlignmentPositionParams(width, height, apType, candidates)
	for _, pass := range [2]struct {
		mode   int
		groups uint32
	}{
		{gpuAlignModePredict, uint32(len(candidates)) * gpuAlignTiles},
		{gpuAlignModeReduce, uint32(len(candidates))},
	} {
		binary.LittleEndian.PutUint32(params[gpuAlignParamMode*4:], uint32(pass.mode))
		if err := recordGPUUpdate(
			recorder, "upload.alignment_params", resident.alignParams, 0, params[:],
		); err != nil {
			return nil, fmt.Errorf("jabcode: update GPU alignment position mode: %w", err)
		}
		if err := recorder.Barrier(resident.alignParams); err != nil {
			return nil, fmt.Errorf("jabcode: synchronize GPU alignment position mode: %w", err)
		}
		if err := recorder.Dispatch(
			resident.alignKernel,
			resident.alignBindings,
			vulki.Workgroups{X: pass.groups, Y: 1, Z: 1},
		); err != nil {
			return nil, fmt.Errorf("jabcode: dispatch GPU alignment position pass: %w", err)
		}
		if err := recorder.Barrier(resident.alignCells, resident.alignTiles); err != nil {
			return nil, fmt.Errorf("jabcode: synchronize GPU alignment position pass: %w", err)
		}
	}
	table := make([]byte, len(candidates)*gpuAlignCellWords*4)
	phaseprobe.Count("download.alignment_positions", len(table))
	if err := recorder.Download(resident.alignCells, 0, table); err != nil {
		return nil, fmt.Errorf("jabcode: record GPU alignment position download: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return nil, fmt.Errorf("jabcode: run GPU alignment position search: %w", err)
	}
	return parseAlignmentCells(table, len(candidates), apType), nil
}

func gpuAlignmentPositionParams(
	width, height int,
	apType int,
	candidates []alignmentCandidate,
) [gpuAlignParamWords * 4]byte {
	var params [gpuAlignParamWords * 4]byte
	put := func(index int, value uint32) {
		binary.LittleEndian.PutUint32(params[index*4:], value)
	}
	putF := func(index int, value float64) {
		put(index, math.Float32bits(float32(value)))
	}
	put(gpuAlignParamWidth, uint32(width))
	put(gpuAlignParamHeight, uint32(height))
	// The candidates are laid out as a single row so the kernel's cell indexing
	// addresses them directly.
	put(gpuAlignParamNAPX, uint32(len(candidates)))
	put(gpuAlignParamNAPY, 1)
	put(gpuAlignParamExplicit, 1)
	core := apCoreColorIndex(apType) * 3
	for channel := range 3 {
		bit := uint32(0)
		if palette.Default[core+channel] != 0 {
			bit = 1
		}
		put(gpuAlignParamCoreR+channel, bit)
	}
	for at, candidate := range candidates {
		base := gpuAlignParamPosition + at*gpuAlignPositionWords
		putF(base+0, candidate.Center.X)
		putF(base+1, candidate.Center.Y)
		putF(base+2, candidate.ModuleSize)
		putF(base+3, candidate.U.X)
		putF(base+4, candidate.U.Y)
		putF(base+5, candidate.V.X)
		putF(base+6, candidate.V.Y)
		putF(base+7, candidate.ModuleMax)
	}
	return params
}

// gpuAlignmentTransform maps the grid's own module coordinates to image space
// through the four located finder centres. Predicting from this is what lets
// every cell search independently: the corners are at known module positions,
// so the map is exact wherever the capture is a perspective of the
// symbol, and no cell has to wait for a neighbour to be placed first.
func gpuAlignmentTransform(grid alignmentGrid) ([9]float64, bool) {
	lastX := grid.posX[grid.nApX-1]
	lastY := grid.posY[grid.nApY-1]
	firstX := grid.posX[0]
	firstY := grid.posY[0]
	if lastX == firstX || lastY == firstY {
		return [9]float64{}, false
	}
	source := [4]core.PointF{
		core.Pt(float64(firstX), float64(firstY)),
		core.Pt(float64(lastX), float64(firstY)),
		core.Pt(float64(lastX), float64(lastY)),
		core.Pt(float64(firstX), float64(lastY)),
	}
	destination := [4]core.PointF{
		grid.corners[0].Center, grid.corners[1].Center,
		grid.corners[2].Center, grid.corners[3].Center,
	}
	coefficients := core.QuadToQuad(source, destination).Coefficients()
	for _, coefficient := range coefficients {
		if math.IsNaN(coefficient) || math.IsInf(coefficient, 0) {
			return [9]float64{}, false
		}
	}
	return coefficients, true
}

// gpuAlignmentSeed lays down the four quad corners the search extrapolates
// from. They are measurements the finder stage already made, so the kernel
// treats them as found and never searches them.
func gpuAlignmentSeed(grid alignmentGrid) []byte {
	seed := make([]byte, grid.nApX*grid.nApY*gpuAlignCellWords*4)
	corners := [4]int{
		0,
		grid.nApX - 1,
		(grid.nApY-1)*grid.nApX + grid.nApX - 1,
		(grid.nApY - 1) * grid.nApX,
	}
	for at, index := range corners {
		fp := grid.corners[at]
		base := index * gpuAlignCellWords * 4
		binary.LittleEndian.PutUint32(seed[base+gpuAlignCellFound*4:], 1)
		binary.LittleEndian.PutUint32(seed[base+gpuAlignCellCX*4:], math.Float32bits(float32(fp.Center.X)))
		binary.LittleEndian.PutUint32(seed[base+gpuAlignCellCY*4:], math.Float32bits(float32(fp.Center.Y)))
		binary.LittleEndian.PutUint32(seed[base+gpuAlignCellModule*4:], math.Float32bits(float32(fp.ModuleSize)))
		binary.LittleEndian.PutUint32(seed[base+gpuAlignCellDir*4:], uint32(int32(fp.direction)))
	}
	return seed
}

func gpuAlignmentParams(
	width, height int,
	grid alignmentGrid,
	transform [9]float64,
	mode int,
) [gpuAlignParamWords * 4]byte {
	var params [gpuAlignParamWords * 4]byte
	put := func(index int, value uint32) {
		binary.LittleEndian.PutUint32(params[index*4:], value)
	}
	putF := func(index int, value float64) {
		put(index, math.Float32bits(float32(value)))
	}
	put(gpuAlignParamWidth, uint32(width))
	put(gpuAlignParamHeight, uint32(height))
	put(gpuAlignParamNAPX, uint32(grid.nApX))
	put(gpuAlignParamNAPY, uint32(grid.nApY))
	put(gpuAlignParamMode, uint32(mode))
	// The mask bit is the binarized channel, so the core's palette component
	// reduces to whether that channel is on for this pattern type.
	core := apCoreColorIndex(grid.apType) * 3
	for channel := range 3 {
		bit := uint32(0)
		if palette.Default[core+channel] != 0 {
			bit = 1
		}
		put(gpuAlignParamCoreR+channel, bit)
	}
	put(gpuAlignParamSideX, uint32(grid.sideX))
	put(gpuAlignParamSideY, uint32(grid.sideY))
	// The ceiling the host's run test uses: a core run at or above it is a
	// field of colour rather than a pattern core.
	putF(gpuAlignParamModuleMx, gpuAlignmentModuleCeiling(grid))
	for at := range 4 {
		putF(gpuAlignParamQuad+at*2, grid.corners[at].Center.X)
		putF(gpuAlignParamQuad+at*2+1, grid.corners[at].Center.Y)
	}
	for at := range grid.posX {
		put(gpuAlignParamAPPosX+at, uint32(grid.posX[at]))
	}
	for at := range grid.posY {
		put(gpuAlignParamAPPosY+at, uint32(grid.posY[at]))
	}
	for at, coefficient := range transform {
		putF(gpuAlignParamXform+at, coefficient)
	}
	for at, fp := range grid.corners {
		putF(gpuAlignParamCornerMS+at, fp.ModuleSize)
	}
	return params
}

// gpuAlignmentModuleCeiling derives the largest core run the search will accept
// from the quad's own scale, so nothing here is a fixed pixel bound: the
// corners span a known number of modules, and a core run may not exceed a small
// multiple of one of them.
func gpuAlignmentModuleCeiling(grid alignmentGrid) float64 {
	span := 0.0
	for _, fp := range grid.corners {
		span += fp.ModuleSize
	}
	return 3 * span / 4
}

func parseAlignmentCells(table []byte, cells, apType int) []FinderPattern {
	out := make([]FinderPattern, cells)
	for index := range out {
		base := index * gpuAlignCellWords * 4
		found := binary.LittleEndian.Uint32(table[base+gpuAlignCellFound*4:])
		out[index] = FinderPattern{
			Typ: apType,
			Center: core.PointF{
				X: float64(math.Float32frombits(binary.LittleEndian.Uint32(table[base+gpuAlignCellCX*4:]))),
				Y: float64(math.Float32frombits(binary.LittleEndian.Uint32(table[base+gpuAlignCellCY*4:]))),
			},
			ModuleSize: float64(math.Float32frombits(binary.LittleEndian.Uint32(table[base+gpuAlignCellModule*4:]))),
			direction:  int(int32(binary.LittleEndian.Uint32(table[base+gpuAlignCellDir*4:]))),
		}
		if found != 0 {
			out[index].FoundCount = 1
		}
	}
	return out
}
