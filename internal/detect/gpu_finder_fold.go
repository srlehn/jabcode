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

//go:embed shaders/finder_fold.wgsl
var finderFoldWGSL string

//go:embed shaders/finder_sort.wgsl
var finderSortWGSL string

//go:embed shaders/finder_select.wgsl
var finderSelectWGSL string

//go:embed shaders/finder_candidates.wgsl
var finderCandidatesWGSL string

//go:embed shaders/finder_pool.wgsl
var finderPoolWGSL string

//go:embed shaders/finder_corner.wgsl
var finderCornerWGSL string

// gpuFinderFamilyPoolSlots bounds the cross-direction candidate union the
// missing-corner search reads. The host pool grows without a bound, so this is
// a real difference and the kernel reports a full pool rather than absorbing
// it: one direction contributes at most maxFinderPatterns-1 entries and a level
// sweeps six, which leaves an order of magnitude of headroom over what a level
// can produce even when no two directions agree on anything.
const gpuFinderFamilyPoolSlots = 8192

// gpuFinderFoldSlots is the candidate buffer's length in records. The ordering
// network needs a power of two and gives the slots past the real count an
// infinite key, so the buffer is rounded up rather than sized to the
// compaction's own capacity.
const gpuFinderFoldSlots = 65536

// Record and record layout, matching finder_fold.wgsl.
const (
	gpuFinderFoldCandidateWords = 6
	gpuFinderFoldPatternWords   = 6
	gpuFinderFoldRecordWords    = 16
	gpuFinderFoldParamWords     = 8

	gpuFinderFoldRecordTotal          = 0
	gpuFinderFoldRecordTypeCount      = 1
	gpuFinderFoldRecordDropped        = 5
	gpuFinderFoldRecordWeakTotal      = 6
	gpuFinderFoldRecordConsumed       = 7
	gpuFinderFoldRecordCrossSurvivors = 8
)

// Assembly record layout, matching finder_candidates.wgsl.
const (
	gpuFinderAssemblyWords    = 4
	gpuFinderAssemblyCount    = 0
	gpuFinderAssemblyDeferred = 1
)

// Pool parameter and record layout, matching finder_pool.wgsl.
const (
	gpuFinderPoolParamWords = 4
	gpuFinderPoolWords      = 4

	gpuFinderPoolCount    = 0
	gpuFinderPoolDropped  = 1
	gpuFinderPoolTypeMask = 2

	gpuFinderPoolModeReplace = 0
	gpuFinderPoolModeAverage = 1
)

// Corner parameter and record layout, matching finder_corner.wgsl.
const (
	gpuFinderCornerParamWords = 4
	gpuFinderCornerWords      = 64

	gpuFinderCornerSource       = 0
	gpuFinderCornerMiss         = 1
	gpuFinderCornerOK           = 2
	gpuFinderCornerAlternatives = 3
	gpuFinderCornerPattern      = 4
	gpuFinderCornerAlternative  = 16
)

// Selection record layout, matching finder_select.wgsl.
const (
	gpuFinderSelectPreprune    = 0
	gpuFinderSelectPreselect   = 4
	gpuFinderSelectSelected    = 8
	gpuFinderSelectMissing     = 12
	gpuFinderSelectPatterns    = 16
	gpuFinderSelectPrePatterns = 40
	gpuFinderSelectWords       = 64
)

// gpuFinderFoldRetainedBytes is what the fold holds on the device for the life
// of a route context.
const gpuFinderFoldRetainedBytes = (gpuFinderFoldParamWords +
	gpuFinderFoldSlots*gpuFinderFoldCandidateWords +
	maxFinderPatterns*gpuFinderFoldPatternWords + gpuFinderFoldRecordWords +
	gpuFinderSelectWords +
	maxContextualFinderSeeds*gpuFinderFoldPatternWords +
	gpuFinderAssemblyWords + gpuFinderAssemblyWords +
	3*gpuFinderPoolParamWords + 3*gpuFinderPoolWords +
	gpuFinderFamilyPoolSlots*gpuFinderFoldPatternWords +
	maxContextualFinderSeeds*gpuFinderFoldPatternWords +
	maxContextualFinderCandidates*gpuFinderFoldPatternWords +
	gpuFinderCornerParamWords + gpuFinderCornerWords) * 4

// initializeFinderFold allocates the fold's buffers and compiles its kernel
// with the rest of the resident stage set.
func (resident *gpuResidentBinarizer) initializeFinderFold() error {
	var err error
	resident.foldParams, err = resident.device.NewBuffer(gpuFinderFoldParamWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU finder fold parameters: %w", err)
	}
	resident.foldCandidates, err = resident.device.NewBuffer(
		gpuFinderFoldSlots * gpuFinderFoldCandidateWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU finder fold candidates: %w", err)
	}
	resident.foldPatterns, err = resident.device.NewBuffer(
		maxFinderPatterns * gpuFinderFoldPatternWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU finder fold patterns: %w", err)
	}
	resident.foldRecord, err = resident.device.NewBuffer(gpuFinderFoldRecordWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU finder fold record: %w", err)
	}
	resident.foldSelection, err = resident.device.NewBuffer(gpuFinderSelectWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU finder selection: %w", err)
	}
	resident.foldWeak, err = resident.device.NewBuffer(
		maxContextualFinderSeeds * gpuFinderFoldPatternWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU finder weak seeds: %w", err)
	}
	resident.assemblyParams, err = resident.device.NewBuffer(gpuFinderAssemblyWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU finder assembly parameters: %w", err)
	}
	resident.assemblyRecord, err = resident.device.NewBuffer(gpuFinderAssemblyWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU finder assembly record: %w", err)
	}
	for _, params := range []**vulki.Buffer{
		&resident.familyPoolParams, &resident.groupParams, &resident.contextualParams,
	} {
		*params, err = resident.device.NewBuffer(gpuFinderPoolParamWords * 4)
		if err != nil {
			return fmt.Errorf("jabcode: allocate resident GPU finder pool parameters: %w", err)
		}
	}
	resident.contextualGroups, err = resident.device.NewBuffer(
		maxContextualFinderSeeds * gpuFinderFoldPatternWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU finder seed groups: %w", err)
	}
	resident.contextualGroupsRecord, err = resident.device.NewBuffer(gpuFinderPoolWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU finder seed group record: %w", err)
	}
	resident.contextualPool, err = resident.device.NewBuffer(
		maxContextualFinderCandidates * gpuFinderFoldPatternWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU finder contextual pool: %w", err)
	}
	resident.contextualPoolRecord, err = resident.device.NewBuffer(gpuFinderPoolWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU finder contextual pool record: %w", err)
	}
	resident.familyPool, err = resident.device.NewBuffer(
		gpuFinderFamilyPoolSlots * gpuFinderFoldPatternWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU finder family pool: %w", err)
	}
	resident.familyPoolRecord, err = resident.device.NewBuffer(gpuFinderPoolWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU finder family pool record: %w", err)
	}
	resident.cornerParams, err = resident.device.NewBuffer(gpuFinderCornerParamWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU finder corner parameters: %w", err)
	}
	resident.cornerRecord, err = resident.device.NewBuffer(gpuFinderCornerWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU finder corner record: %w", err)
	}
	resident.assemblyKernel, err = resident.kernels.finderCandidates()
	if err != nil {
		return err
	}
	resident.poolKernel, err = resident.kernels.finderPool()
	if err != nil {
		return err
	}
	resident.cornerKernel, err = resident.kernels.finderCorner()
	if err != nil {
		return err
	}
	resident.cornerBindings, err = resident.cornerKernel.NewBindings(
		vulki.BindBuffer(0, resident.cornerParams),
		vulki.BindBuffer(1, resident.foldSelection),
		vulki.BindBuffer(2, resident.familyPool),
		vulki.BindBuffer(3, resident.familyPoolRecord),
		vulki.BindBuffer(4, resident.cornerRecord),
		vulki.BindBuffer(5, resident.contextualPool),
		vulki.BindBuffer(6, resident.contextualPoolRecord),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU finder corner: %w", err)
	}
	resident.familyPoolBindings, err = resident.poolKernel.NewBindings(
		vulki.BindBuffer(0, resident.familyPoolParams),
		vulki.BindBuffer(1, resident.foldPatterns),
		vulki.BindBuffer(2, resident.foldRecord),
		vulki.BindBuffer(3, resident.familyPool),
		vulki.BindBuffer(4, resident.familyPoolRecord),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU finder family pool: %w", err)
	}
	resident.groupBindings, err = resident.poolKernel.NewBindings(
		vulki.BindBuffer(0, resident.groupParams),
		vulki.BindBuffer(1, resident.foldWeak),
		vulki.BindBuffer(2, resident.foldRecord),
		vulki.BindBuffer(3, resident.contextualGroups),
		vulki.BindBuffer(4, resident.contextualGroupsRecord),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU finder seed grouping: %w", err)
	}
	resident.contextualPoolBindings, err = resident.poolKernel.NewBindings(
		vulki.BindBuffer(0, resident.contextualParams),
		vulki.BindBuffer(1, resident.contextualGroups),
		vulki.BindBuffer(2, resident.contextualGroupsRecord),
		vulki.BindBuffer(3, resident.contextualPool),
		vulki.BindBuffer(4, resident.contextualPoolRecord),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU finder contextual pool: %w", err)
	}
	resident.foldKernel, err = resident.kernels.finderFold()
	if err != nil {
		return err
	}
	resident.sortKernel, err = resident.kernels.finderSort()
	if err != nil {
		return err
	}
	resident.sortBindings, err = resident.sortKernel.NewBindings(
		vulki.BindBuffer(0, resident.foldParams),
		vulki.BindBuffer(1, resident.foldCandidates),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU finder order: %w", err)
	}
	resident.selectKernel, err = resident.kernels.finderSelect()
	if err != nil {
		return err
	}
	resident.selectBindings, err = resident.selectKernel.NewBindings(
		vulki.BindBuffer(0, resident.foldParams),
		vulki.BindBuffer(1, resident.foldPatterns),
		vulki.BindBuffer(2, resident.foldRecord),
		vulki.BindBuffer(3, resident.foldSelection),
		vulki.BindBuffer(4, resident.contextualPoolRecord),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU finder selection: %w", err)
	}
	resident.foldBindings, err = resident.foldKernel.NewBindings(
		vulki.BindBuffer(0, resident.foldParams),
		vulki.BindBuffer(1, resident.foldCandidates),
		vulki.BindBuffer(2, resident.foldPatterns),
		vulki.BindBuffer(3, resident.foldRecord),
		vulki.BindBuffer(4, resident.foldWeak),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU finder fold: %w", err)
	}
	return nil
}

// newFinderAssemblyBindings binds the candidate assembly over one outcome
// buffer. The buffer is the caller's because the compacted outcomes stay where
// the chain wrote them: the batched retry keeps a region per direction, a
// single sweep keeps one, and neither is copied here.
func (resident *gpuResidentBinarizer) newFinderAssemblyBindings(
	outcomes *vulki.Buffer,
) (*vulki.BindingSet, error) {
	bindings, err := resident.assemblyKernel.NewBindings(
		vulki.BindBuffer(0, resident.assemblyParams),
		vulki.BindBuffer(1, outcomes),
		vulki.BindBuffer(2, resident.foldCandidates),
		vulki.BindBuffer(3, resident.foldParams),
		vulki.BindBuffer(4, resident.assemblyRecord),
	)
	if err != nil {
		return nil, fmt.Errorf("jabcode: bind resident GPU finder assembly: %w", err)
	}
	return bindings, nil
}

// gpuFinderFoldCandidate is one admitted candidate in the terms the fold reads.
// The chain's flags are not among them: a candidate reaching the fold has
// already passed them, so carrying them further would only invite the fold to
// second-guess a decision made where the evidence was.
type gpuFinderFoldCandidate struct {
	Typ        int
	Direction  int
	Centre     core.PointF
	ModuleSize float64
}

// gpuFinderFoldSelection is what the selection stage made of the folded list:
// the four chosen patterns, the four the prune saw, and the counters the scan
// stats record. Pre is kept because what the prune removes cannot be recovered
// afterwards, and "a true corner was deleted for being rarer than a background
// blob" is indistinguishable from "the corner was never found" without it.
type gpuFinderFoldSelection struct {
	Patterns  [4]FinderPattern
	Pre       [4]FinderPattern
	Preprune  [4]int
	Preselect [4]int
	Selected  [4]int
	Missing   int
}

// gpuFinderFoldResult is the merged pattern list and the counts the caller
// tracks alongside it.
type gpuFinderFoldResult struct {
	Patterns  []FinderPattern
	Weak      []FinderPattern
	Selection gpuFinderFoldSelection
	TypeCount [4]int
	// FamilyPool is the union of folded candidates over the directions run
	// since the last reset, and PoolTypes the finder types in it. PoolDropped
	// counts entries the pool had no room for; the host pool has no bound, so a
	// nonzero drop means the union no longer answers the same question.
	FamilyPool []FinderPattern
	// ContextualPool is the union of grouped weak seeds over the same
	// directions, and PoolTypes the finder types in it - the mask the
	// selection's prune reads to decide whether an absent type was still
	// crossed repeatedly somewhere.
	ContextualPool []FinderPattern
	PoolTypes      [4]bool
	PoolDropped    int
	// Corner is the fourth corner completed from the pool, when exactly one was
	// absent. The local seek that would follow a failed pool search is not on
	// the device: it reads source pixels rather than any list here.
	Corner gpuFinderCornerResult
	// CrossSurvivors counts admitted survivors per type, which is not the same
	// as TypeCount: repeated crossings of one physical finder merge into a
	// single accumulated pattern but each still counts as a survivor.
	CrossSurvivors [4]int
	// Consumed is how far into the candidate sequence the fold got before the
	// stop rule abandoned it, and Deferred counts outcomes whose colour verdict
	// the chain never stamped. Either being nonzero means this route did not
	// see the whole direction.
	Consumed int
	Deferred int
	// Dropped counts candidates that found no slot because the list was full.
	// With the consumer's stop applied the fold cannot reach that bound, so a
	// drop means the stop was not in force and the list silently truncated.
	Dropped int
}

// FoldFinderCandidates orders directional candidates and merges them into the
// accumulated pattern list, both where they already lie.
//
// The ordering is not a convenience for the fold, it is what makes the fold's
// answer reproducible: a merge moves the entry it merged into, so the merged
// set is a function of the sequence, and the compaction that produced these
// candidates reserved its slots through an atomic with no defined order. The
// two stages are one submission because nothing between them is a host
// decision.
//
// The arithmetic is f32 here and f64 on the host, so merged centres can differ
// in the last place and a candidate sitting exactly on the merge radius can
// fall the other way. That is the accepted trade for this route - decode
// outcomes are the gate, not bit parity - and it is why the comparison against
// the host is a tolerance and the census is what actually decides.
func (resident *gpuResidentBinarizer) FoldFinderCandidates(
	candidates []gpuFinderFoldCandidate,
	printPass bool,
	contextualTypes [4]bool,
) (gpuFinderFoldResult, error) {
	var result gpuFinderFoldResult
	if resident == nil || resident.closed || resident.foldBindings == nil {
		return result, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	if len(candidates) > gpuFinderDirectionalCompactCapacity {
		return result, fmt.Errorf("jabcode: GPU finder fold takes up to %d candidates, got %d",
			gpuFinderDirectionalCompactCapacity, len(candidates))
	}
	if len(candidates) == 0 {
		return result, nil
	}

	resident.mu.Lock()
	defer resident.mu.Unlock()

	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return result, fmt.Errorf("jabcode: create GPU finder fold recorder: %w", err)
	}
	defer recorder.Abort()

	padded := 1
	for padded < len(candidates) {
		padded <<= 1
	}
	params := foldParamsBlock(len(candidates), padded, printPass, contextualTypes, false)
	if err := recorder.Update(resident.foldParams, 0, params); err != nil {
		return result, fmt.Errorf("jabcode: update GPU finder fold parameters: %w", err)
	}
	packed := make([]byte, len(candidates)*gpuFinderFoldCandidateWords*4)
	for i, candidate := range candidates {
		at := i * gpuFinderFoldCandidateWords * 4
		put := func(word int, value uint32) {
			binary.LittleEndian.PutUint32(packed[at+word*4:], value)
		}
		// These arrive already admitted, and a candidate list with no seeds in
		// it is all survivors.
		put(0, chainFlagSurvivor)
		put(1, uint32(candidate.Typ))
		put(2, uint32(int32(candidate.Direction)))
		put(3, math.Float32bits(float32(candidate.Centre.X)))
		put(4, math.Float32bits(float32(candidate.Centre.Y)))
		put(5, math.Float32bits(float32(candidate.ModuleSize)))
	}
	// A recorded buffer update is bounded at 64 KB by the command itself, and a
	// full candidate list is several times that, so it goes in chunks. The
	// chunk size stays word aligned because the whole record layout is.
	const updateChunk = 64 << 10
	for at := 0; at < len(packed); at += updateChunk {
		end := min(at+updateChunk, len(packed))
		if err := recorder.Update(resident.foldCandidates, uint64(at), packed[at:end]); err != nil {
			return result, fmt.Errorf("jabcode: update GPU finder fold candidates: %w", err)
		}
	}
	if err := recorder.Barrier(resident.foldParams, resident.foldCandidates); err != nil {
		return result, fmt.Errorf("jabcode: synchronize GPU finder fold inputs: %w", err)
	}
	return resident.finishFinderFold(recorder, false, true)
}

// FoldFinderOutcomes runs the same chain over the compacted outcomes of one
// direction, where the device chain already left them. count records starting
// at base are assembled into candidates, ordered, folded and selected in a
// single submission, so nothing about the direction crosses the bus on the way
// in.
//
// mirror also brings back the intermediate lists - the folded candidates, the
// weak seeds and both pools. They are what the parity tests compare against the
// host arm and are close to two megabytes together, so the route leaves them
// resident and reads only the selection, the corner and the record words.
func (resident *gpuResidentBinarizer) FoldFinderOutcomes(
	bindings *vulki.BindingSet,
	base, count int,
	frame image.Point,
	printPass bool,
	contextualTypes [4]bool,
	mirror bool,
) (gpuFinderFoldResult, error) {
	var result gpuFinderFoldResult
	if resident == nil || resident.closed || resident.foldBindings == nil {
		return result, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	if count < 0 || count > gpuFinderDirectionalCompactCapacity {
		return result, fmt.Errorf("jabcode: GPU finder assembly takes up to %d outcomes, got %d",
			gpuFinderDirectionalCompactCapacity, count)
	}

	resident.mu.Lock()
	defer resident.mu.Unlock()
	// The fold merges into both pools, so anything already fetched from them
	// describes a state this submission is about to leave behind.
	resident.invalidateFinderPoolMirror()

	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return result, fmt.Errorf("jabcode: create GPU finder assembly recorder: %w", err)
	}
	defer recorder.Abort()

	// The candidate count is not known until the assembly has run, so the two
	// words that carry it are left for the kernel to fill.
	params := foldParamsBlock(0, 0, printPass, contextualTypes, true)
	if err := recorder.Update(resident.foldParams, 0, params); err != nil {
		return result, fmt.Errorf("jabcode: update GPU finder fold parameters: %w", err)
	}
	var assembly [gpuFinderAssemblyWords * 4]byte
	binary.LittleEndian.PutUint32(assembly[0:], uint32(base))
	binary.LittleEndian.PutUint32(assembly[4:], uint32(count))
	if err := recorder.Update(resident.assemblyParams, 0, assembly[:]); err != nil {
		return result, fmt.Errorf("jabcode: update GPU finder assembly parameters: %w", err)
	}
	if err := recorder.Barrier(resident.foldParams, resident.assemblyParams); err != nil {
		return result, fmt.Errorf("jabcode: synchronize GPU finder assembly inputs: %w", err)
	}
	if err := recorder.Dispatch(
		resident.assemblyKernel, bindings, vulki.Workgroups{X: 1, Y: 1, Z: 1},
	); err != nil {
		return result, fmt.Errorf("jabcode: dispatch GPU finder assembly: %w", err)
	}
	stages := []struct {
		params   *vulki.Buffer
		capacity uint32
		minFound uint32
		count    uint32
		mode     uint32
	}{
		{resident.familyPoolParams, gpuFinderFamilyPoolSlots, 0,
			gpuFinderFoldRecordTotal, gpuFinderPoolModeReplace},
		{resident.groupParams, maxContextualFinderSeeds, 0,
			gpuFinderFoldRecordWeakTotal, gpuFinderPoolModeAverage},
		{resident.contextualParams, maxContextualFinderCandidates, minFinderCrossings,
			gpuFinderPoolCount, gpuFinderPoolModeReplace},
	}
	for _, stage := range stages {
		var pool [gpuFinderPoolParamWords * 4]byte
		binary.LittleEndian.PutUint32(pool[0:], stage.capacity)
		binary.LittleEndian.PutUint32(pool[4:], stage.minFound)
		binary.LittleEndian.PutUint32(pool[8:], stage.count)
		binary.LittleEndian.PutUint32(pool[12:], stage.mode)
		if err := recorder.Update(stage.params, 0, pool[:]); err != nil {
			return result, fmt.Errorf("jabcode: update GPU finder pool parameters: %w", err)
		}
	}
	// The seed grouping is local to one direction, so its accumulation starts
	// empty every time. Only the two pools carry over.
	if err := recorder.Fill(resident.contextualGroupsRecord, 0, gpuFinderPoolWords*4, 0); err != nil {
		return result, fmt.Errorf("jabcode: clear GPU finder seed groups: %w", err)
	}
	var cornerParams [gpuFinderCornerParamWords * 4]byte
	binary.LittleEndian.PutUint32(cornerParams[0:], uint32(max(frame.X, 1)))
	binary.LittleEndian.PutUint32(cornerParams[4:], uint32(max(frame.Y, 1)))
	if err := recorder.Update(resident.cornerParams, 0, cornerParams[:]); err != nil {
		return result, fmt.Errorf("jabcode: update GPU finder corner parameters: %w", err)
	}
	if err := recorder.Barrier(
		resident.foldCandidates, resident.foldParams, resident.assemblyRecord,
		resident.familyPoolParams, resident.groupParams, resident.contextualParams,
		resident.contextualGroupsRecord, resident.cornerParams,
	); err != nil {
		return result, fmt.Errorf("jabcode: synchronize GPU finder assembly: %w", err)
	}
	return resident.finishFinderFold(recorder, true, mirror)
}

// MaterializeFinderPool fills in the family candidate union the device
// accumulated, for the host searches that genuinely read every candidate rather
// than the four the selection kept.
//
// The union stays on the device because the route never looks at it: the
// selection, the prune and the corner completion all run where it lies. What
// still reads it is the consensus fallback, which runs only after the per-type
// selection has already failed on every direction, and the diagnostics. Both
// are rare enough that fetching the pool on demand costs less than returning it
// from every fold, and one fetch answers all of them because the pool cannot
// change without another fold or a reset.
func (resident *gpuResidentBinarizer) MaterializeFinderPool() ([]FinderPattern, bool) {
	if resident == nil {
		return nil, false
	}
	resident.mu.Lock()
	defer resident.mu.Unlock()
	if resident.closed || resident.familyPoolBindings == nil {
		return nil, false
	}
	if resident.finderPoolMirrored {
		return resident.finderPoolMirror, true
	}

	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return nil, false
	}
	defer recorder.Abort()
	record := make([]byte, gpuFinderPoolWords*4)
	if err := recorder.Download(resident.familyPoolRecord, 0, record); err != nil {
		return nil, false
	}
	pool := make([]byte, gpuFinderFamilyPoolSlots*gpuFinderFoldPatternWords*4)
	phaseprobe.Count("download.finder_pool", len(record)+len(pool))
	if err := recorder.Download(resident.familyPool, 0, pool); err != nil {
		return nil, false
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return nil, false
	}
	entries, dropped, _, err := parseGPUFinderPool(record, pool, gpuFinderFamilyPoolSlots)
	if err != nil {
		return nil, false
	}
	// The host union has no bound, so a device pool that overflowed is a
	// different set from the one the fallback expects to search. Declining
	// sends the caller to its own accumulation rather than to a truncated one.
	if dropped > 0 {
		return nil, false
	}
	resident.finderPoolMirror = entries
	resident.finderPoolMirrored = true
	return entries, true
}

// invalidateFinderPoolMirror is called wherever the device pool changes. The
// caller holds the lock.
func (resident *gpuResidentBinarizer) invalidateFinderPoolMirror() {
	resident.finderPoolMirror = nil
	resident.finderPoolMirrored = false
}

// ResetFinderPools empties the accumulations that outlive a direction. The
// pools union candidates over the directions of one locate, so they are cleared
// where that locate begins and nowhere else: clearing them per direction would
// leave the missing-corner search with only the direction that lost the corner.
func (resident *gpuResidentBinarizer) ResetFinderPools() error {
	if resident == nil || resident.closed || resident.familyPoolBindings == nil {
		return fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	resident.mu.Lock()
	defer resident.mu.Unlock()
	resident.invalidateFinderPoolMirror()

	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return fmt.Errorf("jabcode: create GPU finder pool reset recorder: %w", err)
	}
	defer recorder.Abort()
	// Only the record is cleared. The entries past the count are unreachable,
	// so wiping them would be a megabyte of writes to prove a bound already
	// holds.
	for _, record := range []*vulki.Buffer{
		resident.familyPoolRecord, resident.contextualPoolRecord,
	} {
		if err := recorder.Fill(record, 0, gpuFinderPoolWords*4, 0); err != nil {
			return fmt.Errorf("jabcode: clear GPU finder pool: %w", err)
		}
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return fmt.Errorf("jabcode: reset GPU finder pools: %w", err)
	}
	return nil
}

// finishFinderFold appends the ordering, fold and selection stages to a
// recording that has already put candidates in place, then reads back what the
// host cannot derive for itself.
func (resident *gpuResidentBinarizer) finishFinderFold(
	recorder *vulki.Recorder,
	assembled, mirror bool,
) (gpuFinderFoldResult, error) {
	var result gpuFinderFoldResult
	if err := recorder.Dispatch(
		resident.sortKernel, resident.sortBindings, vulki.Workgroups{X: 1, Y: 1, Z: 1},
	); err != nil {
		return result, fmt.Errorf("jabcode: dispatch GPU finder order: %w", err)
	}
	if err := recorder.Barrier(resident.foldCandidates); err != nil {
		return result, fmt.Errorf("jabcode: synchronize GPU finder order: %w", err)
	}
	if err := recorder.Dispatch(
		resident.foldKernel, resident.foldBindings, vulki.Workgroups{X: 1, Y: 1, Z: 1},
	); err != nil {
		return result, fmt.Errorf("jabcode: dispatch GPU finder fold: %w", err)
	}
	if err := recorder.Barrier(
		resident.foldPatterns, resident.foldRecord, resident.foldWeak,
	); err != nil {
		return result, fmt.Errorf("jabcode: synchronize GPU finder fold output: %w", err)
	}
	// The pools only exist on the route. FoldFinderCandidates is the harness
	// the ordering, fold and selection stages are held to individually, and
	// letting it accumulate would make each of its cases depend on the ones
	// before it.
	// The pools run in the order the host builds them: the family union, then
	// this direction's seed groups, then the contextual union the selection's
	// prune reads. The last two are a chain, so each waits on the one before it.
	if assembled {
		for _, stage := range []struct {
			bindings *vulki.BindingSet
			outputs  [2]*vulki.Buffer
			name     string
		}{
			{resident.familyPoolBindings,
				[2]*vulki.Buffer{resident.familyPool, resident.familyPoolRecord}, "family pool"},
			{resident.groupBindings,
				[2]*vulki.Buffer{resident.contextualGroups, resident.contextualGroupsRecord},
				"seed grouping"},
			{resident.contextualPoolBindings,
				[2]*vulki.Buffer{resident.contextualPool, resident.contextualPoolRecord},
				"contextual pool"},
		} {
			if err := recorder.Dispatch(
				resident.poolKernel, stage.bindings, vulki.Workgroups{X: 1, Y: 1, Z: 1},
			); err != nil {
				return result, fmt.Errorf("jabcode: dispatch GPU finder %s: %w", stage.name, err)
			}
			if err := recorder.Barrier(stage.outputs[0], stage.outputs[1]); err != nil {
				return result, fmt.Errorf("jabcode: synchronize GPU finder %s: %w", stage.name, err)
			}
		}
	}
	if err := recorder.Dispatch(
		resident.selectKernel, resident.selectBindings, vulki.Workgroups{X: 1, Y: 1, Z: 1},
	); err != nil {
		return result, fmt.Errorf("jabcode: dispatch GPU finder selection: %w", err)
	}
	if err := recorder.Barrier(resident.foldSelection); err != nil {
		return result, fmt.Errorf("jabcode: synchronize GPU finder selection: %w", err)
	}
	// The corner completion reads the selection and the family pool, so it is
	// the last stage and needs both of them settled.
	if assembled {
		if err := recorder.Dispatch(
			resident.cornerKernel, resident.cornerBindings, vulki.Workgroups{X: 1, Y: 1, Z: 1},
		); err != nil {
			return result, fmt.Errorf("jabcode: dispatch GPU finder corner: %w", err)
		}
		if err := recorder.Barrier(resident.cornerRecord); err != nil {
			return result, fmt.Errorf("jabcode: synchronize GPU finder corner: %w", err)
		}
	}

	// What the route needs from a fold is the selection, the completed corner
	// and the record words that say whether either can be trusted. The lists
	// behind those - the candidates, the weak seeds and both pools - are the
	// stages the device now owns, and they are between two thirds and a whole
	// megabyte each. Only a comparison against the host arm asks for them.
	selection := make([]byte, gpuFinderSelectWords*4)
	if err := recorder.Download(resident.foldSelection, 0, selection); err != nil {
		return result, fmt.Errorf("jabcode: record GPU finder selection download: %w", err)
	}
	record := make([]byte, gpuFinderFoldRecordWords*4)
	if err := recorder.Download(resident.foldRecord, 0, record); err != nil {
		return result, fmt.Errorf("jabcode: record GPU finder fold record download: %w", err)
	}
	var patterns []byte
	if mirror {
		patterns = make([]byte, maxFinderPatterns*gpuFinderFoldPatternWords*4)
		if err := recorder.Download(resident.foldPatterns, 0, patterns); err != nil {
			return result, fmt.Errorf("jabcode: record GPU finder fold pattern download: %w", err)
		}
	}
	var assemblyRecord, weak, poolRecord, familyPool []byte
	var contextualRecord, contextualPool, cornerRecord []byte
	if assembled {
		assemblyRecord = make([]byte, gpuFinderAssemblyWords*4)
		if err := recorder.Download(resident.assemblyRecord, 0, assemblyRecord); err != nil {
			return result, fmt.Errorf("jabcode: record GPU finder assembly record download: %w", err)
		}
		poolRecord = make([]byte, gpuFinderPoolWords*4)
		if err := recorder.Download(resident.familyPoolRecord, 0, poolRecord); err != nil {
			return result, fmt.Errorf("jabcode: record GPU finder pool record download: %w", err)
		}
		contextualRecord = make([]byte, gpuFinderPoolWords*4)
		if err := recorder.Download(resident.contextualPoolRecord, 0, contextualRecord); err != nil {
			return result, fmt.Errorf("jabcode: record GPU finder contextual record download: %w", err)
		}
		cornerRecord = make([]byte, gpuFinderCornerWords*4)
		if err := recorder.Download(resident.cornerRecord, 0, cornerRecord); err != nil {
			return result, fmt.Errorf("jabcode: record GPU finder corner download: %w", err)
		}
		if mirror {
			weak = make([]byte, maxContextualFinderSeeds*gpuFinderFoldPatternWords*4)
			if err := recorder.Download(resident.foldWeak, 0, weak); err != nil {
				return result, fmt.Errorf("jabcode: record GPU finder weak seed download: %w", err)
			}
			familyPool = make([]byte, gpuFinderFamilyPoolSlots*gpuFinderFoldPatternWords*4)
			if err := recorder.Download(resident.familyPool, 0, familyPool); err != nil {
				return result, fmt.Errorf("jabcode: record GPU finder family pool download: %w", err)
			}
			contextualPool = make([]byte, maxContextualFinderCandidates*gpuFinderFoldPatternWords*4)
			if err := recorder.Download(resident.contextualPool, 0, contextualPool); err != nil {
				return result, fmt.Errorf("jabcode: record GPU finder contextual pool download: %w", err)
			}
		}
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return result, fmt.Errorf("jabcode: run GPU finder fold: %w", err)
	}
	result, err := parseGPUFinderFold(record, patterns, weak)
	if err != nil {
		return result, err
	}
	if assemblyRecord != nil {
		result.Deferred = int(binary.LittleEndian.Uint32(
			assemblyRecord[gpuFinderAssemblyDeferred*4:]))
	}
	if poolRecord != nil {
		result.FamilyPool, result.PoolDropped, _, err =
			parseGPUFinderPool(poolRecord, familyPool, gpuFinderFamilyPoolSlots)
		if err != nil {
			return result, err
		}
		var dropped int
		result.ContextualPool, dropped, result.PoolTypes, err =
			parseGPUFinderPool(contextualRecord, contextualPool, maxContextualFinderCandidates)
		if err != nil {
			return result, err
		}
		result.PoolDropped += dropped
	}
	if cornerRecord != nil {
		result.Corner = parseGPUFinderCorner(cornerRecord)
	}
	result.Selection = parseGPUFinderSelection(selection)
	return result, nil
}

// gpuFinderCornerResult is the fourth corner the device completed: where it came
// from, which of the four it is, and whether the completion is usable at all.
// Usable is separate from the source because an estimate that lands outside the
// frame is still a construction, and it is one the caller must not sample.
type gpuFinderCornerResult struct {
	Source  CornerSource
	Miss    int
	OK      bool
	Pattern FinderPattern
	// Alternatives are the contextual candidates ranked for the corner the
	// construction left standing, best first. They are hypotheses rather than a
	// choice: the strict chain stays the admission boundary, so each is a
	// complete quad the caller may try in turn.
	Alternatives []FinderPattern
}

func parseGPUFinderCorner(record []byte) gpuFinderCornerResult {
	word := func(index int) uint32 {
		return binary.LittleEndian.Uint32(record[index*4:])
	}
	pattern := func(at int) FinderPattern {
		return FinderPattern{
			Typ:        int(word(at)),
			direction:  int(int32(word(at + 1))),
			Center:     core.PointF{X: foldFloat(record, at+2), Y: foldFloat(record, at+3)},
			ModuleSize: foldFloat(record, at+4),
			FoundCount: int(word(at + 5)),
		}
	}
	out := gpuFinderCornerResult{
		Source:  CornerSource(word(gpuFinderCornerSource)),
		Miss:    int(int32(word(gpuFinderCornerMiss))),
		OK:      word(gpuFinderCornerOK) != 0,
		Pattern: pattern(gpuFinderCornerPattern),
	}
	count := min(int(word(gpuFinderCornerAlternatives)), maxContextualFinderQuads)
	for i := range count {
		out.Alternatives = append(out.Alternatives,
			pattern(gpuFinderCornerAlternative+i*gpuFinderFoldPatternWords))
	}
	return out
}

func parseGPUFinderPool(record, pool []byte, capacity int) ([]FinderPattern, int, [4]bool, error) {
	var types [4]bool
	count := int(binary.LittleEndian.Uint32(record[gpuFinderPoolCount*4:]))
	if count > capacity {
		return nil, 0, types, fmt.Errorf("jabcode: GPU finder pool reported %d entries", count)
	}
	mask := binary.LittleEndian.Uint32(record[gpuFinderPoolTypeMask*4:])
	for typ := range types {
		types[typ] = mask&(1<<typ) != 0
	}
	dropped := int(binary.LittleEndian.Uint32(record[gpuFinderPoolDropped*4:]))
	if pool == nil {
		// The record is a few words and the pool behind it is most of a
		// megabyte, so the route reads the counts and leaves the entries where
		// they are. Only a comparison against the host wants them here.
		return nil, dropped, types, nil
	}
	entries := make([]FinderPattern, count)
	for i := range entries {
		at := i * gpuFinderFoldPatternWords
		entries[i] = FinderPattern{
			Typ:        int(binary.LittleEndian.Uint32(pool[at*4:])),
			direction:  int(int32(binary.LittleEndian.Uint32(pool[(at+1)*4:]))),
			Center:     core.PointF{X: foldFloat(pool, at+2), Y: foldFloat(pool, at+3)},
			ModuleSize: foldFloat(pool, at+4),
			FoundCount: int(binary.LittleEndian.Uint32(pool[(at+5)*4:])),
		}
	}
	return entries, dropped, types, nil
}

// foldParamsBlock builds the parameter block the ordering, fold and selection
// stages share. The two bounds the fold enforces are the host's own and are
// written here rather than restated in the shader, so each has one definition.
func foldParamsBlock(
	count, padded int,
	printPass bool,
	contextualTypes [4]bool,
	poolTypes bool,
) []byte {
	params := make([]byte, gpuFinderFoldParamWords*4)
	binary.LittleEndian.PutUint32(params[0:], uint32(count))
	binary.LittleEndian.PutUint32(params[4:], uint32(padded))
	if printPass {
		binary.LittleEndian.PutUint32(params[8:], 1)
	}
	contextual := uint32(0)
	for typ, ok := range contextualTypes {
		if ok {
			contextual |= 1 << typ
		}
	}
	binary.LittleEndian.PutUint32(params[12:], contextual)
	binary.LittleEndian.PutUint32(params[16:], uint32(maxFinderPatterns-1))
	binary.LittleEndian.PutUint32(params[20:], uint32(maxContextualFinderSeeds))
	if poolTypes {
		binary.LittleEndian.PutUint32(params[24:], 1)
	}
	return params
}

func parseGPUFinderSelection(selection []byte) gpuFinderFoldSelection {
	var out gpuFinderFoldSelection
	word := func(index int) uint32 {
		return binary.LittleEndian.Uint32(selection[index*4:])
	}
	pattern := func(base, slot int) FinderPattern {
		at := base + slot*gpuFinderFoldPatternWords
		return FinderPattern{
			Typ:        int(word(at)),
			direction:  int(int32(word(at + 1))),
			Center:     core.PointF{X: foldFloat(selection, at+2), Y: foldFloat(selection, at+3)},
			ModuleSize: foldFloat(selection, at+4),
			FoundCount: int(word(at + 5)),
		}
	}
	for typ := range 4 {
		out.Preprune[typ] = int(word(gpuFinderSelectPreprune + typ))
		out.Preselect[typ] = int(word(gpuFinderSelectPreselect + typ))
		out.Selected[typ] = int(word(gpuFinderSelectSelected + typ))
		out.Patterns[typ] = pattern(gpuFinderSelectPatterns, typ)
		out.Pre[typ] = pattern(gpuFinderSelectPrePatterns, typ)
	}
	out.Missing = int(word(gpuFinderSelectMissing))
	return out
}

func parseGPUFinderFold(record, patterns, weak []byte) (gpuFinderFoldResult, error) {
	var result gpuFinderFoldResult
	word := func(buf []byte, index int) uint32 {
		return binary.LittleEndian.Uint32(buf[index*4:])
	}
	total := int(word(record, gpuFinderFoldRecordTotal))
	if total > maxFinderPatterns {
		return result, fmt.Errorf("jabcode: GPU finder fold reported %d patterns", total)
	}
	weakTotal := int(word(record, gpuFinderFoldRecordWeakTotal))
	if weakTotal > maxContextualFinderSeeds {
		return result, fmt.Errorf("jabcode: GPU finder fold reported %d weak seeds", weakTotal)
	}
	result.Dropped = int(word(record, gpuFinderFoldRecordDropped))
	result.Consumed = int(word(record, gpuFinderFoldRecordConsumed))
	for typ := range result.TypeCount {
		result.TypeCount[typ] = int(word(record, gpuFinderFoldRecordTypeCount+typ))
		result.CrossSurvivors[typ] = int(word(record, gpuFinderFoldRecordCrossSurvivors+typ))
	}
	if patterns != nil {
		result.Patterns = make([]FinderPattern, total)
		for i := range result.Patterns {
			at := i * gpuFinderFoldPatternWords
			result.Patterns[i] = FinderPattern{
				Typ:        int(word(patterns, at)),
				direction:  int(int32(word(patterns, at+1))),
				Center:     core.PointF{X: foldFloat(patterns, at+2), Y: foldFloat(patterns, at+3)},
				ModuleSize: foldFloat(patterns, at+4),
				FoundCount: int(word(patterns, at+5)),
			}
		}
	}
	if weak == nil {
		return result, nil
	}
	// A weak seed is one crossing: the fold never merges these, and the
	// grouping that gives them found-counts runs after it.
	result.Weak = make([]FinderPattern, weakTotal)
	for i := range result.Weak {
		at := i * gpuFinderFoldPatternWords
		result.Weak[i] = FinderPattern{
			Typ:        int(word(weak, at)),
			direction:  int(int32(word(weak, at+1))),
			Center:     core.PointF{X: foldFloat(weak, at+2), Y: foldFloat(weak, at+3)},
			ModuleSize: foldFloat(weak, at+4),
			FoundCount: int(word(weak, at+5)),
		}
	}
	return result, nil
}

func foldFloat(buf []byte, index int) float64 {
	return float64(math.Float32frombits(binary.LittleEndian.Uint32(buf[index*4:])))
}
