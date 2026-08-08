//go:build !js

package detect

import (
	"cmp"
	"encoding/binary"
	"image"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
)

// outcomeSet builds one direction's compacted chain outcomes with the mix a
// real sweep produces: clustered survivors that merge, scattered singletons,
// contextual seeds that never merge, and records the chain already rejected.
//
// The rejected ones are the point of the generator. A set of admitted records
// only would prove the fold works; it would not prove the assembly applies the
// same admission the host applies, and admitting one extra crossing shifts
// every merge after it.
func outcomeSet(seed uint64, sites, perSite, singles int) []finderChainOutcome {
	rng := rand.New(rand.NewPCG(seed, 0x2545f491))
	narrow := func(v float64) float64 { return float64(float32(v)) }
	// Colour is stamped on every record because the resident chain always has a
	// colour source. A record without it is the one case the device cannot
	// judge, and it has its own test.
	admit := func(typ int) uint32 {
		flags := uint32(chainFlagSurvivor | chainFlagColorEvaluated)
		if typ == fp1 || typ == fp2 {
			flags |= chainFlagColorOK
		}
		return flags
	}
	outcomes := make([]finderChainOutcome, 0, sites*perSite+singles)
	for site := range sites {
		cx := 40 + float64(site%4)*300 + rng.Float64()*20
		cy := 40 + float64(site/4)*300 + rng.Float64()*20
		module := 3 + rng.Float64()*4
		typ := site % 4
		for cross := range perSite + site {
			flags := admit(typ)
			switch {
			case cross%7 == 3:
				// Crossed enough to be worth remembering, not enough to pass the
				// standalone cross-check.
				flags = chainFlagContextualSeed | chainFlagColorEvaluated | chainFlagColorOK
			case cross%11 == 5:
				flags = chainFlagBranchBlue
			case (typ == fp1 || typ == fp2) && cross%13 == 7:
				flags = chainFlagSurvivor | chainFlagColorEvaluated
			}
			outcomes = append(outcomes, finderChainOutcome{
				flags:      flags,
				typ:        typ,
				direction:  rng.IntN(5) - 2,
				centerX:    narrow(cx + (rng.Float64()-0.5)*module),
				centerY:    narrow(cy + (rng.Float64()-0.5)*module),
				moduleSize: narrow(module + (rng.Float64()-0.5)*0.5),
			})
		}
	}
	for i := range singles {
		typ := rng.IntN(4)
		flags := admit(typ)
		if i%5 == 2 {
			flags = chainFlagContextualSeed | chainFlagColorEvaluated | chainFlagColorOK
		}
		outcomes = append(outcomes, finderChainOutcome{
			flags:      flags,
			typ:        typ,
			direction:  rng.IntN(5) - 2,
			centerX:    narrow(rng.Float64() * 1600),
			centerY:    narrow(rng.Float64() * 1200),
			moduleSize: narrow(2 + rng.Float64()*8),
		})
	}
	rng.Shuffle(len(outcomes), func(i, j int) {
		outcomes[i], outcomes[j] = outcomes[j], outcomes[i]
	})
	return outcomes
}

func packFinderOutcomes(outcomes []finderChainOutcome) []byte {
	packed := make([]byte, len(outcomes)*gpuFinderChainOutcomeWords*4)
	for i, outcome := range outcomes {
		at := i * gpuFinderChainOutcomeWords * 4
		put := func(word int, value uint32) {
			binary.LittleEndian.PutUint32(packed[at+word*4:], value)
		}
		put(0, outcome.flags)
		put(1, uint32(outcome.typ))
		put(2, uint32(int32(outcome.direction)))
		put(3, math.Float32bits(float32(outcome.centerX)))
		put(4, math.Float32bits(float32(outcome.centerY)))
		put(5, math.Float32bits(float32(outcome.moduleSize)))
	}
	return packed
}

// hostConsumeOutcomes runs the route the device is replacing: the sort
// sweepDirectionalFamily applies to a device sweep, then
// consumeDirectionalFamilyOutcomes itself rather than a restatement of it.
func hostConsumeOutcomes(outcomes []finderChainOutcome) (primaryFamilyScan, [4]int) {
	hits := make([]finderDirHit, len(outcomes))
	for i, outcome := range outcomes {
		hits[i] = finderDirHit{
			centre:     core.PointF{X: outcome.centerX, Y: outcome.centerY},
			module:     outcome.moduleSize,
			outcome:    outcome,
			chained:    true,
			summarized: true,
		}
	}
	slices.SortFunc(hits, func(a, b finderDirHit) int {
		if c := cmp.Compare(a.centre.Y, b.centre.Y); c != 0 {
			return c
		}
		if c := cmp.Compare(a.centre.X, b.centre.X); c != 0 {
			return c
		}
		return cmp.Compare(a.module, b.module)
	})
	d := &PrimaryDetector{Stats: DetectorStats{Passes: []FinderPassStats{{}}}}
	state := newPrimaryFamilyScan()
	d.consumeDirectionalFamilyOutcomes(newScanDirection(0), hits, &state)
	return state, d.pass().CrossSurvivors
}

// uploadOutcomes puts one direction's outcomes in a device buffer shaped like
// the region the chain preserves per direction, so the assembly reads them at
// an offset rather than from the front of a buffer sized to fit.
func uploadOutcomes(t *testing.T, device *vulki.Device, slot int, packed []byte) *vulki.Buffer {
	t.Helper()
	buffer, err := device.NewBuffer(gpuFinderDirectionalBatchMax * gpuFinderDirectionalOutcomeBytes)
	if err != nil {
		t.Fatalf("allocate outcome buffer: %v", err)
	}
	t.Cleanup(func() {
		if err := buffer.Close(); err != nil {
			t.Errorf("close outcome buffer: %v", err)
		}
	})
	recorder, err := device.NewRecorder()
	if err != nil {
		t.Fatalf("create outcome recorder: %v", err)
	}
	defer recorder.Abort()
	const updateChunk = 64 << 10
	base := uint64(slot * gpuFinderDirectionalOutcomeBytes)
	for at := 0; at < len(packed); at += updateChunk {
		end := min(at+updateChunk, len(packed))
		if err := recorder.Update(buffer, base+uint64(at), packed[at:end]); err != nil {
			t.Fatalf("update outcome buffer: %v", err)
		}
	}
	if err := recorder.SubmitAndWait(); err != nil {
		t.Fatalf("upload outcomes: %v", err)
	}
	return buffer
}

// TestGPUFinderAssemblyMatchesHost holds the whole device path - admission,
// order-preserving compaction, ordering, fold - to the host consumer over the
// same outcomes.
//
// The comparison is entry by entry in accumulated order and covers both lists.
// The weak list is not incidental: a seed's place in it depends on how many
// survivors preceded it, so a weak list that agrees as a set but not in order
// would mean the two routes read the sequence differently.
func TestGPUFinderAssemblyMatchesHost(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	resident, err := newGPUResidentBinarizerWithDevice(device, 64, 64)
	if err != nil {
		_ = device.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		if err := resident.Close(); err != nil {
			t.Errorf("close resident GPU binarizer: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close Vulkan device: %v", err)
		}
	})

	sets := []struct {
		name                    string
		seed                    uint64
		slot                    int
		sites, perSite, singles int
	}{
		{"clustered", 1, 0, 8, 40, 0},
		{"scattered", 2, 2, 0, 0, 200},
		{"mixed", 3, 4, 12, 25, 60},
		{"one site", 4, 1, 1, 4, 0},
		{"dense clusters", 5, 3, 16, 200, 30},
	}
	for _, set := range sets {
		t.Run(set.name, func(t *testing.T) {
			// Each case is one locate. The pools outlive a direction, so a case
			// that inherited the last one's would be comparing against a host
			// that started empty.
			if err := resident.ResetFinderPools(); err != nil {
				t.Fatalf("reset pools: %v", err)
			}
			outcomes := outcomeSet(set.seed, set.sites, set.perSite, set.singles)
			buffer := uploadOutcomes(t, device, set.slot, packFinderOutcomes(outcomes))
			bindings, err := resident.newFinderAssemblyBindings(buffer)
			if err != nil {
				t.Fatalf("bind assembly: %v", err)
			}
			defer func() {
				if err := bindings.Close(); err != nil {
					t.Errorf("close assembly bindings: %v", err)
				}
			}()

			base := set.slot * gpuFinderDirectionalCompactCapacity
			got, err := resident.FoldFinderOutcomes(
				bindings, base, len(outcomes), image.Pt(2000, 2000), false, [4]bool{})
			if err != nil {
				t.Fatalf("device assembly: %v", err)
			}
			state, crossSurvivors := hostConsumeOutcomes(outcomes)
			want := state.fps[:state.total]
			if len(want) == 0 || len(state.weak) == 0 {
				t.Fatal("the host consumed nothing into one of its lists, so the comparison proved nothing")
			}
			if got.Deferred != 0 {
				t.Errorf("device deferred %d outcomes it should have judged", got.Deferred)
			}
			if got.Dropped != 0 {
				t.Errorf("device dropped %d candidates", got.Dropped)
			}
			if got.CrossSurvivors != crossSurvivors {
				t.Errorf("survivor counts %v, want %v", got.CrossSurvivors, crossSurvivors)
			}
			if got.TypeCount != state.typeCount {
				t.Errorf("type counts %v, want %v", got.TypeCount, state.typeCount)
			}
			comparePatternLists(t, "pattern", got.Patterns, want)
			comparePatternLists(t, "seed", got.Weak, state.weak)
		})
	}
}

func comparePatternLists(t *testing.T, kind string, got, want []FinderPattern) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("device produced %d %ss, host %d", len(got), kind, len(want))
	}
	for i := range want {
		w, g := want[i], got[i]
		if g.Typ != w.Typ || g.FoundCount != w.FoundCount || g.direction != w.direction {
			t.Errorf("%s %d: typ=%d found=%d dir=%d, want typ=%d found=%d dir=%d",
				kind, i, g.Typ, g.FoundCount, g.direction, w.Typ, w.FoundCount, w.direction)
			continue
		}
		if math.Abs(g.Center.X-w.Center.X) > foldTolerance(w.Center.X, w.FoundCount) ||
			math.Abs(g.Center.Y-w.Center.Y) > foldTolerance(w.Center.Y, w.FoundCount) ||
			math.Abs(g.ModuleSize-w.ModuleSize) > foldTolerance(w.ModuleSize, w.FoundCount) {
			t.Errorf("%s %d: centre (%.6f,%.6f) module %.6f, want (%.6f,%.6f) module %.6f",
				kind, i, g.Center.X, g.Center.Y, g.ModuleSize,
				w.Center.X, w.Center.Y, w.ModuleSize)
		}
	}
}

// TestGPUFinderPoolMatchesHost holds the device pool to
// accumulateFamilyCandidates over a sequence of directions.
//
// A pool is worth its own test because its merge is not the fold's: a matched
// entry is replaced outright by a better-supported newcomer rather than averaged
// with it, and the entry it replaces moves, so every later match depends on it.
// The comparison is entry by entry for that reason.
//
// Several directions run into one pool because a pool that only ever saw one is
// a fold with extra steps. What this has to establish is that the second
// direction finds the first one's entries.
func TestGPUFinderPoolMatchesHost(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	resident, err := newGPUResidentBinarizerWithDevice(device, 64, 64)
	if err != nil {
		_ = device.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		if err := resident.Close(); err != nil {
			t.Errorf("close resident GPU binarizer: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close Vulkan device: %v", err)
		}
	})
	if err := resident.ResetFinderPools(); err != nil {
		t.Fatalf("reset pools: %v", err)
	}

	host := &PrimaryDetector{}
	var got gpuFinderFoldResult
	var wantContextual []FinderPattern
	// Seeds one apart give each direction a set that overlaps the last one
	// heavily and differs at the edges, which is what a real sweep of the same
	// frame from another base produces.
	for direction, seed := range []uint64{11, 12, 13, 14} {
		outcomes := outcomeSet(seed, 10, 20, 40)
		buffer := uploadOutcomes(t, device, direction%gpuFinderDirectionalBatchMax,
			packFinderOutcomes(outcomes))
		bindings, err := resident.newFinderAssemblyBindings(buffer)
		if err != nil {
			t.Fatalf("bind assembly: %v", err)
		}
		base := (direction % gpuFinderDirectionalBatchMax) * gpuFinderDirectionalCompactCapacity
		got, err = resident.FoldFinderOutcomes(bindings, base, len(outcomes), image.Pt(2000, 2000), false, [4]bool{})
		if err != nil {
			t.Fatalf("device assembly: %v", err)
		}
		if err := bindings.Close(); err != nil {
			t.Errorf("close assembly bindings: %v", err)
		}
		state, _ := hostConsumeOutcomes(outcomes)
		host.accumulateFamilyCandidates(FinderFamilyCurrent, state.fps[:state.total])
		host.accumulateContextualFinderCandidates(contextualFinderCandidates(state.weak))
		wantContextual = host.contextualCandidates
	}

	want := host.familyPassCandidates[FinderFamilyCurrent]
	if len(want) == 0 || len(wantContextual) == 0 {
		t.Fatal("the host pooled nothing into one of its pools, so the comparison proved nothing")
	}
	if got.PoolDropped != 0 {
		t.Errorf("device pools dropped %d entries", got.PoolDropped)
	}
	var wantTypes [4]bool
	for _, candidate := range wantContextual {
		wantTypes[candidate.Typ] = true
	}
	if got.PoolTypes != wantTypes {
		t.Errorf("contextual types %v, want %v", got.PoolTypes, wantTypes)
	}
	comparePatternLists(t, "pooled", got.FamilyPool, want)
	comparePatternLists(t, "contextual", got.ContextualPool, wantContextual)
}

// TestGPUFinderPoolTypesReachThePrune pins that the contextual pool the device
// builds is the one the selection's prune reads.
//
// The prune stops early when the single absent type was crossed repeatedly
// somewhere, and that is the only thing the contextual types decide, so it is
// the only construction that can show them arriving. The set here has three
// types with survivors - one of them outvoted - and a fourth present only as
// grouped seeds. With the pool reaching the prune the outvoted type is kept and
// one type is missing; without it the outvoted type is pruned too and two are.
func TestGPUFinderPoolTypesReachThePrune(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	resident, err := newGPUResidentBinarizerWithDevice(device, 64, 64)
	if err != nil {
		_ = device.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		if err := resident.Close(); err != nil {
			t.Errorf("close resident GPU binarizer: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close Vulkan device: %v", err)
		}
	})

	var outcomes []finderChainOutcome
	crossing := func(flags uint32, typ int, x, y float64) finderChainOutcome {
		return finderChainOutcome{
			flags: flags | chainFlagColorEvaluated | chainFlagColorOK, typ: typ,
			direction: 1, centerX: x, centerY: y, moduleSize: 4,
		}
	}
	// Far enough apart that no site merges into another.
	for range 20 {
		outcomes = append(outcomes,
			crossing(chainFlagSurvivor, fp0, 100, 100),
			crossing(chainFlagSurvivor, fp1, 900, 100))
	}
	for range 5 {
		outcomes = append(outcomes, crossing(chainFlagSurvivor, fp2, 900, 900))
	}
	for range 6 {
		outcomes = append(outcomes, crossing(chainFlagContextualSeed, fp3, 100, 900))
	}

	buffer := uploadOutcomes(t, device, 0, packFinderOutcomes(outcomes))
	bindings, err := resident.newFinderAssemblyBindings(buffer)
	if err != nil {
		t.Fatalf("bind assembly: %v", err)
	}
	defer func() {
		if err := bindings.Close(); err != nil {
			t.Errorf("close assembly bindings: %v", err)
		}
	}()
	if err := resident.ResetFinderPools(); err != nil {
		t.Fatalf("reset pools: %v", err)
	}
	got, err := resident.FoldFinderOutcomes(bindings, 0, len(outcomes), image.Pt(2000, 2000), false, [4]bool{})
	if err != nil {
		t.Fatalf("device assembly: %v", err)
	}

	if !got.PoolTypes[fp3] {
		t.Fatalf("the seeds did not reach the contextual pool, types %v", got.PoolTypes)
	}
	if got.Selection.Preselect[fp2] != 5 || got.Selection.Preselect[fp3] != 0 {
		t.Fatalf("the set did not come out outvoted-with-one-absent: preselect %v",
			got.Selection.Preselect)
	}
	if got.Selection.Missing != 1 {
		t.Errorf("selection is missing %d types, want 1 - the outvoted type was pruned, "+
			"so the contextual pool did not reach the prune", got.Selection.Missing)
	}
	if got.Selection.Selected[fp2] != 5 {
		t.Errorf("the outvoted type kept %d crossings, want the 5 it was found with",
			got.Selection.Selected[fp2])
	}

	// The host reaches the same verdict from its own pools, which is what makes
	// the numbers above the route's rather than this test's.
	state, _ := hostConsumeOutcomes(outcomes)
	host := &PrimaryDetector{}
	host.accumulateContextualFinderCandidates(contextualFinderCandidates(state.weak))
	var types [4]bool
	for _, candidate := range host.contextualCandidates {
		types[candidate.Typ] = true
	}
	var stats FinderFamilyScanStats
	var pre [4]FinderPattern
	selected := make([]FinderPattern, maxFinderPatterns)
	copy(selected, state.fps[:state.total])
	missing := host.selectBestPatternsFor(
		selected, state.total, state.typeCount[:], types, &stats, &pre)
	if missing != got.Selection.Missing {
		t.Errorf("host selection is missing %d types, device %d", missing, got.Selection.Missing)
	}
	if stats.Selected != got.Selection.Selected {
		t.Errorf("host kept %v, device %v", stats.Selected, got.Selection.Selected)
	}
}

// TestGPUFinderPoolResetEmptiesIt pins that a pool is emptied where a locate
// begins and not per direction. Without the reset the next locate would search
// the previous frame's corners, and with it per direction the missing-corner
// search would see only the direction that lost the corner.
func TestGPUFinderPoolResetEmptiesIt(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	resident, err := newGPUResidentBinarizerWithDevice(device, 64, 64)
	if err != nil {
		_ = device.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		if err := resident.Close(); err != nil {
			t.Errorf("close resident GPU binarizer: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close Vulkan device: %v", err)
		}
	})

	outcomes := outcomeSet(21, 6, 12, 20)
	buffer := uploadOutcomes(t, device, 0, packFinderOutcomes(outcomes))
	bindings, err := resident.newFinderAssemblyBindings(buffer)
	if err != nil {
		t.Fatalf("bind assembly: %v", err)
	}
	defer func() {
		if err := bindings.Close(); err != nil {
			t.Errorf("close assembly bindings: %v", err)
		}
	}()

	run := func() int {
		got, err := resident.FoldFinderOutcomes(bindings, 0, len(outcomes), image.Pt(2000, 2000), false, [4]bool{})
		if err != nil {
			t.Fatalf("device assembly: %v", err)
		}
		return len(got.FamilyPool)
	}
	if err := resident.ResetFinderPools(); err != nil {
		t.Fatalf("reset pools: %v", err)
	}
	first := run()
	if first == 0 {
		t.Fatal("the pool took nothing, so the reset proved nothing")
	}
	// The same direction again adds nothing: every entry matches one already
	// there. A growing pool would mean the merge missed its own entries.
	if again := run(); again != first {
		t.Errorf("the same direction pooled again grew the pool to %d, want %d", again, first)
	}
	if err := resident.ResetFinderPools(); err != nil {
		t.Fatalf("reset pools: %v", err)
	}
	if after := run(); after != first {
		t.Errorf("after a reset the pool holds %d, want the %d one direction fills",
			after, first)
	}
}

// cornerOutcomeSet builds a quad with one corner absent from the selection but
// present in the pool, plus the crossings that put it there.
//
// The corner has to be reachable two ways for this to test anything: absent
// where the selection looks, present where the pool does. A direction whose own
// crossings found it would never call the completion at all, so the pool entry
// is planted by a first direction whose fourth corner is then withheld from the
// second.
func cornerOutcomeSet(centres [4]core.PointF, present [4]bool, crossings int) []finderChainOutcome {
	var outcomes []finderChainOutcome
	for typ := range 4 {
		if !present[typ] {
			continue
		}
		for i := range crossings {
			outcomes = append(outcomes, finderChainOutcome{
				flags:     chainFlagSurvivor | chainFlagColorEvaluated | chainFlagColorOK,
				typ:       typ,
				direction: 1,
				// A sub-module jitter so the merge runs rather than the append
				// path alone, which is what a real crossing sequence looks like.
				centerX:    float64(float32(centres[typ].X + float64(i%3)*0.25)),
				centerY:    float64(float32(centres[typ].Y + float64(i%2)*0.25)),
				moduleSize: 4,
			})
		}
	}
	return outcomes
}

// TestGPUFinderCornerMatchesHost holds the device corner completion to
// estimateMissingPattern over the same partial quad and the same pool.
//
// Both outcomes it can reach are covered: a pooled candidate close enough and
// consistent enough to displace the construction, and a pool with nothing
// eligible in it, which leaves the construction standing. Only the first proves
// the search runs; only the second proves it does not fire on anything.
func TestGPUFinderCornerMatchesHost(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	resident, err := newGPUResidentBinarizerWithDevice(device, 64, 64)
	if err != nil {
		_ = device.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		if err := resident.Close(); err != nil {
			t.Errorf("close resident GPU binarizer: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close Vulkan device: %v", err)
		}
	})

	// A square symbol at module size 4: the finder centres span 100 pixels, so
	// 25 modules of centre span and a legal side size of 32.
	centres := [4]core.PointF{
		{X: 200, Y: 200}, {X: 300, Y: 200}, {X: 300, Y: 300}, {X: 200, Y: 300},
	}
	frame := image.Pt(640, 480)

	for _, set := range []struct {
		name       string
		poolCorner core.PointF
		want       CornerSource
	}{
		// Half a module off the exact construction: inside the noise radius and
		// consistent with the partial quad, so it displaces it.
		{"pooled", core.PointF{X: 202, Y: 300}, CornerPooled},
		// Far enough that no radius reaches it.
		{"constructed", core.PointF{X: 40, Y: 460}, CornerConstructed},
	} {
		t.Run(set.name, func(t *testing.T) {
			if err := resident.ResetFinderPools(); err != nil {
				t.Fatalf("reset pools: %v", err)
			}
			planted := centres
			planted[fp3] = set.poolCorner

			var got gpuFinderFoldResult
			host := &PrimaryDetector{}
			// The bounds check reads the channel geometry and nothing else, so
			// shape-only bitmaps are what the device is given too.
			for i := range host.Ch {
				host.Ch[i] = &core.Bitmap{Width: frame.X, Height: frame.Y, Channels: 1}
			}
			var state primaryFamilyScan
			// The first direction plants all four; the second withholds FP3, so
			// the completion has to find it in what the first left behind.
			for direction, present := range [2][4]bool{
				{true, true, true, true}, {true, true, true, false},
			} {
				outcomes := cornerOutcomeSet(planted, present, 8)
				buffer := uploadOutcomes(t, device, direction, packFinderOutcomes(outcomes))
				bindings, err := resident.newFinderAssemblyBindings(buffer)
				if err != nil {
					t.Fatalf("bind assembly: %v", err)
				}
				base := direction * gpuFinderDirectionalCompactCapacity
				got, err = resident.FoldFinderOutcomes(
					bindings, base, len(outcomes), frame, false, [4]bool{})
				if err != nil {
					t.Fatalf("device assembly: %v", err)
				}
				if err := bindings.Close(); err != nil {
					t.Errorf("close assembly bindings: %v", err)
				}
				state, _ = hostConsumeOutcomes(outcomes)
				host.accumulateFamilyCandidates(FinderFamilyCurrent, state.fps[:state.total])
			}

			if got.Selection.Missing != 1 {
				t.Fatalf("the second direction is missing %d types, want the one withheld",
					got.Selection.Missing)
			}

			// The host runs its own selection and completion over the same
			// evidence. The bitmap is nil, which is the seek declining - the
			// device has no seek either, so the two are comparable exactly here.
			selected := make([]FinderPattern, maxFinderPatterns)
			copy(selected, state.fps[:state.total])
			var stats FinderFamilyScanStats
			var pre [4]FinderPattern
			missing := host.selectBestPatternsFor(
				selected, state.total, state.typeCount[:], [4]bool{}, &stats, &pre)
			if missing != 1 {
				t.Fatalf("host selection is missing %d types, want 1", missing)
			}
			source, miss, ok := estimateMissingPattern(
				func() *core.Bitmap { return nil }, host.Ch, selected,
				host.familyPassCandidates[FinderFamilyCurrent])

			if got.Corner.Source != source {
				t.Errorf("device corner came from %v, host from %v", got.Corner.Source, source)
			}
			if got.Corner.Source != set.want {
				t.Errorf("corner came from %v, want %v", got.Corner.Source, set.want)
			}
			if got.Corner.Miss != miss {
				t.Errorf("device completed corner %d, host %d", got.Corner.Miss, miss)
			}
			if got.Corner.OK != ok {
				t.Errorf("device reports usable=%t, host %t", got.Corner.OK, ok)
			}
			w, g := selected[miss], got.Corner.Pattern
			if g.Typ != w.Typ || g.FoundCount != w.FoundCount || g.direction != w.direction {
				t.Fatalf("corner typ=%d found=%d dir=%d, want typ=%d found=%d dir=%d",
					g.Typ, g.FoundCount, g.direction, w.Typ, w.FoundCount, w.direction)
			}
			if math.Abs(g.Center.X-w.Center.X) > foldTolerance(w.Center.X, w.FoundCount) ||
				math.Abs(g.Center.Y-w.Center.Y) > foldTolerance(w.Center.Y, w.FoundCount) ||
				math.Abs(g.ModuleSize-w.ModuleSize) > foldTolerance(w.ModuleSize, w.FoundCount) {
				t.Errorf("corner (%.6f,%.6f) module %.6f, want (%.6f,%.6f) module %.6f",
					g.Center.X, g.Center.Y, g.ModuleSize,
					w.Center.X, w.Center.Y, w.ModuleSize)
			}
		})
	}
}

// TestGPUFinderCornerAlternativesMatchHost holds the ranked contextual
// hypotheses to contextualFinderQuads over the same construction and pool.
//
// Several candidates are planted at different distances so the ranking has
// something to order rather than one entry to return. They are seeds, not
// survivors: a survivor would be selected as the corner outright and the
// alternatives would never be reached.
func TestGPUFinderCornerAlternativesMatchHost(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	resident, err := newGPUResidentBinarizerWithDevice(device, 64, 64)
	if err != nil {
		_ = device.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		if err := resident.Close(); err != nil {
			t.Errorf("close resident GPU binarizer: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close Vulkan device: %v", err)
		}
	})
	if err := resident.ResetFinderPools(); err != nil {
		t.Fatalf("reset pools: %v", err)
	}

	frame := image.Pt(640, 480)
	var outcomes []finderChainOutcome
	for typ := range 3 {
		centre := [3]core.PointF{{X: 200, Y: 200}, {X: 300, Y: 200}, {X: 300, Y: 300}}[typ]
		for range 8 {
			outcomes = append(outcomes, finderChainOutcome{
				flags:     chainFlagSurvivor | chainFlagColorEvaluated | chainFlagColorOK,
				typ:       typ,
				direction: 1, centerX: centre.X, centerY: centre.Y, moduleSize: 4,
			})
		}
	}
	// Four FP3 seeds spread around the construction at (200,300). Each is
	// crossed a different number of times so support breaks no ties by accident.
	for i, offset := range []core.PointF{
		{X: 0, Y: 0}, {X: 3, Y: -2}, {X: -4, Y: 5}, {X: 2, Y: 6},
	} {
		for range 4 + i {
			outcomes = append(outcomes, finderChainOutcome{
				flags:     chainFlagContextualSeed | chainFlagColorEvaluated | chainFlagColorOK,
				typ:       fp3,
				direction: 1,
				centerX:   200 + offset.X, centerY: 300 + offset.Y, moduleSize: 4,
			})
		}
	}

	buffer := uploadOutcomes(t, device, 0, packFinderOutcomes(outcomes))
	bindings, err := resident.newFinderAssemblyBindings(buffer)
	if err != nil {
		t.Fatalf("bind assembly: %v", err)
	}
	defer func() {
		if err := bindings.Close(); err != nil {
			t.Errorf("close assembly bindings: %v", err)
		}
	}()
	got, err := resident.FoldFinderOutcomes(bindings, 0, len(outcomes), frame, false, [4]bool{})
	if err != nil {
		t.Fatalf("device assembly: %v", err)
	}
	if got.Corner.Source != CornerConstructed {
		t.Fatalf("corner came from %v, want a construction for the alternatives to complete",
			got.Corner.Source)
	}

	host := &PrimaryDetector{}
	for i := range host.Ch {
		host.Ch[i] = &core.Bitmap{Width: frame.X, Height: frame.Y, Channels: 1}
	}
	state, _ := hostConsumeOutcomes(outcomes)
	host.accumulateContextualFinderCandidates(contextualFinderCandidates(state.weak))
	host.accumulateFamilyCandidates(FinderFamilyCurrent, state.fps[:state.total])
	selected := make([]FinderPattern, maxFinderPatterns)
	copy(selected, state.fps[:state.total])
	var stats FinderFamilyScanStats
	var pre [4]FinderPattern
	if missing := host.selectBestPatternsFor(
		selected, state.total, state.typeCount[:], [4]bool{}, &stats, &pre); missing != 1 {
		t.Fatalf("host selection is missing %d types, want 1", missing)
	}
	source, miss, _ := estimateMissingPattern(
		func() *core.Bitmap { return nil }, host.Ch, selected,
		host.familyPassCandidates[FinderFamilyCurrent])
	if source != CornerConstructed {
		t.Fatalf("host corner came from %v, want a construction", source)
	}
	want := contextualFinderQuads(selected, miss, host.contextualCandidates)
	if len(want) < 2 {
		t.Fatalf("the host ranked %d hypotheses, too few to test an ordering", len(want))
	}

	if len(got.Corner.Alternatives) != len(want) {
		t.Fatalf("device ranked %d hypotheses, host %d",
			len(got.Corner.Alternatives), len(want))
	}
	for i := range want {
		w, g := want[i].Patterns[miss], got.Corner.Alternatives[i]
		if g.Typ != w.Typ || g.FoundCount != w.FoundCount || g.direction != w.direction {
			t.Errorf("hypothesis %d: typ=%d found=%d dir=%d, want typ=%d found=%d dir=%d",
				i, g.Typ, g.FoundCount, g.direction, w.Typ, w.FoundCount, w.direction)
			continue
		}
		if math.Abs(g.Center.X-w.Center.X) > foldTolerance(w.Center.X, w.FoundCount) ||
			math.Abs(g.Center.Y-w.Center.Y) > foldTolerance(w.Center.Y, w.FoundCount) {
			t.Errorf("hypothesis %d: centre (%.6f,%.6f), want (%.6f,%.6f)",
				i, g.Center.X, g.Center.Y, w.Center.X, w.Center.Y)
		}
	}
}

// TestGPUFinderAssemblyIsReproducible pins that the same outcomes give the same
// answer every run, which is what the prefix-sum compaction is for: slots
// reserved through an atomic would make the candidate sequence a function of
// scheduling as well as of the frame.
//
// Only records whose three sort keys are exactly equal can show it, since the
// ordering stage decides everything else, and those cannot be aimed at
// deliberately - a set built entirely of ties folds by type alone and would pass
// whatever the sequence was. So this runs a realistic set repeatedly instead,
// and compares both lists and the selection.
func TestGPUFinderAssemblyIsReproducible(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	resident, err := newGPUResidentBinarizerWithDevice(device, 64, 64)
	if err != nil {
		_ = device.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		if err := resident.Close(); err != nil {
			t.Errorf("close resident GPU binarizer: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close Vulkan device: %v", err)
		}
	})

	outcomes := outcomeSet(9, 12, 25, 60)
	buffer := uploadOutcomes(t, device, 0, packFinderOutcomes(outcomes))
	bindings, err := resident.newFinderAssemblyBindings(buffer)
	if err != nil {
		t.Fatalf("bind assembly: %v", err)
	}
	defer func() {
		if err := bindings.Close(); err != nil {
			t.Errorf("close assembly bindings: %v", err)
		}
	}()

	var first gpuFinderFoldResult
	for run := range 4 {
		if err := resident.ResetFinderPools(); err != nil {
			t.Fatalf("reset pools: %v", err)
		}
		got, err := resident.FoldFinderOutcomes(bindings, 0, len(outcomes), image.Pt(2000, 2000), false, [4]bool{})
		if err != nil {
			t.Fatalf("device assembly: %v", err)
		}
		if run == 0 {
			first = got
			if len(first.Patterns) < 2 || len(first.Weak) == 0 {
				t.Fatalf("the set folded to %d patterns and %d seeds, so little was compared",
					len(first.Patterns), len(first.Weak))
			}
			continue
		}
		comparePatternLists(t, "pattern", got.Patterns, first.Patterns)
		comparePatternLists(t, "seed", got.Weak, first.Weak)
		compareSelection(t, got.Selection, first.Selection)
	}
}

// TestGPUFinderAssemblyDefersUnjudgedColour pins the one verdict the device
// cannot reach. An FP1 or FP2 candidate whose colour the chain never evaluated
// needs a source RGB read to decide, so the assembly counts it and leaves it
// out; silently admitting or silently discarding it would both change the fold
// with nothing to say so.
func TestGPUFinderAssemblyDefersUnjudgedColour(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	resident, err := newGPUResidentBinarizerWithDevice(device, 64, 64)
	if err != nil {
		_ = device.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		if err := resident.Close(); err != nil {
			t.Errorf("close resident GPU binarizer: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close Vulkan device: %v", err)
		}
	})

	outcomes := []finderChainOutcome{
		{flags: chainFlagSurvivor | chainFlagColorEvaluated, typ: fp0, centerX: 10, centerY: 10, moduleSize: 4},
		{flags: chainFlagSurvivor, typ: fp1, centerX: 200, centerY: 10, moduleSize: 4},
		{flags: chainFlagSurvivor, typ: fp2, centerX: 400, centerY: 10, moduleSize: 4},
		{flags: chainFlagSurvivor | chainFlagColorEvaluated, typ: fp3, centerX: 600, centerY: 10, moduleSize: 4},
	}
	buffer := uploadOutcomes(t, device, 0, packFinderOutcomes(outcomes))
	bindings, err := resident.newFinderAssemblyBindings(buffer)
	if err != nil {
		t.Fatalf("bind assembly: %v", err)
	}
	defer func() {
		if err := bindings.Close(); err != nil {
			t.Errorf("close assembly bindings: %v", err)
		}
	}()
	got, err := resident.FoldFinderOutcomes(bindings, 0, len(outcomes), image.Pt(2000, 2000), false, [4]bool{})
	if err != nil {
		t.Fatalf("device assembly: %v", err)
	}
	if got.Deferred != 2 {
		t.Errorf("device deferred %d outcomes, want the two unjudged FP1/FP2 records", got.Deferred)
	}
	if len(got.Patterns) != 2 {
		t.Fatalf("device folded %d patterns, want the two whose colour needed no verdict",
			len(got.Patterns))
	}
	for i, typ := range []int{fp0, fp3} {
		if got.Patterns[i].Typ != typ {
			t.Errorf("pattern %d is type %d, want %d", i, got.Patterns[i].Typ, typ)
		}
	}
}
