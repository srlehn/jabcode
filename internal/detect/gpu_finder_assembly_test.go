//go:build !js

package detect

import (
	"cmp"
	"encoding/binary"
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
				bindings, base, len(outcomes), false, [4]bool{})
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
		got, err = resident.FoldFinderOutcomes(bindings, base, len(outcomes), false, [4]bool{})
		if err != nil {
			t.Fatalf("device assembly: %v", err)
		}
		if err := bindings.Close(); err != nil {
			t.Errorf("close assembly bindings: %v", err)
		}
		state, _ := hostConsumeOutcomes(outcomes)
		host.accumulateFamilyCandidates(FinderFamilyCurrent, state.fps[:state.total])
	}

	want := host.familyPassCandidates[FinderFamilyCurrent]
	if len(want) == 0 {
		t.Fatal("the host pooled nothing, so the comparison proved nothing")
	}
	if got.PoolDropped != 0 {
		t.Errorf("device pool dropped %d entries", got.PoolDropped)
	}
	var wantTypes [4]bool
	for _, candidate := range want {
		wantTypes[candidate.Typ] = true
	}
	if got.PoolTypes != wantTypes {
		t.Errorf("pool types %v, want %v", got.PoolTypes, wantTypes)
	}
	comparePatternLists(t, "pooled", got.FamilyPool, want)
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
		got, err := resident.FoldFinderOutcomes(bindings, 0, len(outcomes), false, [4]bool{})
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
		got, err := resident.FoldFinderOutcomes(bindings, 0, len(outcomes), false, [4]bool{})
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
	got, err := resident.FoldFinderOutcomes(bindings, 0, len(outcomes), false, [4]bool{})
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
