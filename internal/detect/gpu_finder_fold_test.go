//go:build !js

package detect

import (
	"cmp"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
)

// hostFoldCandidates runs the host's own ordering and fold over a candidate
// list: the same sort key sweepDirectionalFamily applies before consuming a
// device sweep, then saveFinderPattern itself rather than a restatement of it.
func hostFoldCandidates(
	t *testing.T,
	candidates []gpuFinderFoldCandidate,
	printPass bool,
	contextualTypes [4]bool,
) gpuFinderFoldResult {
	t.Helper()
	ordered := append([]gpuFinderFoldCandidate(nil), candidates...)
	slices.SortFunc(ordered, func(a, b gpuFinderFoldCandidate) int {
		if c := cmp.Compare(a.Centre.Y, b.Centre.Y); c != 0 {
			return c
		}
		if c := cmp.Compare(a.Centre.X, b.Centre.X); c != 0 {
			return c
		}
		return cmp.Compare(a.ModuleSize, b.ModuleSize)
	})
	candidates = ordered
	fps := make([]FinderPattern, maxFinderPatterns)
	total := 0
	var typeCount [4]int
	for _, candidate := range candidates {
		// consumeDirectionalFamilyOutcomes' own stop, not the list's length: the
		// consumer abandons the direction one short of the bound, so a fold that
		// filled the list would be folding candidates nothing ever reads.
		if total >= maxFinderPatterns-1 {
			break
		}
		fp := FinderPattern{
			Typ:        candidate.Typ,
			ModuleSize: candidate.ModuleSize,
			Center:     candidate.Centre,
			FoundCount: 1,
			direction:  candidate.Direction,
		}
		saveFinderPattern(&fp, fps, &total, typeCount[:])
	}
	folded := append([]FinderPattern(nil), fps[:total]...)

	// selectBestPatternsFor reduces fps in place, so the selection runs over a
	// copy and the folded list stays comparable.
	selected := make([]FinderPattern, maxFinderPatterns)
	copy(selected, folded)
	var stats FinderFamilyScanStats
	var pre [4]FinderPattern
	d := &PrimaryDetector{printPass: printPass}
	missing := d.selectBestPatternsFor(
		selected, total, typeCount[:], contextualTypes, &stats, &pre)

	out := gpuFinderFoldResult{Patterns: folded, TypeCount: typeCount}
	out.Selection = gpuFinderFoldSelection{
		Pre:       pre,
		Preprune:  stats.Preprune,
		Preselect: stats.Preselect,
		Selected:  stats.Selected,
		Missing:   missing,
	}
	copy(out.Selection.Patterns[:], selected[:4])
	return out
}

// foldCandidateSet builds a candidate sequence with the structure a real sweep
// produces: a handful of true pattern sites seen many times with small jitter,
// plus scattered singletons. Both are needed - the clusters exercise the merge
// and its running average, the singletons exercise the append path and the
// type counts.
// Every coordinate is rounded through float32 because that is what a real
// candidate is: the chain writes f32 and the host widens it. Generating wider
// values would make the two sides sort different numbers and would test a
// difference the pipeline cannot produce.
func foldCandidateSet(seed uint64, sites, perSite, singles int) []gpuFinderFoldCandidate {
	rng := rand.New(rand.NewPCG(seed, 0x9e3779b9))
	narrow := func(v float64) float64 { return float64(float32(v)) }
	candidates := make([]gpuFinderFoldCandidate, 0, sites*perSite+singles)
	for site := range sites {
		cx := 40 + float64(site%4)*300 + rng.Float64()*20
		cy := 40 + float64(site/4)*300 + rng.Float64()*20
		module := 3 + rng.Float64()*4
		typ := site % 4
		// Each site is crossed a different number of times. Two sites of one
		// type crossed equally often leave bestPattern's tie-break to decide,
		// and that tie-break is not determined at this precision - see the
		// module-deviation tie test below.
		for range perSite + site {
			candidates = append(candidates, gpuFinderFoldCandidate{
				Typ:       typ,
				Direction: rng.IntN(5) - 2,
				Centre: core.PointF{
					X: narrow(cx + (rng.Float64()-0.5)*module),
					Y: narrow(cy + (rng.Float64()-0.5)*module),
				},
				ModuleSize: narrow(module + (rng.Float64()-0.5)*0.5),
			})
		}
	}
	for range singles {
		candidates = append(candidates, gpuFinderFoldCandidate{
			Typ:       rng.IntN(4),
			Direction: rng.IntN(5) - 2,
			Centre: core.PointF{
				X: narrow(rng.Float64() * 1600), Y: narrow(rng.Float64() * 1200),
			},
			ModuleSize: narrow(2 + rng.Float64()*8),
		})
	}
	rng.Shuffle(len(candidates), func(i, j int) {
		candidates[i], candidates[j] = candidates[j], candidates[i]
	})
	return candidates
}

// foldTolerance bounds how far a merged value may drift from the host's.
//
// The accumulated centre is a running mean carried in f32 against the host's
// f64, so its error is not a constant: each merge rounds once, and the rounding
// is as likely to go either way, which makes the drift grow with the square
// root of the merge count rather than staying put. One f32 ulp of the value is
// the per-merge step; the factor of two is headroom over that random walk and
// not a number fitted to an observation.
func foldTolerance(value float64, found int) float64 {
	const f32ulp = 1.1920929e-7
	step := max(math.Abs(value)*f32ulp, 1e-7)
	return 2 * math.Sqrt(float64(max(found, 1))) * step
}

// TestGPUFinderFoldMatchesHost holds the device candidate fold to the host's
// saveFinderPattern over the same sequence.
//
// The fold is order-dependent - a merge moves the entry it merged into, so
// every later match depends on it - which is why the comparison is entry by
// entry in the accumulated order rather than as a set. A fold that produced the
// right patterns in the wrong slots would pick a different quad downstream.
//
// The device works in f32 and the host in f64, so centres are compared to a
// tolerance. Counts, types and found-counts are not: those are integers on both
// sides, and a difference in any of them is a different merge decision rather
// than a rounding difference.
func TestGPUFinderFoldMatchesHost(t *testing.T) {
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
		sites, perSite, singles int
	}{
		{"clustered", 1, 8, 40, 0},
		{"scattered", 2, 0, 0, 120},
		{"mixed", 3, 12, 25, 60},
		{"one candidate", 4, 1, 1, 0},
		{"dense clusters", 5, 16, 200, 30},
	}
	for _, set := range sets {
		t.Run(set.name, func(t *testing.T) {
			candidates := foldCandidateSet(set.seed, set.sites, set.perSite, set.singles)
			want := hostFoldCandidates(t, candidates, false, [4]bool{})
			got, err := resident.FoldFinderCandidates(candidates, false, [4]bool{})
			if err != nil {
				t.Fatalf("device fold: %v", err)
			}
			if got.Dropped != 0 {
				t.Fatalf("device fold dropped %d candidates", got.Dropped)
			}
			if len(got.Patterns) != len(want.Patterns) {
				t.Fatalf("device folded to %d patterns, host to %d",
					len(got.Patterns), len(want.Patterns))
			}
			if len(want.Patterns) == 0 {
				t.Fatal("the host fold produced nothing, so the comparison proved nothing")
			}
			if got.TypeCount != want.TypeCount {
				t.Errorf("type counts %v, want %v", got.TypeCount, want.TypeCount)
			}
			merged := 0
			for i := range want.Patterns {
				w, g := want.Patterns[i], got.Patterns[i]
				if w.FoundCount > 1 {
					merged++
				}
				if g.Typ != w.Typ || g.FoundCount != w.FoundCount || g.direction != w.direction {
					t.Errorf("pattern %d: typ=%d found=%d dir=%d, want typ=%d found=%d dir=%d",
						i, g.Typ, g.FoundCount, g.direction, w.Typ, w.FoundCount, w.direction)
					continue
				}
				if math.Abs(g.Center.X-w.Center.X) > foldTolerance(w.Center.X, w.FoundCount) ||
					math.Abs(g.Center.Y-w.Center.Y) > foldTolerance(w.Center.Y, w.FoundCount) ||
					math.Abs(g.ModuleSize-w.ModuleSize) > foldTolerance(w.ModuleSize, w.FoundCount) {
					t.Errorf("pattern %d: centre (%.6f,%.6f) module %.6f, want (%.6f,%.6f) module %.6f",
						i, g.Center.X, g.Center.Y, g.ModuleSize,
						w.Center.X, w.Center.Y, w.ModuleSize)
				}
			}
			// Without a merge the fold is a plain append and the running average
			// this test exists for never runs.
			if set.perSite > 1 && merged == 0 {
				t.Fatal("no pattern was merged, so the fold's running average was never exercised")
			}
			compareSelection(t, got.Selection, want.Selection)
		})
	}
}

// TestGPUFinderFoldStopsWhereTheConsumerDoes pins what the device does when
// more distinct patterns arrive than the direction is allowed to accumulate.
// The host consumer stops one short of its list's length and abandons the rest
// of the sequence, so the two mechanisms are checked together: the list has to
// end at the stop, and nothing may be reported dropped. A dropped candidate
// would mean the stop was not in force and the fold ran on to truncate the list
// itself, which looks like a successful fold and is not.
func TestGPUFinderFoldStopsWhereTheConsumerDoes(t *testing.T) {
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

	// Far enough apart that nothing merges, so every candidate needs a slot.
	over := maxFinderPatterns + 37
	candidates := make([]gpuFinderFoldCandidate, over)
	for i := range candidates {
		candidates[i] = gpuFinderFoldCandidate{
			Typ:        i % 4,
			Centre:     core.PointF{X: float64(i%64) * 200, Y: float64(i/64) * 200},
			ModuleSize: 4,
		}
	}
	got, err := resident.FoldFinderCandidates(candidates, false, [4]bool{})
	if err != nil {
		t.Fatalf("device fold: %v", err)
	}
	if len(got.Patterns) != maxFinderPatterns-1 {
		t.Errorf("device folded to %d patterns, want the consumer's %d-pattern stop",
			len(got.Patterns), maxFinderPatterns-1)
	}
	if got.Dropped != 0 {
		t.Errorf("device dropped %d candidates, so the stop did not hold the list", got.Dropped)
	}
	if got.Consumed != maxFinderPatterns-1 {
		t.Errorf("device consumed %d of %d candidates, want %d",
			got.Consumed, over, maxFinderPatterns-1)
	}
}

// compareSelection holds the device selection to the host's on every counter
// the scan stats carry and on both pattern sets.
//
// The counters are compared exactly. They are integers on both sides, and the
// prune's whole job is deciding which of them stay nonzero, so a difference in
// any of them is a different direction outcome rather than a rounding
// difference.
func compareSelection(t *testing.T, got, want gpuFinderFoldSelection) {
	t.Helper()
	if got.Missing != want.Missing {
		t.Errorf("missing types %d, want %d", got.Missing, want.Missing)
	}
	if got.Preprune != want.Preprune {
		t.Errorf("pre-prune group sizes %v, want %v", got.Preprune, want.Preprune)
	}
	if got.Preselect != want.Preselect {
		t.Errorf("pre-prune selection %v, want %v", got.Preselect, want.Preselect)
	}
	if got.Selected != want.Selected {
		t.Errorf("post-prune selection %v, want %v", got.Selected, want.Selected)
	}
	for typ := range 4 {
		for _, pair := range []struct {
			what string
			g, w FinderPattern
		}{
			{"selected", got.Patterns[typ], want.Patterns[typ]},
			{"pre-prune", got.Pre[typ], want.Pre[typ]},
		} {
			if pair.g.FoundCount != pair.w.FoundCount || pair.g.Typ != pair.w.Typ {
				t.Errorf("%s type %d: typ=%d found=%d, want typ=%d found=%d",
					pair.what, typ, pair.g.Typ, pair.g.FoundCount, pair.w.Typ, pair.w.FoundCount)
				continue
			}
			if pair.w.FoundCount == 0 {
				continue
			}
			if math.Abs(pair.g.Center.X-pair.w.Center.X) > foldTolerance(pair.w.Center.X, pair.w.FoundCount) ||
				math.Abs(pair.g.Center.Y-pair.w.Center.Y) > foldTolerance(pair.w.Center.Y, pair.w.FoundCount) {
				t.Errorf("%s type %d: centre (%.6f,%.6f), want (%.6f,%.6f)",
					pair.what, typ, pair.g.Center.X, pair.g.Center.Y,
					pair.w.Center.X, pair.w.Center.Y)
			}
		}
	}
}

// outvoteCandidateSet builds one strong site per type and one deliberately weak
// site, so the outvoted-type prune has something to remove. Without a spread in
// found-counts the prune never fires and the branch this test exists for is
// never reached.
func outvoteCandidateSet(weakType int, weakCrossings int) []gpuFinderFoldCandidate {
	narrow := func(v float64) float64 { return float64(float32(v)) }
	var candidates []gpuFinderFoldCandidate
	for typ := range 4 {
		crossings := 40
		if typ == weakType {
			crossings = weakCrossings
		}
		cx := narrow(100 + float64(typ)*400)
		cy := narrow(100 + float64(typ)*250)
		for k := range crossings {
			candidates = append(candidates, gpuFinderFoldCandidate{
				Typ:        typ,
				Direction:  1,
				Centre:     core.PointF{X: narrow(cx + float64(k%3)*0.25), Y: cy},
				ModuleSize: 6,
			})
		}
	}
	return candidates
}

// TestGPUFinderSelectionPrune reaches the outvoted-type prune and its three
// stopping rules, which the clustered sets above never exercise because their
// types are all crossed equally often.
func TestGPUFinderSelectionPrune(t *testing.T) {
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

	arms := []struct {
		name          string
		weakType      int
		weakCrossings int
		printPass     bool
		contextual    [4]bool
		wantPruned    bool
	}{
		{name: "outvoted type pruned", weakType: 2, weakCrossings: 5, wantPruned: true},
		{name: "print pass keeps it", weakType: 2, weakCrossings: 5, printPass: true},
		{name: "strong enough to keep", weakType: 2, weakCrossings: 30},
	}
	for _, arm := range arms {
		t.Run(arm.name, func(t *testing.T) {
			candidates := outvoteCandidateSet(arm.weakType, arm.weakCrossings)
			want := hostFoldCandidates(t, candidates, arm.printPass, arm.contextual)
			got, err := resident.FoldFinderCandidates(candidates, arm.printPass, arm.contextual)
			if err != nil {
				t.Fatalf("device fold: %v", err)
			}
			// The arm has to have reached the branch it names, or agreeing with
			// the host proves only that neither side pruned.
			pruned := want.Selection.Preselect[arm.weakType] > 0 &&
				want.Selection.Selected[arm.weakType] == 0
			if pruned != arm.wantPruned {
				t.Fatalf("host pruned=%v, want %v: the arm did not reach its branch",
					pruned, arm.wantPruned)
			}
			compareSelection(t, got.Selection, want.Selection)
		})
	}
}

// TestGPUFinderSelectionModuleDeviationTie pins a case the two routes are not
// required to agree on, so that it is documented rather than discovered.
//
// bestPattern breaks a found-count tie on which member's module size sits
// closest to the group mean. For a group of two that distance is the same for
// both members in exact arithmetic, so the winner is decided by whichever way
// the mean rounds - and f32 and f64 need not round it the same way. Every
// count still agrees, because which of two equally-crossed candidates wins does
// not change how many there were.
func TestGPUFinderSelectionModuleDeviationTie(t *testing.T) {
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

	narrow := func(v float64) float64 { return float64(float32(v)) }
	var candidates []gpuFinderFoldCandidate
	// Two sites of one type, crossed equally often, with module sizes placed
	// symmetrically about their own mean.
	for site, module := range []float64{4.3, 6.7} {
		for range 12 {
			candidates = append(candidates, gpuFinderFoldCandidate{
				Typ:        0,
				Centre:     core.PointF{X: narrow(200 + float64(site)*500), Y: 200},
				ModuleSize: narrow(module),
			})
		}
	}
	want := hostFoldCandidates(t, candidates, false, [4]bool{})
	got, err := resident.FoldFinderCandidates(candidates, false, [4]bool{})
	if err != nil {
		t.Fatalf("device fold: %v", err)
	}
	if want.Selection.Preprune[0] != 2 {
		t.Fatalf("the tie never formed: type 0 grouped %d members, want 2",
			want.Selection.Preprune[0])
	}
	if got.Selection.Preprune != want.Selection.Preprune ||
		got.Selection.Preselect != want.Selection.Preselect ||
		got.Selection.Selected != want.Selection.Selected ||
		got.Selection.Missing != want.Selection.Missing {
		t.Errorf("counts differ across the tie: %+v, want %+v", got.Selection, want.Selection)
	}
}
