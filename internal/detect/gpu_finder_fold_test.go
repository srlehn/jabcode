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
func hostFoldCandidates(candidates []gpuFinderFoldCandidate) gpuFinderFoldResult {
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
		if total >= maxFinderPatterns {
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
	return gpuFinderFoldResult{Patterns: fps[:total], TypeCount: typeCount}
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
		for range perSite {
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
			want := hostFoldCandidates(candidates)
			got, err := resident.FoldFinderCandidates(candidates)
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
		})
	}
}

// TestGPUFinderFoldReportsOverflow pins what the device does when more distinct
// patterns arrive than the list holds. The host fold has no bound of its own,
// so a silent truncation here would be a fold that looks successful and is not.
func TestGPUFinderFoldReportsOverflow(t *testing.T) {
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
	got, err := resident.FoldFinderCandidates(candidates)
	if err != nil {
		t.Fatalf("device fold: %v", err)
	}
	if len(got.Patterns) != maxFinderPatterns {
		t.Errorf("device folded to %d patterns, want the %d-pattern bound",
			len(got.Patterns), maxFinderPatterns)
	}
	if got.Dropped != over-maxFinderPatterns {
		t.Errorf("device reported %d dropped, want %d", got.Dropped, over-maxFinderPatterns)
	}
}
