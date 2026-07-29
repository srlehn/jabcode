package detect

import (
	"slices"
	"sort"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
)

// quadFromCenters builds a type-ordered finder quad (FP0..FP3) with the given
// centers and per-corner module sizes for the consistency gate tests.
func quadFromCenters(centers [4][2]float64, modules [4]float64) []FinderPattern {
	fps := make([]FinderPattern, 4)
	for i := range fps {
		fps[i] = FinderPattern{
			Typ:        i,
			ModuleSize: modules[i],
			Center:     core.PointF{X: centers[i][0], Y: centers[i][1]},
			FoundCount: 1,
		}
	}
	return fps
}

func TestConsistentFinderQuad(t *testing.T) {
	tests := []struct {
		name    string
		centers [4][2]float64
		modules [4]float64
		want    bool
	}{
		{
			name:    "clean square",
			centers: [4][2]float64{{100, 100}, {900, 100}, {900, 900}, {100, 900}},
			modules: [4]float64{9, 9, 9, 9},
			want:    true,
		},
		{
			// An off-axis capture: opposite edges foreshorten and module sizes
			// carry a perspective gradient, both within the gross-inconsistency
			// thresholds. It must pass so its good decode is never disturbed.
			name:    "off-axis perspective",
			centers: [4][2]float64{{120, 140}, {820, 100}, {900, 860}, {100, 900}},
			modules: [4]float64{8.0, 7.0, 9.5, 8.5},
			want:    true,
		},
		{
			// Field class A: a spurious small-scale finder (module ~3.6) wins one
			// type alongside real corners (module ~7), so the module sizes span a
			// factor of two. Numbers taken from run8-0002's selected quad.
			name:    "class A module-scale mismatch",
			centers: [4][2]float64{{377, 548}, {923, 532}, {863, 1380}, {149, 1401}},
			modules: [4]float64{6.0, 3.6, 7.0, 7.4},
			want:    false,
		},
		{
			// Field class A degenerate: the greedy FP1 sits mid-image so the top
			// edge is a fraction of the bottom edge. Numbers from run7-0016's
			// selected quad (FP1 at 289,437 instead of the real 925,581).
			name:    "class A degenerate edge",
			centers: [4][2]float64{{108, 584}, {289, 437}, {926, 1388}, {118, 1400}},
			modules: [4]float64{9.7, 9.4, 8.9, 9.0},
			want:    false,
		},
		{
			// The corrected assembly the consensus search should reach for the
			// same frame: a near-square quad with the real FP1.
			name:    "class A corrected",
			centers: [4][2]float64{{108, 584}, {925, 581}, {926, 1388}, {118, 1400}},
			modules: [4]float64{9.7, 9.4, 8.9, 9.0},
			want:    true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := ConsistentFinderQuad(quadFromCenters(tc.centers, tc.modules)); got != tc.want {
				t.Errorf("ConsistentFinderQuad = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestConsistentFinderQuadShortInput(t *testing.T) {
	if ConsistentFinderQuad(make([]FinderPattern, 3)) {
		t.Fatal("a quad with fewer than four finders must be reported inconsistent")
	}
}

func TestScoreFinderQuadRejectsOppositeEdgeScaleMismatch(t *testing.T) {
	fps := quadFromCenters(
		[4][2]float64{{100, 100}, {156, 100}, {156, 156}, {100, 156}},
		[4]float64{4, 4, 2.6, 2.6},
	)
	if !ConsistentFinderQuad(fps) {
		t.Fatal("fixture must pass the coarse gate before testing the stricter span gate")
	}
	if _, ok := ScoreFinderQuad(fps[0], fps[1], fps[2], fps[3]); ok {
		t.Fatal("equal pixel edges with incompatible endpoint scales must be rejected")
	}
}

func TestFinderQuadConsensusCounters(t *testing.T) {
	d := &PrimaryDetector{Candidates: quadFromCenters(
		[4][2]float64{{100, 100}, {900, 100}, {900, 900}, {100, 900}},
		[4]float64{9, 9, 9, 9},
	)}
	if _, ok := d.SelectFinderQuadByGeometry(); !ok {
		t.Fatal("expected the clean candidate quad to pass")
	}
	if got := d.Stats.Consensus.GeometryTuples; got != 1 {
		t.Fatalf("geometry tuples = %d, want 1", got)
	}
	if got := d.Stats.Consensus.GeometryScores; got != 1 {
		t.Fatalf("geometry scores = %d, want 1", got)
	}
}

// TestInterpolatedTripleKeepsMasksPacked pins the consensus fallback as a
// bitmap-only consumer. It is the field-rescue path a garbage locate reaches,
// so expanding the masks there would pay for every mask pixel on exactly the
// captures that already cost the most. The interpolation seek averages a
// bounded window of the balanced image and the search only ever needs frame
// dimensions, so a deferred GPU snapshot must survive the whole call packed.
func TestInterpolatedTripleKeepsMasksPacked(t *testing.T) {
	candidates := quadFromCenters(
		[4][2]float64{{100, 100}, {900, 100}, {900, 900}, {100, 900}},
		[4]float64{9, 9, 9, 9},
	)
	d := &PrimaryDetector{
		BM:         core.NewBitmap(1000, 1000, 4),
		Candidates: candidates[:3], // FP3 absent, so the triple must interpolate it
	}
	for i := range d.Ch {
		d.Ch[i] = &core.Bitmap{Width: 1000, Height: 1000, Channels: 1}
	}
	d.materializeChannels = func() error {
		for i := range d.Ch {
			d.Ch[i].Pix = make([]byte, 1000*1000)
		}
		return nil
	}

	if _, ok := d.SelectFinderQuadByInterpolatedTriple(); !ok {
		t.Fatal("expected the consistent triple to interpolate its missing corner")
	}
	if got := d.Stats.Consensus.InterpolatedSeeks; got == 0 {
		t.Fatal("no interpolation seek ran, so the test never reached the mask-reading candidate")
	}
	if got := d.ChannelExpansionCount(); got != 0 {
		t.Fatalf("interpolated-triple consensus expanded channels %d times, want 0", got)
	}
	for channel, ch := range d.Ch {
		if ch.Pix != nil {
			t.Fatalf("interpolated-triple consensus materialized channel %d", channel)
		}
	}
}

func TestFinderCandidateIndexMatchesBounds(t *testing.T) {
	items := make([]FinderPattern, 0, 25)
	for y := 0; y < 5; y++ {
		for x := 0; x < 5; x++ {
			items = append(items, FinderPattern{
				Center:     core.PointF{X: float64(x * 37), Y: float64(y * 29)},
				ModuleSize: 7,
			})
		}
	}
	index := newFinderCandidateIndex(items)
	for minY := -3.0; minY < 130; minY += 11 {
		for minX := -5.0; minX < 160; minX += 13 {
			maxX, maxY := minX+48, minY+41
			want := make([]int, 0)
			for i, item := range items {
				if item.Center.X >= minX && item.Center.X <= maxX &&
					item.Center.Y >= minY && item.Center.Y <= maxY {
					want = append(want, i)
				}
			}
			// Mirror the consensus loop exactly: an X-range slice plus the
			// inline Y rejection. Visitation follows X order rather than
			// candidate order, so the gated property is set equality with a
			// brute-force scan: no admissible candidate skipped, none added.
			got := make([]int, 0, len(want))
			for _, indexed := range index.xRange(minX, maxX) {
				if indexed.pattern.Center.Y < minY || indexed.pattern.Center.Y > maxY {
					continue
				}
				got = append(got, indexed.order)
			}
			sort.Ints(got)
			if !slices.Equal(got, want) {
				t.Fatalf("bounds (%.1f,%.1f)-(%.1f,%.1f): candidates %v, want %v",
					minX, minY, maxX, maxY, got, want)
			}
		}
	}
}
