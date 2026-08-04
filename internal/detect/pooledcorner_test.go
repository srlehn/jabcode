package detect

import (
	"math"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
)

// partialQuad builds a frontal 85-module symbol at module size 8, then removes
// one corner so the interpolation has something to reconstruct. Interpolating
// from the remaining three reproduces the removed corner exactly, which is the
// premise the pooled search is bounded against.
func partialQuad(miss int) []FinderPattern {
	const ms, side = 8.0, 85.0
	span := ms * (side - 7) // finder centres sit 3.5 modules inside each edge
	corners := [4]core.PointF{
		core.Pt(200, 200),
		core.Pt(200+span, 200),
		core.Pt(200+span, 200+span),
		core.Pt(200, 200+span),
	}
	fps := make([]FinderPattern, maxFinderPatterns)
	for i := range 4 {
		fps[i] = FinderPattern{Typ: i, Center: corners[i], ModuleSize: ms, FoundCount: 6}
	}
	fps[miss] = FinderPattern{Typ: miss}
	return fps
}

// TestPooledCornerPrefersTheDetectedCorner is the case the pool exists for: the
// scan direction that published lost one type, another direction found it, and
// the candidate sits where the partial quad predicts. A real detection there is
// better evidence than either the construction or a fresh box search.
func TestPooledCornerPrefersTheDetectedCorner(t *testing.T) {
	fps := partialQuad(fp1)
	want := core.Pt(200+8*78+5, 200-4)
	pool := []FinderPattern{
		{Typ: fp1, Center: want, ModuleSize: 8.3, FoundCount: 4},
	}
	miss, ok := interpolateMissingPattern(fps)
	if !ok || miss != fp1 {
		t.Fatalf("interpolateMissingPattern = %d, %v", miss, ok)
	}
	got, ok := pickPooledCorner(pool, fps, miss)
	if !ok {
		t.Fatalf("no pooled corner within %.1f px of the prediction", pooledCornerRadius(fps, miss))
	}
	if got.Center != want {
		t.Errorf("centre %v, want the pooled candidate at %v", got.Center, want)
	}
}

// TestPooledCornerNeedsBothBounds is the constraint that keeps this from
// becoming a confident mistake. A cluttered background yields candidates that
// passed the whole cross-check chain, so neither proximity nor a plausible
// module size is sufficient on its own.
func TestPooledCornerNeedsBothBounds(t *testing.T) {
	fps := partialQuad(fp1)
	miss, _ := interpolateMissingPattern(fps)
	pred := fps[miss].Center
	radius := pooledCornerRadius(fps, miss)

	for _, tc := range []struct {
		name string
		cand FinderPattern
	}{{
		// Right where the corner is predicted, at a scale no part of this
		// symbol has: a blob of the background, not a corner of the code.
		name: "near the prediction, wrong scale",
		cand: FinderPattern{Typ: fp1, Center: pred, ModuleSize: 2.5, FoundCount: 9},
	}, {
		// The right scale somewhere else entirely. Snapping to it would move
		// the corner further from truth than the construction already was.
		name: "right scale, outside the neighbourhood",
		cand: FinderPattern{Typ: fp1, Center: core.Pt(pred.X+3*radius, pred.Y), ModuleSize: 8, FoundCount: 9},
	}, {
		// Another type's candidate is not a completion of this quad whatever
		// its position.
		name: "wrong type at the prediction",
		cand: FinderPattern{Typ: fp2, Center: pred, ModuleSize: 8, FoundCount: 9},
	}, {
		// Right type, right place, right scale, but confirmed by too few
		// crossings to have entered a type group. Measured on a clean synthetic
		// docking, where such a candidate sat about a module from the true
		// corner and taking it over the construction cost the read.
		name: "below the selection's own support floor",
		cand: FinderPattern{Typ: fp1, Center: pred, ModuleSize: 8, FoundCount: minFinderCrossings - 1},
	}} {
		t.Run(tc.name, func(t *testing.T) {
			if got, ok := pickPooledCorner([]FinderPattern{tc.cand}, fps, miss); ok {
				t.Errorf("accepted %v at module size %.1f", got.Center, got.ModuleSize)
			}
		})
	}
}

// TestPooledCornerRadiusFollowsForeshortening pins the bound to its derivation
// rather than to a tolerance someone tuned. With no module-size gradient the
// construction is exact and only centre noise is allowed; a gradient is the
// measure of keystone, and the parallelogram completion misses by an amount
// proportional to it.
func TestPooledCornerRadiusFollowsForeshortening(t *testing.T) {
	frontal := partialQuad(fp1)
	if _, ok := interpolateMissingPattern(frontal); !ok {
		t.Fatal("interpolateMissingPattern failed")
	}
	flat := pooledCornerRadius(frontal, fp1)
	if want := pooledCornerNoiseModules * frontal[fp1].ModuleSize; math.Abs(flat-want) > 1e-9 {
		t.Errorf("frontal radius %.3f, want centre noise alone (%.3f)", flat, want)
	}

	skewed := partialQuad(fp1)
	skewed[fp2].ModuleSize = 12 // the near edge images larger than the far one
	if _, ok := interpolateMissingPattern(skewed); !ok {
		t.Fatal("interpolateMissingPattern failed")
	}
	if wide := pooledCornerRadius(skewed, fp1); wide <= flat {
		t.Errorf("keystone radius %.3f did not exceed the frontal %.3f", wide, flat)
	}
}

// TestContextualFinderQuadsRecoverARepeatedWeakCorner covers the directional
// split from the side-view capture: one direction qualifies the rejected corner,
// then a later direction supplies the strong triple it completes.
func TestContextualFinderQuadsRecoverARepeatedWeakCorner(t *testing.T) {
	fps := make([]FinderPattern, 4)
	fps[fp0] = FinderPattern{Typ: fp0, Center: core.Pt(100, 600), ModuleSize: 10, FoundCount: 6}
	fps[fp2] = FinderPattern{Typ: fp2, Center: core.Pt(1000, 550), ModuleSize: 9, FoundCount: 8}
	fps[fp3] = FinderPattern{Typ: fp3, Center: core.Pt(700, 1050), ModuleSize: 10, FoundCount: 9}
	miss, ok := interpolateMissingPattern(fps)
	if !ok || miss != fp1 {
		t.Fatalf("interpolateMissingPattern = %d, %v", miss, ok)
	}

	seeds := []FinderPattern{
		{Typ: fp1, Center: core.Pt(499, 199), ModuleSize: 10, FoundCount: 1},
		{Typ: fp1, Center: core.Pt(500, 200), ModuleSize: 10, FoundCount: 1},
		{Typ: fp1, Center: core.Pt(501, 201), ModuleSize: 10, FoundCount: 1},
		{Typ: fp1, Center: core.Pt(400, 100), ModuleSize: 3, FoundCount: 9},
	}
	d := PrimaryDetector{}
	d.accumulateContextualFinderCandidates(contextualFinderCandidates(seeds))
	hypotheses := contextualFinderQuads(fps, miss, d.contextualCandidates)
	if len(hypotheses) == 0 {
		t.Fatal("no contextual quad survived")
	}
	got := hypotheses[0].Patterns[fp1]
	if dist(got.Center, core.Pt(500, 200)) > 2 {
		t.Errorf("contextual centre = %v, want the repeated weak corner", got.Center)
	}
	if got.FoundCount != minFinderCrossings {
		t.Errorf("contextual support = %d, want %d", got.FoundCount, minFinderCrossings)
	}
}

func TestContextualFinderQuadsKeepTheSupportFloor(t *testing.T) {
	fps := make([]FinderPattern, 4)
	fps[fp0] = FinderPattern{Typ: fp0, Center: core.Pt(100, 600), ModuleSize: 10, FoundCount: 6}
	fps[fp2] = FinderPattern{Typ: fp2, Center: core.Pt(1000, 550), ModuleSize: 9, FoundCount: 8}
	fps[fp3] = FinderPattern{Typ: fp3, Center: core.Pt(700, 1050), ModuleSize: 10, FoundCount: 9}
	miss, _ := interpolateMissingPattern(fps)
	seeds := []FinderPattern{
		{Typ: fp1, Center: core.Pt(499, 199), ModuleSize: 10, FoundCount: 1},
		{Typ: fp1, Center: core.Pt(501, 201), ModuleSize: 10, FoundCount: 1},
	}
	candidates := contextualFinderCandidates(seeds)
	if got := contextualFinderQuads(fps, miss, candidates); len(got) != 0 {
		t.Fatalf("two weak crossings produced %d hypotheses", len(got))
	}
}

func TestEstimateMissingPatternFallsBackToConstruction(t *testing.T) {
	fps := partialQuad(fp1)
	ch := [3]*core.Bitmap{core.NewBitmap(1200, 1200, 1), nil, nil}
	bm := core.NewBitmap(1200, 1200, 3)

	src, miss, ok := estimateMissingPattern(func() *core.Bitmap { return bm }, ch, fps, nil)
	if !ok {
		t.Fatal("estimateMissingPattern rejected an in-frame estimate")
	}
	if src != CornerConstructed || miss != fp1 {
		t.Errorf("corner result = (%s, %d), want (constructed, %d)", src, miss, fp1)
	}
	if want := core.Pt(200+8*78, 200); fps[fp1].Center != want {
		t.Errorf("centre %v, want the exact affine completion %v", fps[fp1].Center, want)
	}
}
