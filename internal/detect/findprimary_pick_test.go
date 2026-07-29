package detect

import (
	"image"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
)

// squareQuad builds four finder centres on a square of the given side in
// pixels, in FP0..FP3 cyclic order, all reporting the same module size.
func squareQuad(side, ms float64) []FinderPattern {
	return []FinderPattern{
		{Center: core.PointF{X: 100, Y: 100}, ModuleSize: ms, Typ: fp0, FoundCount: 5},
		{Center: core.PointF{X: 100 + side, Y: 100}, ModuleSize: ms, Typ: fp1, FoundCount: 5},
		{Center: core.PointF{X: 100 + side, Y: 100 + side}, ModuleSize: ms, Typ: fp2, FoundCount: 5},
		{Center: core.PointF{X: 100, Y: 100 + side}, ModuleSize: ms, Typ: fp3, FoundCount: 5},
	}
}

// The smallest exact quad must pass both gates. The directional retry still
// uses ConsistentFinderQuad because ScoreFinderQuad is the stricter consensus
// fallback gate, not because the two disagree on sound geometry.
func TestExactSmallQuadPassesConsistencyAndScore(t *testing.T) {
	const ms = 4.0
	// Side version 1 is 21 modules; the finder centres span 21-7 = 14 of them.
	fps := squareQuad(14*ms, ms)
	if !ConsistentFinderQuad(fps) {
		t.Fatal("an exact side-21 quad must be consistent")
	}
	if score, ok := ScoreFinderQuad(fps[0], fps[1], fps[2], fps[3]); !ok || score != 0 {
		t.Fatalf("ScoreFinderQuad = (%v, %v), want (0, true)", score, ok)
	}
}

// measuredSliver is the quad direction 30 assembles on
// display_camera/8c_side_rot45_redshift, whose true symbol is 85x85. Two of the
// four are near-duplicates of their neighbours, so the quad spans 1245 and 1331
// pixels one way against 223 and 255 the other. Opposite edges agree to 1.07
// and 1.14, so edge agreement alone does not reject it; the module sizes,
// spanning 6.35 to 12.39, do.
func measuredSliver() []FinderPattern {
	return []FinderPattern{
		{Center: core.PointF{X: 649.5, Y: 1399.6}, ModuleSize: 8.79, Typ: fp0, FoundCount: 5},
		{Center: core.PointF{X: 1150.1, Y: 259.5}, ModuleSize: 6.35, Typ: fp1, FoundCount: 5},
		{Center: core.PointF{X: 1403.2, Y: 226.0}, ModuleSize: 7.64, Typ: fp2, FoundCount: 5},
		{Center: core.PointF{X: 868.1, Y: 1444.7}, ModuleSize: 12.39, Typ: fp3, FoundCount: 5},
	}
}

func TestFamilyPickKeepsFirstConsistentQuad(t *testing.T) {
	const ms = 4.0
	sliver := measuredSliver()
	if ConsistentFinderQuad(sliver) {
		t.Fatal("the measured sliver must not be consistent")
	}
	good := squareQuad(14*ms, ms)

	var p familyPick
	p.offer(finderFamilyResult{status: core.Success, fps: sliver}, nil)
	if !p.have || p.consistent {
		t.Fatal("an inconsistent quad must be retained but not settle the family")
	}
	p.offer(finderFamilyResult{status: core.Success, fps: good}, nil)
	if !p.consistent || p.result.fps[2].Center.Y != good[2].Center.Y {
		t.Fatal("a later consistent quad must replace a retained inconsistent one")
	}
	// A second consistent quad must not displace the first: ranking between
	// sound quads is what the best-score experiment showed costs captures.
	later := squareQuad(14*ms, ms)
	later[0].Center.X = 999
	p.offer(finderFamilyResult{status: core.Success, fps: later}, nil)
	if p.result.fps[0].Center.X == 999 {
		t.Fatal("the first consistent quad must win")
	}
}

func TestFamilyPickFallsBackToTheFirstLocatedQuad(t *testing.T) {
	first, second := measuredSliver(), measuredSliver()
	second[0].Center.X = 700
	if ConsistentFinderQuad(first) || ConsistentFinderQuad(second) {
		t.Fatal("both fixtures must be inconsistent for this to test the fallback")
	}
	var p familyPick
	p.offer(finderFamilyResult{status: core.Success, fps: first}, nil)
	p.offer(finderFamilyResult{status: core.Success, fps: second}, nil)
	if p.result.fps[0].Center.X != first[0].Center.X {
		t.Errorf("fallback kept x=%v, want the first located quad", p.result.fps[0].Center.X)
	}
}

// Candidates are snapshotted per pick because the accumulation keeps growing as
// later directions scan. Publishing a fallback must restore the candidate set
// the winning direction saw, not a union across directions that a first-wins
// rule would never have assembled.
func TestFamilyPickSnapshotsCandidates(t *testing.T) {
	candidates := squareQuad(56, 4)
	var p familyPick
	p.offer(finderFamilyResult{status: core.Success, fps: squareQuad(56, 4)}, candidates)
	candidates = append(candidates, FinderPattern{Center: core.PointF{X: 7, Y: 7}})
	candidates[0].Center.X = 12345
	if len(p.candidates) != 4 || p.candidates[0].Center.X == 12345 {
		t.Error("the pick must hold its own copy of the candidate set")
	}
}

func TestFamilyPickIgnoresFailedScans(t *testing.T) {
	var p familyPick
	p.offer(finderFamilyResult{status: core.Failure, fps: squareQuad(56, 4)}, nil)
	if p.have {
		t.Error("a failed scan must not be retained")
	}
}

// The current family alone ends the sweep when it is wanted. The historical
// signature is an additive fallback, so its own quad must neither stop the
// search for the requested format nor be discarded by it.
func TestOnlyTheCurrentFamilyEndsTheSweep(t *testing.T) {
	const ms = 4.0
	good := squareQuad(14*ms, ms)
	bad := measuredSliver()
	d := &PrimaryDetector{}

	var picks [finderFamilyCount]familyPick
	picks[FinderFamilyCurrent].offer(finderFamilyResult{status: core.Success, fps: good}, nil)
	picks[FinderFamilyBSI].offer(finderFamilyResult{status: core.Success, fps: bad}, nil)

	if !d.settled(picks, true, false) {
		t.Error("a consistent current-family quad must settle a current-only scan")
	}
	if d.settled(picks, false, true) {
		t.Error("an inconsistent BSI quad must not settle a BSI scan")
	}
	// A tagged build must not sweep the remaining directions for the historical
	// signature once the current one is already sound.
	if !d.settled(picks, true, true) {
		t.Error("a sound current-family quad must settle a joint scan")
	}
	if d.settled([finderFamilyCount]familyPick{}, true, true) {
		t.Error("no consistent quad anywhere must not settle")
	}

	// The reverse, and the reason "any consistent family" was wrong: a sound
	// historical quad found early must not stop the sweep while the requested
	// format is still unresolved, or a later direction's real current-family
	// quad is never reached.
	var spurious [finderFamilyCount]familyPick
	spurious[FinderFamilyBSI].offer(finderFamilyResult{status: core.Success, fps: good}, nil)
	spurious[FinderFamilyCurrent].offer(finderFamilyResult{status: core.Success, fps: bad}, nil)
	if d.settled(spurious, true, true) {
		t.Error("a consistent BSI quad must not settle a scan that still wants the current family")
	}
	if !d.settled(spurious, false, true) {
		t.Error("a consistent BSI quad must settle a scan that does not want the current family")
	}

	// Publishing keeps both families' results, so the sound current quad
	// survives alongside the spurious BSI one.
	found := d.publishPicks(&picks, true, true)
	if !found.Has(FinderFamilyCurrent) || !found.Has(FinderFamilyBSI) {
		t.Errorf("publishPicks dropped a family: %v", found)
	}
	if d.familyResults[FinderFamilyCurrent].fps[2].Center.Y != good[2].Center.Y {
		t.Error("the current family's sound quad was replaced")
	}
}

func TestSquareQuadHelperIsSane(t *testing.T) {
	fps := squareQuad(56, 4)
	if got := (image.Point{X: int(fps[1].Center.X - fps[0].Center.X), Y: int(fps[3].Center.Y - fps[0].Center.Y)}); got != (image.Point{X: 56, Y: 56}) {
		t.Fatalf("helper built %v", got)
	}
}
