package read

import (
	"image"
	"image/color"
	"image/draw"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/testutil"
	"github.com/srlehn/jabcode/internal/wire"
)

// The seeded and region rungs build their own routeAttempt, and both were
// shipped once without the scan direction on it. Neither is reachable from a
// committed whole-image fixture on demand - which rung wins depends on the
// capture - so each entry point is driven directly here. Every case turns on
// the same thing: a dropped field reads as zero, and zero is the row walk.

// A seeded attempt inherits the coarse level's quad, so it must report the
// direction that quad was found at, not the direction of a scan it never ran.
func TestSeededRouteCarriesTheSeedDirection(t *testing.T) {
	const w, h = 1400, 1400
	frame := image.NewNRGBA(image.Rect(0, 0, w, h))
	draw.Draw(frame, frame.Bounds(), &image.Uniform{color.NRGBA{240, 240, 240, 255}}, image.Point{}, draw.Src)
	p := newPyramid(frame)
	if p == nil || p.count() < 2 {
		t.Fatalf("fixture built %v pyramid levels, want at least 2", p)
	}

	seed := finding{located: true, deg: 45}
	corners := [4]core.PointF{{X: 300, Y: 300}, {X: 1100, Y: 300}, {X: 1100, Y: 1100}, {X: 300, Y: 1100}}
	for i := range 4 {
		seed.quad[i] = corners[i]
		seed.sizes[i] = 4
	}
	tr := &routeTrace{level: -1}
	_, _, _ = decodeSeededTracedCapabilities(p, seed, func() bool { return false }, tr, wire.ISO23634.Mask())

	if len(tr.attempts) == 0 {
		t.Fatal("the seeded route recorded no attempt")
	}
	for i, a := range tr.attempts {
		if a.kind != "seeded" {
			t.Fatalf("attempt %d is %q, want seeded", i, a.kind)
		}
		if a.deg != seed.deg {
			t.Errorf("attempt %d reported deg=%v, want the seed's %v", i, a.deg, seed.deg)
		}
	}
}

// A region attempt locates on its own crop, so what it must carry is the
// direction that produced its own quad. Membership in the probe sweep is not
// that: a constant 45 substituted for the propagation belongs to the sweep and
// would pass. Each located attempt is compared against the scan its own
// detector published instead, which the detailed trace records alongside the
// route field. The crop that locates nothing pins the other end: it must report
// -1, where a dropped field reports zero.
//
// Only the two unambiguous stages are asserted on. A quad is recorded on the
// finding after side-size validation, so readNoSideSize is a stage that located
// nothing and correctly reports -1; treating every stage but readNoFinders as
// located would demand a direction from it.
func TestRegionRouteCarriesAScanDirection(t *testing.T) {
	run := func(t *testing.T, frame image.Image) []DiagnosticAttempt {
		t.Helper()
		tr := &routeTrace{level: -1, detailed: true}
		var f finding
		_, _ = decodeRetriesRegionsLevel(
			levelImageOf(frame), func() bool { return false },
			&f, true, tr, wire.ISO23634.Mask(),
		)
		var regions []DiagnosticAttempt
		for _, a := range tr.details {
			if a.Route.Kind == "roi" {
				regions = append(regions, a)
			}
		}
		// Skipping here would leave the region construction site unpinned the
		// moment proposal behaviour changes. Failing makes someone re-establish
		// the cover.
		if len(regions) == 0 {
			t.Fatal("the fixture proposed no regions, so the region route site is untested")
		}
		return regions
	}

	t.Run("a located region carries the direction that produced its quad", func(t *testing.T) {
		var published []float64
		for _, deg := range []float64{0, 30} {
			decoded := 0
			for i, a := range run(t, regionFixture(t, deg)) {
				if a.Stage != readDecoded.String() {
					continue
				}
				decoded++
				published = append(published, assertRouteDegIsThePublishedScan(t, i, a))
			}
			if decoded == 0 {
				t.Fatalf("no region decoded at %v degrees, so nothing pins the located case", deg)
			}
		}
		requireDistinctDirections(t, published)
	})

	t.Run("a region that locates nothing reports -1", func(t *testing.T) {
		empty := 0
		for i, a := range run(t, regionClutter()) {
			if a.Stage != readNoFinders.String() {
				continue
			}
			empty++
			if a.Route.Deg != -1 {
				t.Errorf("region attempt %d located nothing but reported deg=%v;"+
					" an unset field reads as 0, the row walk", i, a.Route.Deg)
			}
		}
		if empty == 0 {
			t.Fatal("every proposed region found something, so nothing pins the -1 case")
		}
	})
}

// assertRouteDegIsThePublishedScan compares one attempt's route direction with
// the scan its own detector stats mark as published, and returns that scan. The
// stats and the family are recorded on the same attempt, so this asks whether
// the route carried the direction that produced this quad - which an inherited
// value or a dropped field both fail - rather than whether it named a plausible
// one.
func assertRouteDegIsThePublishedScan(t *testing.T, i int, a DiagnosticAttempt) float64 {
	t.Helper()
	want := a.Detector.PublishedScanDegrees(a.FindersFamily)
	if want == -1 {
		t.Errorf("attempt %d decoded but its own stats publish no scan, so its deg=%v is unchecked",
			i, a.Route.Deg)
		return want
	}
	if a.Route.Deg != want {
		t.Errorf("attempt %d reported deg=%v, but the scan that produced its quad was %v",
			i, a.Route.Deg, want)
	}
	return want
}

// requireDistinctDirections fails when every checked attempt published the same
// direction. The equality above is worth only as much as the spread of the cases
// it runs over: with a single direction in play, a constant substituted for the
// propagation satisfies it. Which render settles on which direction is not a
// function of its rotation, so this asserts nothing about any angle - it is a
// statement about the cover, and if it fires the fixtures need re-choosing
// rather than the comparison weakening.
func requireDistinctDirections(t *testing.T, published []float64) {
	t.Helper()
	if len(published) < 2 {
		t.Errorf("only %d attempt checked, so a constant would satisfy the comparison", len(published))
		return
	}
	for _, deg := range published {
		if deg != published[0] {
			return
		}
	}
	t.Errorf("every checked attempt published %v, so a constant would satisfy the comparison", published)
}

// regionClutter is a frame busy enough to propose regions and holding no symbol,
// so every region it proposes locates nothing.
func regionClutter() *image.NRGBA {
	const w, h = 3000, 4000
	frame := image.NewNRGBA(image.Rect(0, 0, w, h))
	fill := func(rect image.Rectangle, c color.NRGBA) {
		draw.Draw(frame, rect, &image.Uniform{c}, image.Point{}, draw.Src)
	}
	fill(frame.Bounds(), color.NRGBA{18, 24, 42, 255})
	fill(image.Rect(120, 900, 1500, 2900), color.NRGBA{235, 232, 226, 255})
	for y := 1000; y < 2800; y += 90 {
		fill(image.Rect(200, y, 1400, y+22), color.NRGBA{90, 90, 96, 255})
	}
	return frame
}

// regionFixture drops a symbol turned by deg into one of that frame's regions.
func regionFixture(t *testing.T, deg float64) image.Image {
	t.Helper()
	frame := regionClutter()
	r, err := encode.Render(encode.Config{Colors: 8, ModuleSize: 12, SymbolNumber: 1},
		[]byte("region rung direction"))
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	rotated := testutil.RotateImage(r.Image, deg)
	draw.Draw(frame, rotated.Bounds().Add(image.Pt(1950, 2600)), rotated, image.Point{}, draw.Src)
	return frame
}

// The detailed trace builds its own route record, so the direction has to be
// stamped onto it too. Only -r carried it at first, which left every `--diag`
// attempt line saying `kind=frame` with no way to tell a row-settled read from
// a turned one - the exact ambiguity the rename removed from the short report.
//
// Nothing here asserts that a turned render settles on a turned scan: a finder
// is square, so the row walk can keep a rotated read, and demanding a nonzero
// direction would fail a legitimate result. Each attempt is compared against the
// scan its own detector published, over two renders so no constant satisfies
// both.
func TestDiagnosticRouteCarriesTheScanDirection(t *testing.T) {
	msg := []byte("diagnostic route direction")
	img, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 12, SymbolNumber: 1}, msg)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	var published []float64
	for _, deg := range []float64{0, 45} {
		_, trace, err := DecodeWithTraceCapabilities(testutil.RotateImage(img, deg), wire.ISO23634.Mask())
		if err != nil {
			t.Fatalf("decode at %v degrees: %v", deg, err)
		}
		if len(trace.Attempts) == 0 {
			t.Fatalf("the detailed trace recorded no attempts at %v degrees", deg)
		}
		decoded := 0
		for i, a := range trace.Attempts {
			if a.Stage != readDecoded.String() {
				continue
			}
			decoded++
			published = append(published, assertRouteDegIsThePublishedScan(t, i, a))
		}
		if decoded == 0 {
			t.Fatalf("no attempt decoded at %v degrees, so nothing pins the traced direction", deg)
		}
	}
	requireDistinctDirections(t, published)
}
