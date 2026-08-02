package read

import (
	"image"
	"image/color"
	"image/draw"
	"slices"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/detect"
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

// A region attempt locates on its own crop. What pins the field is the crop that
// locates nothing: that must report -1, where a dropped field reports zero. The
// rotation of the fixture says nothing here - a square finder can still be
// settled by the row walk at a turned angle - so a located crop is only required
// to name a probed direction, never a particular one.
//
// Only the two unambiguous stages are asserted on. A quad is recorded on the
// finding after side-size validation, so readNoSideSize is a stage that located
// nothing and correctly reports -1; treating every stage but readNoFinders as
// located would demand a direction from it.
func TestRegionRouteCarriesAScanDirection(t *testing.T) {
	run := func(t *testing.T, symbol bool) []routeAttempt {
		t.Helper()
		tr := &routeTrace{level: -1}
		var f finding
		_, _ = decodeRetriesRegionsLevel(
			levelImageOf(regionFixture(t, symbol)), func() bool { return false },
			&f, true, tr, wire.ISO23634.Mask(),
		)
		var regions []routeAttempt
		for _, a := range tr.attempts {
			if a.kind == "roi" {
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

	t.Run("a located region names a probed direction", func(t *testing.T) {
		decoded := 0
		for i, a := range run(t, true) {
			if a.stage != readDecoded {
				continue
			}
			decoded++
			if !slices.Contains(detect.ProbeDegrees(), a.deg) {
				t.Errorf("region attempt %d decoded but reported deg=%v, not a probed direction", i, a.deg)
			}
		}
		if decoded == 0 {
			t.Fatal("no region decoded, so nothing pins the located case")
		}
	})

	t.Run("a region that locates nothing reports -1", func(t *testing.T) {
		empty := 0
		for i, a := range run(t, false) {
			if a.stage != readNoFinders {
				continue
			}
			empty++
			if a.deg != -1 {
				t.Errorf("region attempt %d located nothing but reported deg=%v;"+
					" an unset field reads as 0, the row walk", i, a.deg)
			}
		}
		if empty == 0 {
			t.Fatal("every proposed region found something, so nothing pins the -1 case")
		}
	})
}

// regionFixture is a frame cluttered enough to propose regions, optionally with
// a turned symbol dropped into one of them.
func regionFixture(t *testing.T, symbol bool) image.Image {
	t.Helper()
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
	if !symbol {
		return frame
	}
	r, err := encode.Render(encode.Config{Colors: 8, ModuleSize: 12, SymbolNumber: 1},
		[]byte("region rung direction"))
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	rotated := testutil.RotateImage(r.Image, 30)
	draw.Draw(frame, rotated.Bounds().Add(image.Pt(1950, 2600)), rotated, image.Point{}, draw.Src)
	return frame
}

// The detailed trace builds its own route record, so the direction has to be
// stamped onto it too. Only -r carried it at first, which left every `--diag`
// attempt line saying `kind=frame` with no way to tell a row-settled read from
// a turned one - the exact ambiguity the rename removed from the short report.
func TestDiagnosticRouteCarriesTheScanDirection(t *testing.T) {
	msg := []byte("diagnostic route direction")
	img, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 12, SymbolNumber: 1}, msg)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	_, trace, err := DecodeWithTraceCapabilities(testutil.RotateImage(img, 45), wire.ISO23634.Mask())
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(trace.Attempts) == 0 {
		t.Fatal("the detailed trace recorded no attempts")
	}
	turned := 0
	for i, a := range trace.Attempts {
		if a.Stage != readDecoded.String() {
			continue
		}
		if !slices.Contains(detect.ProbeDegrees(), a.Route.Deg) {
			t.Errorf("attempt %d decoded but its trace reports deg=%v, not a probed direction",
				i, a.Route.Deg)
		}
		if a.Route.Deg != 0 {
			turned++
		}
	}
	// A 45-degree render is settled by a turned scan, so a trace reporting the
	// row walk everywhere is reporting a field nobody stamped.
	if turned == 0 {
		t.Errorf("every decoded attempt traced the row walk on a 45-degree render: %v", trace.Attempts)
	}
}
