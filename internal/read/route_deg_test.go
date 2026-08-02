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

// A region attempt locates on its own crop, so it reports either that crop's
// direction or -1 for a crop that found nothing. The fixture is turned 30
// degrees, so zero is neither.
func TestRegionRouteCarriesAScanDirection(t *testing.T) {
	payload := []byte("region rung direction")
	r, err := encode.Render(encode.Config{Colors: 8, ModuleSize: 12, SymbolNumber: 1}, payload)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	rotated := testutil.RotateImage(r.Image, 30)

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
	draw.Draw(frame, rotated.Bounds().Add(image.Pt(1950, 2600)), rotated, image.Point{}, draw.Src)

	tr := &routeTrace{level: -1}
	var f finding
	_, _ = decodeRetriesRegionsLevel(
		levelImageOf(frame), func() bool { return false }, &f, true, tr, wire.ISO23634.Mask(),
	)

	regions := 0
	for i, a := range tr.attempts {
		if a.kind != "roi" {
			continue
		}
		regions++
		if a.deg == 0 {
			t.Errorf("region attempt %d reported the row walk on a 30-degree fixture,"+
				" which is what an unset field looks like", i)
		}
	}
	if regions == 0 {
		t.Skip("this fixture proposed no regions; the seeded case still covers the seam")
	}
}
