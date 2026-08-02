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
// locates nothing: that must report -1, and a dropped field reports zero. The
// rotation of the fixture says nothing here - a square finder can still be
// settled by the row walk at a turned angle - so a located crop is only required
// to name a probed direction, never a particular one.
func TestRegionRouteCarriesAScanDirection(t *testing.T) {
	for _, tc := range []struct {
		name   string
		symbol bool
	}{
		{"a located region names a probed direction", true},
		{"a region that locates nothing reports -1", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tr := &routeTrace{level: -1}
			var f finding
			_, _ = decodeRetriesRegionsLevel(
				levelImageOf(regionFixture(t, tc.symbol)), func() bool { return false },
				&f, true, tr, wire.ISO23634.Mask(),
			)
			regions := 0
			for i, a := range tr.attempts {
				if a.kind != "roi" {
					continue
				}
				regions++
				switch {
				case a.stage == readNoFinders:
					if a.deg != -1 {
						t.Errorf("region attempt %d located nothing but reported deg=%v;"+
							" an unset field reads as 0, the row walk", i, a.deg)
					}
				case !slices.Contains(detect.ProbeDegrees(), a.deg):
					t.Errorf("region attempt %d reported deg=%v, not a probed direction", i, a.deg)
				}
			}
			// Skipping here would leave the region construction site unpinned
			// the moment proposal behaviour changes. Failing makes someone
			// re-establish the cover.
			if regions == 0 {
				t.Fatal("the fixture proposed no regions, so the region route site is untested")
			}
		})
	}
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
