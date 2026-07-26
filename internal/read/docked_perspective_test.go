package read

import (
	"image"
	"image/color"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/wire"
)

// keystone builds a quad whose bottom edge is narrowed by k of the frame width
// on each side, the simplest projective distortion that gives the two module
// axes different pixels-per-module.
func keystone(w, h int, k float64) [4]core.PointF {
	return [4]core.PointF{
		{X: 80, Y: 80},
		{X: float64(w - 80), Y: 80},
		{X: float64(w) - 80 - k*float64(w), Y: float64(h - 80)},
		{X: 80 + k*float64(w), Y: float64(h - 80)},
	}
}

// warpPerspective renders src under the projective map taking its corners to
// dst, sampling nearest neighbour through the inverse. Nearest neighbour on
// purpose: the point is to exercise the detector's geometry, and an
// interpolating resample would soften module edges and confound a geometry
// failure with a sampling one.
func warpPerspective(src image.Image, dst [4]core.PointF, w, h int) image.Image {
	b := src.Bounds()
	corners := [4]core.PointF{
		{X: float64(b.Min.X), Y: float64(b.Min.Y)},
		{X: float64(b.Max.X - 1), Y: float64(b.Min.Y)},
		{X: float64(b.Max.X - 1), Y: float64(b.Max.Y - 1)},
		{X: float64(b.Min.X), Y: float64(b.Max.Y - 1)},
	}
	inv := core.QuadToQuad(dst, corners)
	out := image.NewNRGBA(image.Rect(0, 0, w, h))
	white := color.NRGBA{0xff, 0xff, 0xff, 0xff}
	for y := range h {
		for x := range w {
			p := inv.Warp(core.Pt(float64(x)+0.5, float64(y)+0.5))
			sx, sy := int(p.X), int(p.Y)
			if sx < b.Min.X || sx >= b.Max.X || sy < b.Min.Y || sy >= b.Max.Y {
				out.Set(x, y, white)
				continue
			}
			out.Set(x, y, src.At(sx, sy))
		}
	}
	return out
}

// A projectively distorted symbol is what separates a directional basis from a
// projective one: under foreshortening the two module axes have different
// pixels-per-module, so the module cell's diagonal is no longer the bisector of
// their directions. A pure rotation cannot show this, because there both axes
// scale alike, so the rotation test alone does not establish that the locator
// handles perspective.
//
// The gate is that a docked secondary is found on the first upright route, with
// no frame turned, at a keystone the whole detector handles. The docked
// boundary used to sit below the single-symbol one; correcting the side-size
// bias closed that gap, and both now hold the upright route to about 0.12. The
// limit past there is the finder module size reading low under foreshortening,
// which shows as one axis measuring a whole version too long, so this gate
// moves again only when that estimate does.
func TestDockedSecondaryDecodesUnderPerspective(t *testing.T) {
	payload := []byte("two docked symbols seen off-axis")
	v4 := image.Pt(4, 4)
	img, err := encode.Run(encode.Config{
		Colors: 8, ModuleSize: 12, SymbolNumber: 2,
		SymbolPositions: []int{0, 2}, SymbolVersions: []image.Point{v4, v4}, SymbolECCLevels: []int{0, 0},
	}, payload)
	if err != nil {
		t.Fatal(err)
	}
	b := img.Bounds()
	w, h := b.Dx()+160, b.Dy()+160
	want := string(isoPayload(payload))

	for _, k := range []float64{0.04, 0.08, 0.10, 0.12} {
		got, report, err := DecodeWithRouteCapabilities(
			warpPerspective(img, keystone(w, h, k), w, h), wire.ISO23634.Mask())
		if err != nil {
			t.Errorf("k=%.2f: %v", k, err)
			continue
		}
		if string(got) != want {
			t.Errorf("k=%.2f: payload = %q, want %q", k, got, want)
		}
		if report.Kind != "upright" || report.Angle != 0 || report.Attempts != 1 {
			t.Errorf("k=%.2f: report = %v; want a single upright route", k, report)
		}
	}
}
