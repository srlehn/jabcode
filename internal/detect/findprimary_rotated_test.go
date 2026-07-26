package detect

import (
	"image"
	"math"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/encode"
)

// Whichever scan direction wins, the published quad must be one a symbol could
// actually have, and must size to the encoder's own side size.
//
// This deliberately does NOT pin that the row walk feeds the picker rather than
// short-circuiting on its own quad. That was checked by mutation: restoring the
// old "return on any row-walk locate" leaves this test passing, because on
// these synthetic rotated symbols the row walk either finds nothing or finds a
// sound quad, so the short circuit never fires. Pinning direction 0 needs an
// input whose row walk produces a bad quad, which no synthetic fixture here
// reproduces; the real captures that motivated it are not committed.
func TestRotatedSymbolsPublishAConsistentQuad(t *testing.T) {
	r, err := encode.Render(encode.Config{
		Colors: 8, ModuleSize: 1, SymbolNumber: 1,
		SymbolVersions: []image.Point{{X: 4, Y: 4}},
	}, []byte("row walk must not short circuit the direction sweep"))
	if err != nil {
		t.Fatal(err)
	}
	for _, deg := range []float64{0, 20, 35, 50, 65, 80} {
		rgba, _ := renderRotatedRGBA(r, 8, deg*math.Pi/180)
		bm := core.BitmapFromImage(rgba)
		BalanceRGB(bm)
		ch := BinarizerRGB(bm, nil)
		d := &PrimaryDetector{BM: bm, Ch: ch, Mode: IntensiveDetect}
		if !d.LocateFinders() {
			t.Errorf("deg=%.0f: no finders", deg)
			continue
		}
		// Whichever direction won, the published quad must be one a symbol
		// could actually have. Returning a row-walk sliver would fail here.
		if !ConsistentFinderQuad(d.FPs) {
			t.Errorf("deg=%.0f: published an inconsistent quad: %v", deg, quadEdges(d.FPs))
		}
		if got := CalculateSideSize(bm, d.FPs); got != r.SideSize {
			t.Errorf("deg=%.0f: side = %v, want %v", deg, got, r.SideSize)
		}
	}
}

func quadEdges(fps []FinderPattern) [4]float64 {
	e := func(a, b int) float64 {
		return math.Hypot(fps[a].Center.X-fps[b].Center.X, fps[a].Center.Y-fps[b].Center.Y)
	}
	return [4]float64{e(0, 1), e(1, 2), e(2, 3), e(3, 0)}
}
