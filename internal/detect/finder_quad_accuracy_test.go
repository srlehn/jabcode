package detect

import (
	"image"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/encode"
)

// On an undegraded render the four finder centres are known exactly: a symbol
// of side s at module size m puts them at 3.5*m and (s-3.5)*m on each axis. Any
// residual there is a bias of the cross-check chain itself rather than a
// response to noise, so this pins the whole chain to exact geometry.
//
// It exists because a residual appeared that was invisible in aggregate decode
// counts: the diagonal walk overwrote the centre the axis walks had measured,
// and an m-wide core has exactly one diagonal of each family running its full
// width. Seeded onto a neighbouring one, the walk measures a chord a pixel
// short whose midpoint is half a pixel off - on both axes at once, since a
// diagonal step moves both. The two families sit differently about the core, so
// the error fell on fp2 and fp3 only and tilted the quad instead of shifting
// it, which no centre-to-centre distance and no side-size estimate can see.
func TestFinderQuadExactOnCleanRender(t *testing.T) {
	for _, tc := range []struct {
		version image.Point
		module  int
	}{
		{image.Pt(2, 2), 12},
		{image.Pt(5, 5), 8},
		{image.Pt(8, 8), 6},
		{image.Pt(8, 4), 6},
		{image.Pt(4, 8), 10},
	} {
		r, err := encode.Render(encode.Config{
			Colors: 8, ModuleSize: tc.module, SymbolNumber: 1,
			SymbolVersions: []image.Point{tc.version},
		}, []byte("finder quad accuracy"))
		if err != nil {
			t.Fatal(err)
		}
		bm := core.BitmapFromImage(r.Image)
		BalanceRGB(bm)
		det := &PrimaryDetector{BM: bm, Ch: BinarizerRGB(bm, nil), Mode: IntensiveDetect}
		if !det.LocateFinders() {
			t.Errorf("version=%v module=%d: no finders", tc.version, tc.module)
			continue
		}

		m := float64(tc.module)
		lox, hix := 3.5*m, (float64(r.SideSize.X)-3.5)*m
		loy, hiy := 3.5*m, (float64(r.SideSize.Y)-3.5)*m
		want := [4]core.PointF{{X: lox, Y: loy}, {X: hix, Y: loy}, {X: hix, Y: hiy}, {X: lox, Y: hiy}}
		for i, fp := range det.FPs[:4] {
			if fp.Center != want[i] {
				t.Errorf("version=%v module=%d fp%d: centre = %v, want %v",
					tc.version, tc.module, i, fp.Center, want[i])
			}
			if fp.ModuleSize != m {
				t.Errorf("version=%v module=%d fp%d: module size = %v, want %v",
					tc.version, tc.module, i, fp.ModuleSize, m)
			}
		}
	}
}
