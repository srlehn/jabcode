package core

import (
	"image"
	"math"
	"testing"
)

// TestPerspectiveTransform verifies the fundamental property of
// PerspectiveTransform: warping the symbol's grid corners must reproduce the
// four detected pattern centers.
func TestPerspectiveTransform(t *testing.T) {
	side := image.Pt(21, 21)
	centers := [4]PointF{{10, 12}, {205, 18}, {210, 220}, {15, 215}} // a skewed quad
	pt := PerspectiveTransform(centers[0], centers[1], centers[2], centers[3], side)

	sx, sy := float64(side.X), float64(side.Y)
	corners := [4]PointF{{3.5, 3.5}, {sx - 3.5, 3.5}, {sx - 3.5, sy - 3.5}, {3.5, sy - 3.5}}
	for i, c := range corners {
		got := pt.Warp(c)
		if math.Abs(got.X-centers[i].X) > 1e-6 || math.Abs(got.Y-centers[i].Y) > 1e-6 {
			t.Errorf("corner %d: got (%g, %g), want (%g, %g)", i, got.X, got.Y, centers[i].X, centers[i].Y)
		}
	}
}

// TestWarpRowMatchesWarp pins the y-hoisted WarpRow to the per-point Warp bit
// for bit. The footprint sampler relies on that identity to reproduce every
// sampled pixel, so any rounding divergence here would move harness rows
// silently. The x values include the fractional footprint offsets the sampler
// actually feeds, not just integers.
func TestWarpRowMatchesWarp(t *testing.T) {
	pt := PerspectiveTransform(PointF{10, 12}, PointF{205, 18}, PointF{210, 220}, PointF{15, 215}, image.Pt(37, 29))
	xs := []float64{-3.25, 0, 0.5, 7.7, 18.5, 36.5, 100.125, 1023.5}
	ys := []float64{-2.5, 0, 0.5, 11.3, 28.5, 512.75}
	out := make([]PointF, len(xs))
	for _, y := range ys {
		pt.WarpRow(xs, y, out)
		for i, x := range xs {
			want := pt.Warp(Pt(x, y))
			if out[i] != want {
				t.Fatalf("WarpRow(%g,%g)=%v, Warp=%v", x, y, out[i], want)
			}
		}
	}
}
