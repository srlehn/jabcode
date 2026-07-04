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
