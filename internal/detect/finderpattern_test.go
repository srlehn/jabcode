package detect

import (
	"testing"

	"github.com/srlehn/jabcode/internal/core"
)

// TestCrossCheckColorStaysInBounds pins the boundary behaviour of the colour
// walks: a candidate near the right edge or outside the image must be refused
// instead of indexing into the next row - or, on the last row, past the pixel
// buffer, which panics.
func TestCrossCheckColorStaysInBounds(t *testing.T) {
	const w, h = 32, 24
	bm := &core.Bitmap{Width: w, Height: h, Channels: 1, Pix: make([]byte, w*h)}
	for i := range bm.Pix {
		bm.Pix[i] = 255
	}

	cases := []struct {
		name             string
		centerx, centery int
		dir              int
	}{
		{"diagonal at bottom-right corner", w - 1, h - 1, 2},
		{"diagonal near right edge", w - 2, h / 2, 2},
		{"horizontal at right edge", w - 1, h - 1, 0},
		{"vertical at bottom edge", w - 1, h - 1, 1},
		{"center outside image", w, h, 2},
		{"negative center", -1, -1, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// A module size spanning a large part of the image makes every
			// walk run into a border.
			crossCheckColor(bm, 255, w/2, 5, tc.centerx, tc.centery, tc.dir, 3)
		})
	}
	if !crossCheckColor(bm, 255, 2, 5, w/2, h/2, 2, 3) {
		t.Error("uniform interior candidate rejected")
	}
}

func TestCrossCheckColorReadsDeferredMask(t *testing.T) {
	const w, h = 12, 10
	lazy := &core.Bitmap{Width: w, Height: h, Channels: 1}
	lazy.SetPixelReader(func(x, y int) byte {
		if x >= 3 && x < 9 && y >= 2 && y < 8 {
			return 255
		}
		return 0
	})
	if lazy.Pix != nil {
		t.Fatal("deferred mask unexpectedly materialized")
	}
	if !crossCheckColor(lazy, 255, 2, 5, 6, 5, 2, 3) {
		t.Fatal("cross-check rejected deferred mask pixels")
	}
}
