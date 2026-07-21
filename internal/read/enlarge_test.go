package read

import (
	"image"
	"testing"

	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/wire"
)

// TestEnlargedScaleFrameLimit pins where an interpolated enlargement is worth
// paying for: only a frame too small to have resolved any primary symbol
// placement at all. The limit is derived from the largest primary side at the
// cross-check module floor, so a frame above it failed for some other reason
// and must not pay four times the pixels to fail again.
func TestEnlargedScaleFrameLimit(t *testing.T) {
	limit := detect.SmallestVerifiableFrame()
	if limit < 2*minPyramidSide {
		t.Errorf("frame limit %d dropped below the pyramid threshold %d: the "+
			"single-scale path would no longer reach every enlargeable frame", limit, 2*minPyramidSide)
	}

	cases := []struct {
		size image.Point
		want bool
	}{
		{image.Pt(480, 640), true},
		{image.Pt(limit-1, limit-1), true},
		{image.Pt(limit, limit), false},
		{image.Pt(3024, 4032), false},
	}
	for _, tc := range cases {
		// A blank frame stops at the enlarged ladder's first attempt, so the
		// recorded attempts say whether the enlargement was built at all.
		tr := &routeTrace{level: -1}
		if _, _, ok := decodeEnlarged(image.NewNRGBA(image.Rect(0, 0, tc.size.X, tc.size.Y)),
			nil, tr, wire.ISO23634.Mask()); ok {
			t.Errorf("decodeEnlarged(%v) reported a decode of a blank frame", tc.size)
		}
		if got := len(tr.attempts) > 0; got != tc.want {
			t.Errorf("decodeEnlarged(%v) enlarged = %v, want %v", tc.size, got, tc.want)
		}
	}
}
