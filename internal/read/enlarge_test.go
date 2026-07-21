package read

import (
	"image"
	"testing"

	"github.com/srlehn/jabcode/internal/wire"
)

// TestEnlargedScaleOnlyForSingleScaleFrames pins the threshold that decides
// where an interpolated enlargement is worth paying for: only a frame too
// small to carry a pyramid. A frame that holds one already has real pixels at
// every scale it needs, and enlarging it would quadruple the cost of the
// slowest reads for nothing.
func TestEnlargedScaleOnlyForSingleScaleFrames(t *testing.T) {
	cases := []struct {
		size image.Point
		want bool
	}{
		{image.Pt(480, 640), true},
		{image.Pt(2*minPyramidSide - 1, 4000), true},
		{image.Pt(2 * minPyramidSide, 2 * minPyramidSide), false},
		{image.Pt(3024, 4032), false},
	}
	for _, tc := range cases {
		if got := singleScaleFrame(tc.size); got != tc.want {
			t.Errorf("singleScaleFrame(%v) = %v, want %v", tc.size, got, tc.want)
		}
	}

	// A frame above the threshold must be refused before any enlargement is
	// built, whatever the caller does.
	big := image.NewNRGBA(image.Rect(0, 0, 2*minPyramidSide, 2*minPyramidSide))
	if _, _, ok := decodeEnlarged(big, nil, nil, wire.ISO23634.Mask()); ok {
		t.Error("decodeEnlarged reported a decode on a frame that carries a pyramid")
	}
}
