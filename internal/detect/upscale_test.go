package detect

import (
	"image"
	"testing"
)

// TestUpscaleNRGBA checks the properties the enlarged detection scale relies
// on: the frame grows by the factor, a flat area keeps its exact value (the
// kernel taps sum to one), and an edge stays an edge - placed between the
// source samples instead of duplicated onto the new grid, which is the only
// reason enlarging helps the run-length checks at all.
func TestUpscaleNRGBA(t *testing.T) {
	const w, h = 8, 6
	src := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			v := byte(40)
			if x >= w/2 {
				v = 200
			}
			o := y*src.Stride + x*4
			src.Pix[o+0], src.Pix[o+1], src.Pix[o+2], src.Pix[o+3] = v, v, v, 255
		}
	}

	if got := UpscaleNRGBA(src, 1); got != src {
		t.Error("factor 1 did not return the input frame")
	}

	out := UpscaleNRGBA(src, 2)
	if out.Rect.Dx() != 2*w || out.Rect.Dy() != 2*h {
		t.Fatalf("enlarged bounds = %v, want %dx%d", out.Rect, 2*w, 2*h)
	}
	at := func(x, y int) byte { return out.Pix[y*out.Stride+x*4] }
	if got := at(1, 4); got != 40 {
		t.Errorf("flat dark area = %d, want 40", got)
	}
	if got := at(2*w-2, 4); got != 200 {
		t.Errorf("flat light area = %d, want 200", got)
	}
	if got := out.Pix[3]; got != 255 {
		t.Errorf("alpha = %d, want 255", got)
	}
	// The edge sits between source columns 3 and 4, so the enlarged frame
	// carries intermediate samples there and reaches both extremes around it.
	left, mid, right := at(2*w/2-2, 4), at(2*w/2-1, 4), at(2*w/2, 4)
	if !(left <= mid && mid <= right) {
		t.Errorf("edge not monotone across the enlarged samples: %d %d %d", left, mid, right)
	}
	if mid == 40 || mid == 200 {
		t.Errorf("edge sample %d only duplicates a source value", mid)
	}
}
