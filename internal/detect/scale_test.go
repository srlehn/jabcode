package detect

import (
	"image"
	"image/color"
	"testing"
)

func TestDownscaleToMax(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 1000, 600))
	grey := color.NRGBA{100, 150, 200, 255}
	for y := range 600 {
		for x := range 1000 {
			src.SetNRGBA(x, y, grey)
		}
	}

	out := DownscaleToMax(src, 250)
	if out.Bounds().Dx() > 250 || out.Bounds().Dy() > 250 {
		t.Fatalf("downscaled bounds %v exceed max 250", out.Bounds())
	}
	if out.Bounds().Dx() != 250 {
		t.Errorf("longer side = %d, want 250", out.Bounds().Dx())
	}
	if got := out.NRGBAAt(out.Bounds().Dx()/2, out.Bounds().Dy()/2); got != grey {
		t.Errorf("uniform colour not preserved: got %v, want %v", got, grey)
	}

	if small := DownscaleToMax(src, 2000); small.Bounds().Dx() != 1000 || small.Bounds().Dy() != 600 {
		t.Errorf("image within bound was resized to %v, want 1000x600", small.Bounds())
	}
}
