package detect

import (
	"image"
	"image/color"
	"math"
	"testing"

	"github.com/srlehn/jabcode/internal/encode"
)

// TestDownscaleToMax checks the box-filter downscale bounds the longer side, leaves an
// already-small image's dimensions alone, and preserves a uniform colour exactly.
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

// TestCoarseOrientationRungs checks the coarse search points the full-resolution retry at
// the angle that counter-rotates a rotated code back toward upright.
func TestCoarseOrientationRungs(t *testing.T) {
	msg := []byte("coarse orientation probe")
	img, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 12, SymbolNumber: 1}, msg)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	rungs := CoarseOrientationRungs(RotateImage(img, 30))
	// The probe returns each retained family's four 90-degree turns, so the result is a
	// non-zero multiple of four and the true orientation is among the turns.
	if len(rungs) == 0 || len(rungs)%4 != 0 {
		t.Fatalf("got %d rungs %v, want a non-zero multiple of 4", len(rungs), rungs)
	}
	// The image is rotated +30, so the true counter-rotation is near -30; searching a 90
	// degree window at 15 degree steps puts one returned rung within 7.5 degrees of it.
	best := math.Inf(1)
	for _, r := range rungs {
		if residual := math.Abs(math.Remainder(30+r, 360)); residual < best {
			best = residual
		}
	}
	if best > 7.5 {
		t.Errorf("no rung in %v counter-rotates +30 to within 7.5 deg (best residual %g deg)", rungs, best)
	}
}
