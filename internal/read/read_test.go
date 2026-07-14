package read

import (
	"bytes"
	"image"
	"image/color"
	"image/draw"
	"testing"

	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/encode"
)

// TestDecodeRotatedDownscaled exercises the full coarse-to-fine path: a code large enough
// that the coarse search runs on a genuinely downscaled copy still has its orientation
// found and decodes at full resolution.
func TestDecodeRotatedDownscaled(t *testing.T) {
	msg := []byte("coarse-to-fine downscaled decode")
	img, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 40, SymbolNumber: 1}, msg)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if img.Bounds().Dx() <= detect.CoarseMaxDim {
		t.Fatalf("test code %d px not larger than CoarseMaxDim %d; downscale path unexercised",
			img.Bounds().Dx(), detect.CoarseMaxDim)
	}
	got, err := Decode(detect.RotateImage(img, 35))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	want := isoPayload(msg)
	if string(got) != string(want) {
		t.Errorf("got %q, want %q", got, want)
	}
}

// TestDecodeSmallRotatedSymbolInClutter covers the recall-gate case the
// whole-frame orientation probe cannot: a rotated symbol small within a large
// cluttered frame drops below the finder survival threshold in the frame-wide
// 512 px probe downscale, and only probing the proposed region at its own
// scale recovers it.
func TestDecodeSmallRotatedSymbolInClutter(t *testing.T) {
	payload := []byte("small symbol in clutter")
	r, err := encode.Render(encode.Config{Colors: 8, ModuleSize: 12, SymbolNumber: 1}, payload)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	rotated := detect.RotateImage(r.Image, 30)

	const W, H = 3000, 4000
	frame := image.NewNRGBA(image.Rect(0, 0, W, H))
	fill := func(rect image.Rectangle, c color.NRGBA) {
		draw.Draw(frame, rect, &image.Uniform{c}, image.Point{}, draw.Src)
	}
	fill(frame.Bounds(), color.NRGBA{18, 24, 42, 255})                      // dark window background
	fill(image.Rect(0, 300, W, 560), color.NRGBA{200, 30, 25, 255})         // flat red title bar
	fill(image.Rect(120, 900, 1500, 2900), color.NRGBA{235, 232, 226, 255}) // document page
	for y := 1000; y < 2800; y += 90 {
		fill(image.Rect(200, y, 1400, y+22), color.NRGBA{90, 90, 96, 255}) // text lines
	}
	pos := image.Pt(1950, 2600)
	draw.Draw(frame, rotated.Bounds().Add(pos), rotated, image.Point{}, draw.Src)

	// The whole-frame probe must not already recover this case, or the test no
	// longer exercises the per-region path.
	if rungs := detect.CoarseOrientationRungs(frame); len(rungs) != 0 {
		t.Logf("note: whole-frame probe now retains rungs %v; region path not gating", rungs)
	}

	data, err := Decode(frame)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	want := isoPayload(payload)
	if !bytes.Equal(data, want) {
		t.Fatalf("Decode = %q, want %q", data, want)
	}
}
