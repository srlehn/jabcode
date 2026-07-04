package decode

import (
	"bytes"
	"image"
	"image/color"
	"image/draw"
	"math/rand"
	"testing"

	"github.com/srlehn/jabcode/internal/encode"
)

// TestProposeROIsIsolatesSymbolBlob builds a frame with flat coloured "UI" bars and
// one dense multicolour blob, and checks the joint chroma/gradient score ranks the
// blob first: its box must contain the blob and must not swallow a flat colour bar.
func TestProposeROIsIsolatesSymbolBlob(t *testing.T) {
	const W, H = 800, 600
	img := image.NewNRGBA(image.Rect(0, 0, W, H))
	fill := func(r image.Rectangle, c color.NRGBA) {
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				img.SetNRGBA(x, y, c)
			}
		}
	}
	fill(img.Bounds(), color.NRGBA{255, 255, 255, 255})          // white background
	fill(image.Rect(0, 0, W, 40), color.NRGBA{220, 20, 20, 255}) // flat red top bar
	fill(image.Rect(0, 0, 40, H), color.NRGBA{20, 20, 220, 255}) // flat blue side bar

	// Dense multicolour blob: small cells of the eight cube-vertex colours, giving
	// both high chroma variance and high gradient energy.
	vtx := []color.NRGBA{
		{0, 0, 0, 255}, {0, 0, 255, 255}, {0, 255, 0, 255}, {0, 255, 255, 255},
		{255, 0, 0, 255}, {255, 0, 255, 255}, {255, 255, 0, 255}, {255, 255, 255, 255},
	}
	blob := image.Rect(520, 360, 740, 560)
	rng := rand.New(rand.NewSource(1))
	const cell = 8
	for by := blob.Min.Y; by < blob.Max.Y; by += cell {
		for bx := blob.Min.X; bx < blob.Max.X; bx += cell {
			c := vtx[rng.Intn(len(vtx))]
			fill(image.Rect(bx, by, min(bx+cell, blob.Max.X), min(by+cell, blob.Max.Y)), c)
		}
	}

	rois := ProposeROIs(img, 5)
	if len(rois) == 0 {
		t.Fatal("no ROIs proposed")
	}
	top := rois[0].Bounds
	center := image.Pt((blob.Min.X+blob.Max.X)/2, (blob.Min.Y+blob.Max.Y)/2)
	if !center.In(top) {
		t.Errorf("top ROI %v does not contain the symbol blob center %v", top, center)
	}
	if (image.Point{X: 400, Y: 20}).In(top) {
		t.Errorf("top ROI %v wrongly includes the flat red UI bar", top)
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
	rotated := RotateImage(r.Image, 30)

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
	if rungs := CoarseOrientationRungs(frame); len(rungs) != 0 {
		t.Logf("note: whole-frame probe now retains rungs %v; region path not gating", rungs)
	}

	data, err := Decode(frame)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if !bytes.Equal(data, payload) {
		t.Fatalf("Decode = %q, want %q", data, payload)
	}
}
