package detect

import (
	"image"
	"image/color"
	"math/rand"
	"testing"
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
