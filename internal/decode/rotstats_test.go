//go:build jabharness

package decode

import (
	"fmt"
	"image"
	"math"
	"strings"
	"testing"
)

// TestRotationStats is the rotation gating measurement: for a clean encoded code
// rotated through a range of angles, it dumps the raw finder pass's counters
// (run-length hits, branch routing, the red branch's intermediate gates, per-type
// cross-check survivors) so the detection collapse can be attributed. It rotates
// with both nearest-neighbour and bilinear resampling: if nearest-neighbour
// survives where bilinear collapses, the loss is resampling blur (a relaxed gate
// would help); if both collapse together it is the rotated-raster geometry, for
// which pre-rotation is the main lever, though a digitization-tolerant confirmation
// gate could still widen each rung's residual-angle range. Build-tagged:
//
//	go test -tags jabharness -run TestRotationStats -v ./internal/decode/
func TestRotationStats(t *testing.T) {
	gt := encodeGroundTruth(t, []byte("rotation gating measurement on a clean code"))
	angles := []float64{0, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 35, 40, 45}
	resamplers := []struct {
		name string
		fn   func(image.Image, float64) image.Image
	}{
		{"bilinear", func(img image.Image, a float64) image.Image { return rotateDeg(img, a, nil) }},
		{"nearest", rotateNearest},
	}

	// rCol/rCls are the red branch's intermediate gates (inner core-colour check,
	// then classification); they isolate where FP1/FP2 candidates die before the
	// cross-check. The blue branch (FP0/FP3) has no such intermediate counter, so
	// its sub-gate is not isolated here.
	var b strings.Builder
	fmt.Fprintf(&b, "%-9s %5s  %8s %6s %6s %6s %6s  %-19s %5s %s\n",
		"resample", "ang", "rawHits", "blue", "red", "rCol", "rCls", "cross FP0/1/2/3", "miss", "status")
	for _, rs := range resamplers {
		for _, ang := range angles {
			bm := bitmapFromImage(rs.fn(gt.img, ang))
			balanceRGB(bm)
			ch := binarizerRGB(bm, nil)
			d := &primaryDetector{bm: bm, ch: ch, mode: intensiveDetect}
			st := d.findPrimarySymbol() // raw pass only, no retry/descreen
			p := d.stats.passes[0]
			fmt.Fprintf(&b, "%-9s %5g  %8d %6d %6d %6d %6d  %4d/%-4d/%-4d/%-4d %5d %s\n",
				rs.name, ang, p.rawHits, p.branchBlue, p.branchRed, p.redColor, p.redClassified,
				p.crossSurvivors[0], p.crossSurvivors[1], p.crossSurvivors[2], p.crossSurvivors[3],
				p.missing, statusName(st))
		}
	}
	t.Logf("rotation raw-pass finder stats (clean code):\n%s", b.String())
}

// rotateNearest rotates src by angleDeg about its centre onto an expanded
// white-quiet-zone canvas using nearest-neighbour sampling. Unlike the harness's
// bilinear rotateDeg it introduces no inter-pixel averaging, so comparing the two
// isolates resampling blur from the rotation geometry itself.
func rotateNearest(src image.Image, angleDeg float64) image.Image {
	if angleDeg == 0 {
		return src
	}
	in := toNRGBA(src)
	w, h := in.Bounds().Dx(), in.Bounds().Dy()
	rad := angleDeg * math.Pi / 180
	cs, sn := math.Cos(rad), math.Sin(rad)
	nw := int(math.Ceil(math.Abs(float64(w)*cs) + math.Abs(float64(h)*sn)))
	nh := int(math.Ceil(math.Abs(float64(w)*sn) + math.Abs(float64(h)*cs)))
	out := image.NewNRGBA(image.Rect(0, 0, nw, nh))
	cx, cy := float64(w)/2, float64(h)/2
	ncx, ncy := float64(nw)/2, float64(nh)/2
	for y := range nh {
		for x := range nw {
			dx, dy := float64(x)-ncx, float64(y)-ncy
			sx := cs*dx + sn*dy + cx // inverse-map dest -> source (rotate by -angle)
			sy := -sn*dx + cs*dy + cy
			o := y*out.Stride + x*4
			ix, iy := int(sx+0.5), int(sy+0.5)
			if ix < 0 || iy < 0 || ix >= w || iy >= h {
				out.Pix[o+0], out.Pix[o+1], out.Pix[o+2], out.Pix[o+3] = 255, 255, 255, 255
				continue
			}
			so := iy*in.Stride + ix*4
			out.Pix[o+0], out.Pix[o+1], out.Pix[o+2], out.Pix[o+3] = in.Pix[so+0], in.Pix[so+1], in.Pix[so+2], 255
		}
	}
	return out
}
