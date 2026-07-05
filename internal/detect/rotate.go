package detect

import (
	"image"
	"image/draw"
	"math"

	"github.com/srlehn/jabcode/internal/core"
)

// coarseProbeAngles are the pre-rotation angles, in degrees, the coarse orientation search
// tries when an upright read fails. Because the finder arrangement is identical under a
// 90-degree turn, the orientation family is fully determined within a single 90-degree
// window: counter-rotating by one of these six angles brings any orientation to within
// 7.5 degrees of an alias (15-degree steps tiling [0, 90), with 75 wrapping to 0+90).
// The rotation gating measurement shows 7.5 degrees still detects while 10 degrees can
// notch-fail and beyond ~20 degrees the cross-checks collapse, so the 7.5-degree worst-case
// residual sits inside the measured survival band. The search then expands the chosen rung
// to its four 90-degree turns to cover the other three quadrants.
var coarseProbeAngles = []float64{0, 15, 30, 45, 60, 75}

// RotateImage returns src rotated by angleDeg about its centre onto an expanded
// canvas, bilinearly resampled, with a white quiet-zone background outside the
// source rectangle. Decode uses it to pre-rotate a capture before re-attempting a
// read; the residual angle after a ladder rung is small, so a single bilinear pass
// does not meaningfully degrade detection.
func RotateImage(src image.Image, angleDeg float64) image.Image {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	in := image.NewNRGBA(image.Rect(0, 0, w, h))
	draw.Draw(in, in.Bounds(), src, b.Min, draw.Src)

	rad := angleDeg * math.Pi / 180
	cs, sn := math.Cos(rad), math.Sin(rad)
	nw := int(math.Ceil(math.Abs(float64(w)*cs) + math.Abs(float64(h)*sn)))
	nh := int(math.Ceil(math.Abs(float64(w)*sn) + math.Abs(float64(h)*cs)))
	out := image.NewNRGBA(image.Rect(0, 0, nw, nh))

	cx, cy := float64(w)/2, float64(h)/2
	ncx, ncy := float64(nw)/2, float64(nh)/2
	core.ParallelRows(nh, func(lo, hi int) {
		for y := lo; y < hi; y++ {
			for x := range nw {
				dx, dy := float64(x)-ncx, float64(y)-ncy
				sx := cs*dx + sn*dy + cx // inverse-map dest -> source (rotate by -angle)
				sy := -sn*dx + cs*dy + cy
				o := y*out.Stride + x*4
				r, g, bl, ok := bilinearNRGBA(in, w, h, sx, sy)
				if !ok {
					r, g, bl = 255, 255, 255 // white quiet zone outside the source
				}
				out.Pix[o+0], out.Pix[o+1], out.Pix[o+2], out.Pix[o+3] = r, g, bl, 255
			}
		}
	})
	return out
}

// bilinearNRGBA samples (sx, sy) from in by bilinear interpolation, reporting
// ok=false when the point lies outside the source rectangle.
func bilinearNRGBA(in *image.NRGBA, w, h int, sx, sy float64) (r, g, b byte, ok bool) {
	if sx < 0 || sy < 0 || sx > float64(w-1) || sy > float64(h-1) {
		return 0, 0, 0, false
	}
	x0, y0 := int(sx), int(sy)
	x1, y1 := min(x0+1, w-1), min(y0+1, h-1)
	fx, fy := sx-float64(x0), sy-float64(y0)
	at := func(x, y, c int) float64 { return float64(in.Pix[y*in.Stride+x*4+c]) }
	ch := func(c int) byte {
		return byte(at(x0, y0, c)*(1-fx)*(1-fy) + at(x1, y0, c)*fx*(1-fy) +
			at(x0, y1, c)*(1-fx)*fy + at(x1, y1, c)*fx*fy + 0.5)
	}
	return ch(0), ch(1), ch(2), true
}
