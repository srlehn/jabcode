package detect

// Image scaling shared by the resolution pyramid and the enlarged detection
// scale.

import (
	"image"
	"image/draw"
	"math"

	"github.com/srlehn/jabcode/internal/core"
)

// zeroOriginNRGBA returns src itself when it already is a tightly packed
// zero-origin NRGBA image, or nil. Callers use it to skip a full-frame copy;
// the aliased image must be treated read-only.
func zeroOriginNRGBA(src image.Image) *image.NRGBA {
	n, ok := src.(*image.NRGBA)
	if !ok || n.Rect.Min != (image.Point{}) || n.Stride != n.Rect.Dx()*4 {
		return nil
	}
	return n
}

// DownscaleToMax returns src reduced so its longer side is at most maxDim, by averaging
// each destination pixel over the source box it covers (a box filter, which preserves the
// finder rings better than point sampling at the reduction the coarse search needs). An
// image already within the bound is returned unreduced - aliased when it already is a
// zero-origin NRGBA, as an NRGBA copy otherwise - so treat the result read-only.
func DownscaleToMax(src image.Image, maxDim int) *image.NRGBA {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	in := zeroOriginNRGBA(src)
	if in == nil {
		in = image.NewNRGBA(image.Rect(0, 0, w, h))
		draw.Draw(in, in.Bounds(), src, b.Min, draw.Src)
	}
	if w <= maxDim && h <= maxDim {
		return in
	}
	scale := float64(maxDim) / float64(max(w, h))
	nw := max(int(math.Round(float64(w)*scale)), 1)
	nh := max(int(math.Round(float64(h)*scale)), 1)
	return downscaleNRGBA(in, nw, nh)
}

// HalveNRGBA returns in box-filtered to half size, rounding odd sides up - the
// constructor for successive resolution-pyramid levels. Each destination pixel
// averages the 2x2 source box it covers, so every halving is also a mild
// low-pass: coarse levels arrive pre-smoothed, which is why they can decode
// captures whose full-resolution noise defeats detection.
func HalveNRGBA(in *image.NRGBA) *image.NRGBA {
	w, h := in.Rect.Dx(), in.Rect.Dy()
	return downscaleNRGBA(in, max((w+1)/2, 1), max((h+1)/2, 1))
}

// downscaleNRGBA box-filters in down to nw x nh, averaging each destination
// pixel over the source box it covers.
func downscaleNRGBA(in *image.NRGBA, nw, nh int) *image.NRGBA {
	w, h := in.Rect.Dx(), in.Rect.Dy()
	base := in.PixOffset(in.Rect.Min.X, in.Rect.Min.Y)
	out := image.NewNRGBA(image.Rect(0, 0, nw, nh))
	core.ParallelChunks(nh, 8, func(lo, hi int) {
		for oy := lo; oy < hi; oy++ {
			sy0 := oy * h / nh
			sy1 := max((oy+1)*h/nh, sy0+1)
			for ox := range nw {
				sx0 := ox * w / nw
				sx1 := max((ox+1)*w/nw, sx0+1)
				var rs, gs, bs, n int
				for sy := sy0; sy < sy1; sy++ {
					row := base + sy*in.Stride
					for sx := sx0; sx < sx1; sx++ {
						o := row + sx*4
						rs += int(in.Pix[o+0])
						gs += int(in.Pix[o+1])
						bs += int(in.Pix[o+2])
						n++
					}
				}
				o := oy*out.Stride + ox*4
				out.Pix[o+0] = byte(rs / n)
				out.Pix[o+1] = byte(gs / n)
				out.Pix[o+2] = byte(bs / n)
				out.Pix[o+3] = 255
			}
		}
	})
	return out
}
