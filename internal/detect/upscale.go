package detect

import (
	"image"
	"math"

	"github.com/srlehn/jabcode/internal/core"
)

// crossCheckModulePixels is the module size, in pixels, below which the finder
// cross-checks stop confirming candidates. The five-run n-1-1-1-m machine
// merges any run under three pixels into its neighbour, and checkPatternCross
// then admits each single-module run only within half its layer estimate, so a
// module that quantizes to three pixels fails as soon as binarization rounds
// one edge outward and the opposite one inward. Half a module of headroom over
// that merge threshold is the point where both roundings still fit.
const crossCheckModulePixels = 4.5

// SmallestVerifiableFrame returns the shorter-side length a frame needs before
// a maximum-size primary symbol can present its modules at the scale the
// finder cross-checks require: the largest primary side (side version 32, 145
// modules) at that module scale, edge to edge with no margin. Below it, no
// primary symbol placement resolves - the frame itself is the limit, not the
// framing - which is the one case where enlarging the pixels is worth its
// cost.
//
// Deliberately primary-only: a clustered or docked arrangement spans several
// symbols and would raise this bound far past what a single symbol needs,
// turning a floor into an excuse to enlarge ordinary captures.
func SmallestVerifiableFrame() int {
	return int(math.Ceil(maxModules * crossCheckModulePixels))
}

// UpscaleNRGBA returns in enlarged by an integer factor with a separable
// Catmull-Rom kernel. Nearest-neighbour enlargement would be pointless here -
// it multiplies every run length and every quantization error alike, leaving
// the finder ratio checks exactly where they were - so the interpolation is
// the point: it places module edges from the surrounding samples instead of
// the source grid. The kernel's mild sharpening keeps those edges from
// smearing across the extra pixels the way a bilinear enlargement does.
func UpscaleNRGBA(in *image.NRGBA, factor int) *image.NRGBA {
	w, h := in.Rect.Dx(), in.Rect.Dy()
	if factor < 2 || w == 0 || h == 0 {
		return in
	}
	nw, nh := w*factor, h*factor
	base := in.PixOffset(in.Rect.Min.X, in.Rect.Min.Y)

	// Output pixel ox samples source position (ox+0.5)/factor - 0.5, which
	// keeps the enlargement centred instead of shifting the whole frame by
	// half an output pixel. That position repeats its fractional part every
	// factor pixels, so each phase has one fixed set of four kernel taps and
	// one fixed source offset (the phases below the source pixel centre read
	// one pixel further left).
	taps := make([][4]float32, factor)
	shift := make([]int, factor)
	for phase := range factor {
		t := (float64(phase)+0.5)/float64(factor) - 0.5
		if t < 0 {
			t, shift[phase] = t+1, -1
		}
		taps[phase] = catmullRomTaps(t)
	}

	// Horizontal pass into an intermediate enlarged in x only, then the
	// vertical one: the separable form costs 8 taps per output pixel
	// instead of the 16 a two-dimensional kernel would.
	rows := make([]float32, nw*h*3)
	core.ParallelChunks(h, 8, func(lo, hi int) {
		for y := lo; y < hi; y++ {
			row := base + y*in.Stride
			for ox := range nw {
				phase := ox % factor
				sx := ox/factor + shift[phase]
				k := taps[phase]
				var r, g, b float32
				for tap := range 4 {
					o := row + clampIndex(sx-1+tap, w)*4
					r += k[tap] * float32(in.Pix[o+0])
					g += k[tap] * float32(in.Pix[o+1])
					b += k[tap] * float32(in.Pix[o+2])
				}
				d := (y*nw + ox) * 3
				rows[d+0], rows[d+1], rows[d+2] = r, g, b
			}
		}
	})

	out := image.NewNRGBA(image.Rect(0, 0, nw, nh))
	core.ParallelChunks(nh, 8, func(lo, hi int) {
		for oy := lo; oy < hi; oy++ {
			phase := oy % factor
			sy := oy/factor + shift[phase]
			k := taps[phase]
			for ox := range nw {
				var r, g, b float32
				for tap := range 4 {
					s := (clampIndex(sy-1+tap, h)*nw + ox) * 3
					r += k[tap] * rows[s+0]
					g += k[tap] * rows[s+1]
					b += k[tap] * rows[s+2]
				}
				o := oy*out.Stride + ox*4
				out.Pix[o+0] = clampByte(r)
				out.Pix[o+1] = clampByte(g)
				out.Pix[o+2] = clampByte(b)
				out.Pix[o+3] = 255
			}
		}
	})
	return out
}

// catmullRomTaps returns the four Catmull-Rom weights for a sample sitting a
// fraction t between the second and third of four consecutive source pixels.
func catmullRomTaps(t float64) [4]float32 {
	t2 := t * t
	t3 := t2 * t
	return [4]float32{
		float32(-0.5*t3 + t2 - 0.5*t),
		float32(1.5*t3 - 2.5*t2 + 1),
		float32(-1.5*t3 + 2*t2 + 0.5*t),
		float32(0.5*t3 - 0.5*t2),
	}
}

func clampIndex(i, n int) int {
	return min(max(i, 0), n-1)
}

func clampByte(v float32) byte {
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return byte(v + 0.5)
}
