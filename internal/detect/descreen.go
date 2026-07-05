package detect

import (
	"math"
	"slices"
	"sync"

	"github.com/srlehn/jabcode/internal/core"
)

// descreenSchedule returns the sequence of (rx, ry) box-blur half-widths the
// finder-detection retry walks for a capture whose estimated lattice pitch is
// (px, py) (from EstimatePitch): first ≈ one grid cell, then a coarser ≈ two-cell
// pass for residual moiré. A zero pitch on an axis leaves that axis unblurred.
// Returns nil when no lattice was detected on either axis, so the caller can skip
// descreening entirely rather than copy the bitmap for nothing.
func descreenSchedule(px, py int) [][2]int {
	rx, ry := cellRadius(px), cellRadius(py)
	if rx == 0 && ry == 0 {
		return nil
	}
	return [][2]int{{rx, ry}, {rx * 2, ry * 2}}
}

// cellRadius converts a lattice pitch in pixels to a box half-width spanning ≈ one
// grid cell (window 2r+1 ≈ pitch). A non-positive pitch means no lattice on that
// axis, so the radius is 0 (an identity blur).
func cellRadius(pitch int) int {
	if pitch <= 0 {
		return 0
	}
	return max(1, pitch/2)
}

// Print-retry evidence gate: raw run-length seeds by the hundred with
// (almost) no cross-check survivors is the signature of print structure -
// dark subtractive colours mis-gated to black, halftone cells, dither grain,
// colorant fringes - defeating the finder cross-checks. Both are counts, not
// pixel sizes; the retry's low-pass radius comes from the seeds' own
// module-size estimates.
const (
	printRetryMinSeeds     = 100
	printRetryMaxSurvivors = 2
)

// printBlurLeadRadius is the smallest low-pass radius that still leads the
// print retry: below it the integer radius is a large fraction of the module
// (quantization dominates) and the blur shifts finder centres more than it
// fuses grain, so the sharp pass goes first. Measured: radius 4 on 12 px
// modules recovers geometry the sharp pass got wrong, radius 2 on 6 px
// modules destroys centre precision the sharp pass had.
const printBlurLeadRadius = 3

// seedModuleScale returns the median of the raw seeds' module-size
// estimates, reordering v in place. Even where most seeds are false hits on
// print speckle, their qualifying run windows measure module-ish scale
// (measured median 16.7 px on a 12 px-module print), so the median tracks
// the module size closely enough to derive a blur radius; larger radii were
// measured to cost cross-check survivors.
func seedModuleScale(v []float64) float64 {
	slices.Sort(v)
	return v[len(v)/2]
}

// descreen returns a low-pass copy of bm that fuses display-subpixel stripes and
// suppresses moiré before colour binarization, leaving bm untouched so colour
// sampling still reads the original pixels. The separable box average is computed
// in linear light (sRGB-decoded, then re-encoded) so the fusion is photometric.
// rx and ry are the per-axis box half-widths in pixels (anisotropic, since a
// screen's horizontal subpixel stripe pitch and vertical pitch differ); a radius
// < 1 on an axis is an identity pass, and rx,ry both < 1 is a plain copy.
func descreen(bm *core.Bitmap, rx, ry int) *core.Bitmap {
	out := core.NewBitmap(bm.Width, bm.Height, bm.Channels)
	copy(out.Pix, bm.Pix)
	if rx < 1 && ry < 1 {
		return out
	}
	w, h, bpp := bm.Width, bm.Height, bm.Channels

	var dec [256]float64
	for i := range dec {
		dec[i] = srgbToLinear(float64(i) / 255)
	}

	plane := make([]float64, w*h)
	tmp := make([]float64, w*h)
	for c := range 3 {
		core.ParallelRows(h, func(ylo, yhi int) {
			for y := ylo; y < yhi; y++ {
				off := y*w*bpp + c
				row := y * w
				for x := range w {
					plane[row+x] = dec[bm.Pix[off+x*bpp]]
				}
			}
		})
		boxBlurH(plane, tmp, w, h, rx)
		boxBlurV(tmp, plane, w, h, ry)
		core.ParallelRows(h, func(ylo, yhi int) {
			for y := ylo; y < yhi; y++ {
				off := y*w*bpp + c
				row := y * w
				for x := range w {
					out.Pix[off+x*bpp] = linearToSRGB(plane[row+x])
				}
			}
		})
	}
	return out
}

// boxBlurH writes into dst the horizontal moving average of src over a 2*radius+1
// window with edge clamping, using a running sum per row.
func boxBlurH(src, dst []float64, w, h, radius int) {
	win := float64(2*radius + 1)
	core.ParallelRows(h, func(ylo, yhi int) {
		for y := ylo; y < yhi; y++ {
			base := y * w
			var sum float64
			for k := -radius; k <= radius; k++ {
				sum += src[base+min(max(k, 0), w-1)]
			}
			dst[base] = sum / win
			for x := 1; x < w; x++ {
				sum += src[base+min(max(x+radius, 0), w-1)] - src[base+min(max(x-1-radius, 0), w-1)]
				dst[base+x] = sum / win
			}
		}
	})
}

// boxBlurV is boxBlurH along columns.
func boxBlurV(src, dst []float64, w, h, radius int) {
	win := float64(2*radius + 1)
	core.ParallelChunks(w, 64, func(xlo, xhi int) {
		for x := xlo; x < xhi; x++ {
			var sum float64
			for k := -radius; k <= radius; k++ {
				sum += src[min(max(k, 0), h-1)*w+x]
			}
			dst[x] = sum / win
			for y := 1; y < h; y++ {
				sum += src[min(max(y+radius, 0), h-1)*w+x] - src[min(max(y-1-radius, 0), h-1)*w+x]
				dst[y*w+x] = sum / win
			}
		}
	})
}

// srgbToLinear decodes an sRGB component in [0,1] to linear light.
func srgbToLinear(c float64) float64 {
	if c <= 0.04045 {
		return c / 12.92
	}
	return math.Pow((c+0.055)/1.055, 2.4)
}

// linearToSRGB encodes a linear-light component to an 8-bit sRGB value. It
// binary-searches the boundary table instead of evaluating the closed form's
// math.Pow per pixel; the results are byte-identical (the table is bisected
// out of the closed form itself, and a unit test sweeps the two against each
// other).
func linearToSRGB(c float64) byte {
	bounds := srgbBounds()
	lo, hi := 0, len(bounds)
	for lo < hi {
		mid := (lo + hi) / 2
		if bounds[mid] <= c {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return byte(lo)
}

// linearToSRGBFormula is the closed-form encode, kept as the ground truth the
// boundary table is built from.
func linearToSRGBFormula(c float64) byte {
	var s float64
	if c <= 0.0031308 {
		s = c * 12.92
	} else {
		s = 1.055*math.Pow(c, 1/2.4) - 0.055
	}
	return byte(min(max(s*255+0.5, 0), 255))
}

// srgbBounds returns, at index i, the smallest linear-light value that the
// closed form encodes to a byte greater than i, found by float bisection of
// the closed form itself so any rounding quirk of that form is reproduced
// exactly.
var srgbBounds = sync.OnceValue(func() *[255]float64 {
	var bounds [255]float64
	for i := range bounds {
		b := byte(i + 1)
		lo, hi := 0.0, 1.0 // encode to 0 and 255
		for {
			mid := (lo + hi) / 2
			if mid == lo || mid == hi {
				break
			}
			if linearToSRGBFormula(mid) >= b {
				hi = mid
			} else {
				lo = mid
			}
		}
		bounds[i] = hi
	}
	return &bounds
})
