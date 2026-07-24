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
	enc := srgbEncoder()

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
					out.Pix[off+x*bpp] = enc.encode(plane[row+x])
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

// boxBlurV is boxBlurH along columns. Columns are swept together rather than
// one at a time: a per-column running sum advances a row at a time, so every
// read walks a contiguous run of src instead of striding a full row width per
// access. The arithmetic per column is the same sequence of adds and subs in
// the same order, so the result is bit-identical to the column-at-a-time form.
func boxBlurVScalar(src, dst []float64, w, h, radius int) {
	win := float64(2*radius + 1)
	core.ParallelChunks(w, 64, func(xlo, xhi int) {
		sums := make([]float64, xhi-xlo)
		for k := -radius; k <= radius; k++ {
			row := min(max(k, 0), h-1)*w + xlo
			for i := range sums {
				sums[i] += src[row+i]
			}
		}
		for y := range h {
			out := y*w + xlo
			for i, sum := range sums {
				dst[out+i] = sum / win
			}
			if y+1 == h {
				break
			}
			add := min(max(y+1+radius, 0), h-1)*w + xlo
			sub := min(max(y-radius, 0), h-1)*w + xlo
			for i := range sums {
				sums[i] += src[add+i] - src[sub+i]
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

// linearToSRGB encodes a linear-light component to an 8-bit sRGB value,
// byte-identically to the closed form (the boundary table is bisected out of
// that form itself, and a unit test sweeps the two against each other).
//
// Callers with a pixel loop should hoist srgbEncoder once and use its encode
// method; this wrapper re-resolves the shared tables on every call.
func linearToSRGB(c float64) byte {
	return srgbEncoder().encode(c)
}

// srgbEncodeBits sets the resolution of the direct-index table below. A bucket
// is 2^-16 of the linear range, more than an order of magnitude finer than the
// closest byte boundary spacing (~3e-4 near black, where the encode is
// steepest), so the correction scan almost always settles immediately.
const srgbEncodeBits = 16

// srgbEncode maps a linear-light component onto its sRGB byte in constant
// time. The former per-pixel binary search over 255 boundaries cost eight
// unpredictable branches on the descreen hot path; start indexes the answer
// directly and the exact boundary table only settles the last step.
type srgbEncode struct {
	bounds *[255]float64
	start  [1 << srgbEncodeBits]byte
}

// encode returns the number of boundaries at or below c, which is exactly what
// the boundary binary search computed. start holds that count for the bucket's
// lower edge; since both the bucket and the table ascend, the true answer is
// never below it and the scan only moves forward.
func (e *srgbEncode) encode(c float64) byte {
	if !(c > 0) { // also catches NaN, which the search bottomed out at 0
		return 0
	}
	if c >= 1 {
		return 255
	}
	b := int(e.start[int(c*(1<<srgbEncodeBits))])
	for b < 255 && e.bounds[b] <= c {
		b++
	}
	return byte(b)
}

// srgbEncoder builds the direct-index table by merging the two ascending
// sequences once, rather than searching the boundaries per entry.
var srgbEncoder = sync.OnceValue(func() *srgbEncode {
	e := &srgbEncode{bounds: srgbBounds()}
	count := 0
	for i := range e.start {
		edge := float64(i) / (1 << srgbEncodeBits)
		for count < len(e.bounds) && e.bounds[count] <= edge {
			count++
		}
		e.start[i] = byte(count)
	}
	return e
})

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
