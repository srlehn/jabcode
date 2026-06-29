package jabcode

import "math"

// descreenSchedule returns the sequence of (rx, ry) box-blur half-widths the
// finder-detection retry walks for a capture whose estimated lattice pitch is
// (px, py) (from estimatePitch): first ≈ one grid cell, then a coarser ≈ two-cell
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

// descreen returns a low-pass copy of bm that fuses display-subpixel stripes and
// suppresses moiré before colour binarization, leaving bm untouched so colour
// sampling still reads the original pixels. The separable box average is computed
// in linear light (sRGB-decoded, then re-encoded) so the fusion is photometric.
// rx and ry are the per-axis box half-widths in pixels (anisotropic, since a
// screen's horizontal subpixel stripe pitch and vertical pitch differ); a radius
// < 1 on an axis is an identity pass, and rx,ry both < 1 is a plain copy.
func descreen(bm *bitmap, rx, ry int) *bitmap {
	out := newBitmap(bm.width, bm.height, bm.channels)
	copy(out.pix, bm.pix)
	if rx < 1 && ry < 1 {
		return out
	}
	w, h, bpp := bm.width, bm.height, bm.channels

	var dec [256]float64
	for i := range dec {
		dec[i] = srgbToLinear(float64(i) / 255)
	}

	plane := make([]float64, w*h)
	tmp := make([]float64, w*h)
	for c := range 3 {
		for y := range h {
			off := y*w*bpp + c
			row := y * w
			for x := range w {
				plane[row+x] = dec[bm.pix[off+x*bpp]]
			}
		}
		boxBlurH(plane, tmp, w, h, rx)
		boxBlurV(tmp, plane, w, h, ry)
		for y := range h {
			off := y*w*bpp + c
			row := y * w
			for x := range w {
				out.pix[off+x*bpp] = linearToSRGB(plane[row+x])
			}
		}
	}
	return out
}

// boxBlurH writes into dst the horizontal moving average of src over a 2*radius+1
// window with edge clamping, using a running sum.
func boxBlurH(src, dst []float64, w, h, radius int) {
	win := float64(2*radius + 1)
	for y := range h {
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
}

// boxBlurV is boxBlurH along columns.
func boxBlurV(src, dst []float64, w, h, radius int) {
	win := float64(2*radius + 1)
	for x := range w {
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
}

// srgbToLinear decodes an sRGB component in [0,1] to linear light.
func srgbToLinear(c float64) float64 {
	if c <= 0.04045 {
		return c / 12.92
	}
	return math.Pow((c+0.055)/1.055, 2.4)
}

// linearToSRGB encodes a linear-light component to an 8-bit sRGB value.
func linearToSRGB(c float64) byte {
	var s float64
	if c <= 0.0031308 {
		s = c * 12.92
	} else {
		s = 1.055*math.Pow(c, 1/2.4) - 0.055
	}
	return byte(min(max(s*255+0.5, 0), 255))
}
