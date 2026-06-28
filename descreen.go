package jabcode

import "math"

// descreenRadii is the multi-scale ladder the finder-detection retry walks when
// the raw and avg-RGB passes both fail. It is a fixed placeholder: a screen's
// diode-lattice pitch and a code's module size in pixels both vary with display
// resolution and capture distance, so these radii must be replaced by a per-image
// pitch estimate rather than left as constants.
var descreenRadii = []int{1, 2, 3}

// descreen returns a low-pass copy of bm that fuses display-subpixel stripes and
// suppresses moiré before colour binarization, leaving bm untouched so colour
// sampling still reads the original pixels. The separable box average is computed
// in linear light (sRGB-decoded, then re-encoded) so the fusion is photometric.
// radius is the box half-width in pixels; radius < 1 is a plain copy.
func descreen(bm *bitmap, radius int) *bitmap {
	out := newBitmap(bm.width, bm.height, bm.channels)
	copy(out.pix, bm.pix)
	if radius < 1 {
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
		boxBlurH(plane, tmp, w, h, radius)
		boxBlurV(tmp, plane, w, h, radius)
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
