package decode

import (
	"image"
	"math"
)

// Module-footprint sampling geometry: each module is averaged over the central
// sampleCoverage portion of its warped footprint, on a grid dense enough to
// visit roughly every source pixel there, bounded by maxSamplesPerAxis. The
// inset absorbs registration error and neighbour bleed at the module borders;
// within that margin the window is kept wide, because a wider window averages
// more periods of any screen lattice and so shrinks the phase-dependent
// brightness swing of the module mean (measured on the degradation harness).
const (
	sampleCoverage    = 0.8
	maxSamplesPerAxis = 32
)

// sampleOffsets returns k offsets in module units spanning the central
// sampleCoverage portion of a module, symmetric about the centre.
func sampleOffsets(k int) []float64 {
	offs := make([]float64, k)
	for t := range k {
		offs[t] = sampleCoverage * ((float64(t)+0.5)/float64(k) - 0.5)
	}
	return offs
}

// sampleGridSize derives the per-axis sample counts from the module extent in
// source pixels, so the footprint is read at roughly pixel density: a large
// module is averaged over its area while a small one keeps a centre
// neighbourhood (at least 3x3, preserving the noise averaging of the previous
// fixed 3x3 kernel without spilling into neighbour modules). The extent comes
// from the warped symbol corners; perspective varies the true per-module
// extent only mildly across a capture, and the density only has to be near
// the pixel pitch, not exact.
func sampleGridSize(pt perspective, side image.Point) (kx, ky int) {
	q00 := pt.warp(pointF{0, 0})
	q10 := pt.warp(pointF{float64(side.X), 0})
	q01 := pt.warp(pointF{0, float64(side.Y)})
	q11 := pt.warp(pointF{float64(side.X), float64(side.Y)})
	modW := (math.Hypot(q10.x-q00.x, q10.y-q00.y) + math.Hypot(q11.x-q01.x, q11.y-q01.y)) / (2 * float64(side.X))
	modH := (math.Hypot(q01.x-q00.x, q01.y-q00.y) + math.Hypot(q11.x-q10.x, q11.y-q10.y)) / (2 * float64(side.Y))
	clamp := func(mod float64) int {
		k := int(mod*sampleCoverage + 0.5)
		if k < 3 {
			return 3
		}
		if k > maxSamplesPerAxis {
			return maxSamplesPerAxis
		}
		return k
	}
	return clamp(modW), clamp(modH)
}

// sampleSymbol samples a side.X by side.Y matrix of module colors from the image
// using the perspective transform. Each module's value is the mean over the
// central portion of its warped footprint, sampled at roughly source-pixel
// density; the grid is derived from the module's extent in source pixels, so
// screen-lattice and sensor noise average out on large modules while small
// modules keep a centre neighbourhood. On a clean flat-colour module the mean
// equals the centre value, so clean decodes stay byte-identical. It returns an
// RGBA bitmap of the sampled module values, or nil if a module centre maps too
// far outside the image.
func sampleSymbol(bm *bitmap, pt perspective, side image.Point) *bitmap {
	// Ports sampleSymbol in sample.c, widening its fixed 3x3 centre average to
	// the scale-derived module-footprint mean.
	out := newBitmap(side.X, side.Y, bm.channels)
	bpp := bm.channels
	bytesPerRow := bm.width * bpp

	kx, ky := sampleGridSize(pt, side)
	offX := sampleOffsets(kx)
	offY := sampleOffsets(ky)
	n := float64(kx * ky)
	sums := make([]float64, bpp)

	for i := range side.Y {
		for j := range side.X {
			cx := float64(j) + 0.5
			cy := float64(i) + 0.5
			// The centre bounds check, with one pixel of tolerance, keeps the
			// ported success/failure semantics: a symbol whose modules map
			// outside the image still fails the sample as a whole.
			p := pt.warp(pointF{cx, cy})
			mx, my := int(p.x), int(p.y)
			if mx < -1 || mx > bm.width || my < -1 || my > bm.height {
				return nil
			}
			for c := range sums {
				sums[c] = 0
			}
			for _, dy := range offY {
				for _, dx := range offX {
					q := pt.warp(pointF{cx + dx, cy + dy})
					px, py := int(q.x), int(q.y)
					if px < 0 {
						px = 0
					} else if px > bm.width-1 {
						px = bm.width - 1
					}
					if py < 0 {
						py = 0
					} else if py > bm.height-1 {
						py = bm.height - 1
					}
					o := py*bytesPerRow + px*bpp
					for c := range sums {
						sums[c] += float64(bm.pix[o+c])
					}
				}
			}
			o := (i*side.X + j) * out.channels
			for c := range sums {
				out.pix[o+c] = byte(sums[c]/n + 0.5)
			}
		}
	}
	return out
}
