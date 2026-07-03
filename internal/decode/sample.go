package decode

import (
	"image"
	"math"
)

// Module-footprint sampling geometry: when modules are large enough, each is
// averaged over the central sampleCoverage portion of its warped footprint,
// on a grid dense enough to visit roughly every source pixel there, bounded
// by maxSamplesPerAxis. The coverage balances two measured failure modes: a
// wider window averages more periods of any screen lattice, shrinking the
// phase-dependent brightness swing of the module mean, but its edge samples
// are the ones blur, resampling smear and JPEG chroma bleed contaminate with
// neighbour-module colour - wide enough windows made the best-effort LDPC
// emit silently corrupted payloads.
//
// Below legacySampleBelowPx of module extent the footprint cannot cover
// meaningfully more than the ported fixed 3x3 kernel, and the kernel's
// contiguous uniform average measured strictly better there (footprint
// variants at every coverage corrupted a small-module multi-symbol JPEG
// capture that the 3x3 kernel reads cleanly), so such symbols keep the
// legacy kernel.
const (
	sampleCoverage      = 0.7
	legacySampleBelowPx = 9.0
	maxSamplesPerAxis   = 32
)

// sampleOffsets returns k offsets in module units spanning the central
// sampleCoverage portion of a module, symmetric about the centre, with a
// triangular weight per offset: full at the module centre, approaching zero
// at the footprint edge. The taper serves both failure modes at once -
// samples near the edge are the ones a blur or resampling smear contaminates
// with neighbour-module colour, and a tapered window suppresses the periodic
// brightness ripple a screen lattice leaves in the module mean far better
// than a rectangular window of the same width.
func sampleOffsets(k int) (offs, weights []float64) {
	offs = make([]float64, k)
	weights = make([]float64, k)
	for t := range k {
		u := (float64(t)+0.5)/float64(k) - 0.5
		offs[t] = sampleCoverage * u
		weights[t] = 1 - 2*math.Abs(u)
	}
	return offs, weights
}

// moduleExtent estimates the module's source-pixel extent per axis from the
// warped symbol corners. Perspective varies the true per-module extent only
// mildly across a capture, and the uses (kernel regime, sample density) only
// need it approximately.
func moduleExtent(pt perspective, side image.Point) (modW, modH float64) {
	q00 := pt.warp(pointF{0, 0})
	q10 := pt.warp(pointF{float64(side.X), 0})
	q01 := pt.warp(pointF{0, float64(side.Y)})
	q11 := pt.warp(pointF{float64(side.X), float64(side.Y)})
	modW = (math.Hypot(q10.x-q00.x, q10.y-q00.y) + math.Hypot(q11.x-q01.x, q11.y-q01.y)) / (2 * float64(side.X))
	modH = (math.Hypot(q01.x-q00.x, q01.y-q00.y) + math.Hypot(q11.x-q10.x, q11.y-q10.y)) / (2 * float64(side.Y))
	return modW, modH
}

// sampleGridSize derives the per-axis sample counts from the module extent,
// so the footprint is read at roughly source-pixel density (at least 3x3,
// capped by maxSamplesPerAxis).
func sampleGridSize(modW, modH float64) (kx, ky int) {
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

// sampleSymbol samples a side.X by side.Y matrix of module colors from the
// image using the perspective transform. Small-module symbols use the ported
// 3x3 centre kernel; larger modules use the tent-weighted footprint mean (see
// the geometry constants above). On a clean flat-colour module both equal the
// centre value, so clean decodes stay byte-identical. It returns an RGBA
// bitmap of the sampled module values, or nil if a module maps too far
// outside the image.
func sampleSymbol(bm *bitmap, pt perspective, side image.Point) *bitmap {
	modW, modH := moduleExtent(pt, side)
	if min(modW, modH) < legacySampleBelowPx {
		return sampleSymbolCentre(bm, pt, side)
	}
	return sampleSymbolFootprint(bm, pt, side, modW, modH)
}

// sampleSymbolFootprint is the whole-module path: each module's value is the
// tent-weighted mean over the central portion of its warped footprint,
// sampled at roughly source-pixel density, so screen-lattice and sensor noise
// average out.
func sampleSymbolFootprint(bm *bitmap, pt perspective, side image.Point, modW, modH float64) *bitmap {
	out := newBitmap(side.X, side.Y, bm.channels)
	bpp := bm.channels
	bytesPerRow := bm.width * bpp

	kx, ky := sampleGridSize(modW, modH)
	offX, wX := sampleOffsets(kx)
	offY, wY := sampleOffsets(ky)
	var n float64
	for _, wy := range wY {
		for _, wx := range wX {
			n += wy * wx
		}
	}
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
			for yi, dy := range offY {
				for xi, dx := range offX {
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
					weight := wY[yi] * wX[xi]
					for c := range sums {
						sums[c] += weight * float64(bm.pix[o+c])
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

// sampleSymbolCentre is the small-module path: a 3x3 neighborhood average at
// each module centre. Its contiguous uniform average over integer pixels is
// the better estimator when the module barely exceeds the kernel.
func sampleSymbolCentre(bm *bitmap, pt perspective, side image.Point) *bitmap {
	// Ports sampleSymbol in sample.c.
	out := newBitmap(side.X, side.Y, bm.channels)
	bpp := bm.channels
	bytesPerRow := bm.width * bpp

	points := make([]pointF, side.X)
	for i := 0; i < side.Y; i++ {
		for j := 0; j < side.X; j++ {
			points[j] = pt.warp(pointF{float64(j) + 0.5, float64(i) + 0.5})
		}
		for j := 0; j < side.X; j++ {
			mx := int(points[j].x)
			my := int(points[j].y)
			if mx < 0 || mx > bm.width-1 {
				switch mx {
				case -1:
					mx = 0
				case bm.width:
					mx = bm.width - 1
				default:
					return nil
				}
			}
			if my < 0 || my > bm.height-1 {
				switch my {
				case -1:
					my = 0
				case bm.height:
					my = bm.height - 1
				default:
					return nil
				}
			}
			for c := 0; c < out.channels; c++ {
				sum := 0.0
				for dx := -1; dx <= 1; dx++ {
					for dy := -1; dy <= 1; dy++ {
						px, py := mx+dx, my+dy
						if px < 0 || px > bm.width-1 {
							px = mx
						}
						if py < 0 || py > bm.height-1 {
							py = my
						}
						sum += float64(bm.pix[py*bytesPerRow+px*bpp+c])
					}
				}
				out.pix[(i*side.X+j)*out.channels+c] = byte(sum/9.0 + 0.5)
			}
		}
	}
	return out
}
