package detect

import (
	"image"
	"math"
	"slices"

	"github.com/srlehn/jabcode/internal/core"
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

	// Channel-offset search: a channel whose sampled deciles span less than
	// minBimodalRange has no usable modes, and an offset is only adopted when
	// it beats the nominal position's score by channelOffsetMinGain - both
	// guard the search against chasing noise on undamaged rows.
	minBimodalRange      = 32.0
	channelOffsetMinGain = 0.25
)

// channelOffsetGrid is the candidate offset grid of the per-channel search,
// in module units per axis (plane misregistration beyond half a module makes
// the overlay unreadable anyway, so the grid stops there).
var channelOffsetGrid = []float64{-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4}

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
func moduleExtent(pt core.Perspective, side image.Point) (modW, modH float64) {
	q00 := pt.Warp(core.Pt(0, 0))
	q10 := pt.Warp(core.Pt(float64(side.X), 0))
	q01 := pt.Warp(core.Pt(0, float64(side.Y)))
	q11 := pt.Warp(core.Pt(float64(side.X), float64(side.Y)))
	modW = (math.Hypot(q10.X-q00.X, q10.Y-q00.Y) + math.Hypot(q11.X-q01.X, q11.Y-q01.Y)) / (2 * float64(side.X))
	modH = (math.Hypot(q01.X-q00.X, q01.Y-q00.Y) + math.Hypot(q11.X-q10.X, q11.Y-q10.Y)) / (2 * float64(side.Y))
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

// SampleSymbol samples a side.X by side.Y matrix of module colors from the
// image using the perspective transform. Small-module symbols use the ported
// 3x3 centre kernel; larger modules use the tent-weighted footprint mean (see
// the geometry constants above). On a clean flat-colour module both equal the
// centre value, so clean decodes stay byte-identical. It returns an RGBA
// bitmap of the sampled module values, or nil if a module maps too far
// outside the image.
func SampleSymbol(bm *core.Bitmap, pt core.Perspective, side image.Point) *core.Bitmap {
	return SampleSymbolOffset(bm, pt, side, [3]core.PointF{})
}

// SampleSymbolOffset is SampleSymbol with a per-channel source-pixel offset
// added to every sampling position: channel c of each module is read at the
// warped point plus delta[c]. Misregistered colorant planes displace each
// channel's content from the nominal grid, so sampling each channel where its
// plane actually landed recovers the module colour the overlay destroyed.
// Zero deltas reproduce SampleSymbol exactly.
func SampleSymbolOffset(bm *core.Bitmap, pt core.Perspective, side image.Point, delta [3]core.PointF) *core.Bitmap {
	modW, modH := moduleExtent(pt, side)
	if min(modW, modH) < legacySampleBelowPx {
		return sampleSymbolCentre(bm, pt, side, delta)
	}
	return sampleSymbolFootprint(bm, pt, side, modW, modH, delta)
}

// chanDelta returns the sampling offset for channel c; channels past the
// colour planes (alpha) stay unshifted.
func chanDelta(delta [3]core.PointF, c int) core.PointF {
	if c < 3 {
		return delta[c]
	}
	return core.PointF{}
}

// SearchChannelOffsets searches, independently per colour channel, for the
// source-pixel sampling offset that makes the channel's module values most
// bimodal, over a grid of candidate offsets spanning half a module. A module's
// channel value is ideally one of two levels (ink or no ink in that plane);
// colorant-plane misregistration slides the plane off the module grid, so the
// centre samples mix neighbours and collapse toward mid-range. The offset that
// restores bimodality is where the plane actually landed. Zero is in the
// candidate grid and wins ties, so channels without real misregistration keep
// their nominal positions and the result degrades to plain sampling.
func SearchChannelOffsets(bm *core.Bitmap, pt core.Perspective, side image.Point) [3]core.PointF {
	modW, modH := moduleExtent(pt, side)
	if min(modW, modH) < legacySampleBelowPx {
		// Below the footprint regime the grid steps quantize to a pixel or
		// two and the scoring footprint shrinks to dot scale, where halftone
		// structure games the objective; measured on the 6 px harness rows.
		return [3]core.PointF{}
	}

	// Module-centre warp positions over a stride-2 subset, computed once.
	type spot struct{ x, y float64 }
	var spots []spot
	for i := 0; i < side.Y; i += 2 {
		for j := 0; j < side.X; j += 2 {
			p := pt.Warp(core.Pt(float64(j)+0.5, float64(i)+0.5))
			spots = append(spots, spot{p.X, p.Y})
		}
	}
	if len(spots) < 16 {
		return [3]core.PointF{}
	}

	bpp := bm.Channels
	bytesPerRow := bm.Width * bpp
	vals := make([]float64, len(spots))
	sorted := make([]float64, len(spots))

	// Each module value is a small footprint average, like the real sampler:
	// on a halftoned print a single pixel is ink or paper and thus bimodal at
	// dot scale no matter where it lands, which a point-sampled score would
	// chase instead of the plane displacement.
	foot := []struct{ fx, fy float64 }{
		{0, 0}, {-0.2, 0}, {0.2, 0}, {0, -0.2}, {0, 0.2},
		{-0.2, -0.2}, {0.2, -0.2}, {-0.2, 0.2}, {0.2, 0.2},
	}

	// score is the mean distance of every module's channel value to the
	// nearer of the channel's low/high deciles: small when bimodal. It is
	// evaluated over one of two interleaved module subsets (parity), so a
	// candidate offset must win on both independently.
	score := func(c int, dx, dy float64, parity int) float64 {
		n := 0
		for k, s := range spots {
			if k%2 != parity {
				continue
			}
			sum := 0.0
			for _, f := range foot {
				px := min(max(int(s.x+dx+f.fx*modW), 0), bm.Width-1)
				py := min(max(int(s.y+dy+f.fy*modH), 0), bm.Height-1)
				sum += float64(bm.Pix[py*bytesPerRow+px*bpp+c])
			}
			vals[n] = sum / float64(len(foot))
			n++
		}
		copy(sorted[:n], vals[:n])
		slices.Sort(sorted[:n])
		lo := sorted[n/10]
		hi := sorted[n-1-n/10]
		if hi-lo < minBimodalRange {
			return math.Inf(1) // a flat channel has no modes to sharpen
		}
		sum := 0.0
		for _, v := range vals[:n] {
			sum += min(math.Abs(v-lo), math.Abs(v-hi))
		}
		return sum / float64(n)
	}

	var delta [3]core.PointF
	for c := range 3 {
		var bx, by [2]float64
		agree := true
		for parity := range 2 {
			base := score(c, 0, 0, parity)
			best := base
			for _, fy := range channelOffsetGrid {
				for _, fx := range channelOffsetGrid {
					if fx == 0 && fy == 0 {
						continue
					}
					dx, dy := fx*modW, fy*modH
					// Demand a clear win over the nominal position: on rows
					// whose damage is not misregistration the search
					// otherwise chases sampling noise.
					if s := score(c, dx, dy, parity); s < best && s < base*(1-channelOffsetMinGain) {
						best, bx[parity], by[parity] = s, dx, dy
					}
				}
			}
		}
		// A real plane displacement wins on both halves at neighbouring grid
		// cells; noise-driven winners scatter.
		if math.Abs(bx[0]-bx[1]) > modW*0.15 || math.Abs(by[0]-by[1]) > modH*0.15 {
			agree = false
		}
		if agree {
			delta[c] = core.PointF{X: (bx[0] + bx[1]) / 2, Y: (by[0] + by[1]) / 2}
		}
	}
	return delta
}

// sampleSymbolFootprint is the whole-module path: each module's value is the
// tent-weighted mean over the central portion of its warped footprint,
// sampled at roughly source-pixel density, so screen-lattice and sensor noise
// average out.
func sampleSymbolFootprint(bm *core.Bitmap, pt core.Perspective, side image.Point, modW, modH float64, delta [3]core.PointF) *core.Bitmap {
	out := core.NewBitmap(side.X, side.Y, bm.Channels)
	bpp := bm.Channels
	bytesPerRow := bm.Width * bpp

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
			p := pt.Warp(core.Pt(cx, cy))
			mx, my := int(p.X), int(p.Y)
			if mx < -1 || mx > bm.Width || my < -1 || my > bm.Height {
				return nil
			}
			for c := range sums {
				sums[c] = 0
			}
			for yi, dy := range offY {
				for xi, dx := range offX {
					q := pt.Warp(core.Pt(cx+dx, cy+dy))
					weight := wY[yi] * wX[xi]
					if delta == ([3]core.PointF{}) {
						px, py := int(q.X), int(q.Y)
						if px < 0 {
							px = 0
						} else if px > bm.Width-1 {
							px = bm.Width - 1
						}
						if py < 0 {
							py = 0
						} else if py > bm.Height-1 {
							py = bm.Height - 1
						}
						o := py*bytesPerRow + px*bpp
						for c := range sums {
							sums[c] += weight * float64(bm.Pix[o+c])
						}
						continue
					}
					for c := range sums {
						d := chanDelta(delta, c)
						px, py := int(q.X+d.X), int(q.Y+d.Y)
						if px < 0 {
							px = 0
						} else if px > bm.Width-1 {
							px = bm.Width - 1
						}
						if py < 0 {
							py = 0
						} else if py > bm.Height-1 {
							py = bm.Height - 1
						}
						sums[c] += weight * float64(bm.Pix[py*bytesPerRow+px*bpp+c])
					}
				}
			}
			o := (i*side.X + j) * out.Channels
			for c := range sums {
				out.Pix[o+c] = byte(sums[c]/n + 0.5)
			}
		}
	}
	return out
}

// sampleSymbolCentre is the small-module path: a 3x3 neighborhood average at
// each module centre. Its contiguous uniform average over integer pixels is
// the better estimator when the module barely exceeds the kernel.
func sampleSymbolCentre(bm *core.Bitmap, pt core.Perspective, side image.Point, delta [3]core.PointF) *core.Bitmap {
	// Ports sampleSymbol in sample.c.
	out := core.NewBitmap(side.X, side.Y, bm.Channels)
	bpp := bm.Channels
	bytesPerRow := bm.Width * bpp

	points := make([]core.PointF, side.X)
	for i := 0; i < side.Y; i++ {
		for j := 0; j < side.X; j++ {
			points[j] = pt.Warp(core.Pt(float64(j)+0.5, float64(i)+0.5))
		}
		for j := 0; j < side.X; j++ {
			mx := int(points[j].X)
			my := int(points[j].Y)
			if mx < 0 || mx > bm.Width-1 {
				switch mx {
				case -1:
					mx = 0
				case bm.Width:
					mx = bm.Width - 1
				default:
					return nil
				}
			}
			if my < 0 || my > bm.Height-1 {
				switch my {
				case -1:
					my = 0
				case bm.Height:
					my = bm.Height - 1
				default:
					return nil
				}
			}
			for c := 0; c < out.Channels; c++ {
				// The channel offset shifts only the kernel centre; the ported
				// bounds semantics above stay tied to the unshifted position.
				d := chanDelta(delta, c)
				cmx := min(max(mx+int(math.Round(d.X)), 0), bm.Width-1)
				cmy := min(max(my+int(math.Round(d.Y)), 0), bm.Height-1)
				sum := 0.0
				for dx := -1; dx <= 1; dx++ {
					for dy := -1; dy <= 1; dy++ {
						px, py := cmx+dx, cmy+dy
						if px < 0 || px > bm.Width-1 {
							px = cmx
						}
						if py < 0 || py > bm.Height-1 {
							py = cmy
						}
						sum += float64(bm.Pix[py*bytesPerRow+px*bpp+c])
					}
				}
				out.Pix[(i*side.X+j)*out.Channels+c] = byte(sum/9.0 + 0.5)
			}
		}
	}
	return out
}
