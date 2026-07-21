package detect

import (
	"bytes"
	"image"
	"math/rand/v2"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
)

// sampleSymbolFootprintDirect is the pre-hoist footprint sampler: every point
// is warped in full and the tent weight is remultiplied per module. It is the
// reference the y-hoisted, weight-tabulated form must reproduce byte for byte.
func sampleSymbolFootprintDirect(bm *core.Bitmap, pt core.Perspective, side image.Point, modW, modH float64, delta [3]core.PointF) *core.Bitmap {
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

// TestSampleSymbolFootprintMatchesDirectForm pins the y-hoisted, weight-tabulated
// footprint sampler to the direct form it replaces. Sampling feeds LDPC, which
// has no payload integrity check, so a single differing byte could turn into a
// silently corrupted decode; byte identity here is what lets the rewrite skip
// the full sampling harness matrix. Cases span the footprint regime from just
// above the legacy cutoff to a dense grid, a skewed perspective, both channel
// counts, and both the nominal and offset sampling paths.
func TestSampleSymbolFootprintMatchesDirectForm(t *testing.T) {
	rng := rand.New(rand.NewPCG(0x5a3f, 0x0007))
	// A span wide enough that side modules exceed legacySampleBelowPx (9 px):
	// the quad corners sit inside a 480x480 frame with a deliberate skew so the
	// perspective denominator actually varies across the grid.
	quads := [][4]core.PointF{
		{{X: 40, Y: 40}, {X: 440, Y: 44}, {X: 446, Y: 452}, {X: 36, Y: 448}}, // near-rectangular
		{{X: 60, Y: 30}, {X: 430, Y: 70}, {X: 450, Y: 470}, {X: 20, Y: 430}}, // skewed
		{{X: 35, Y: 55}, {X: 455, Y: 35}, {X: 435, Y: 455}, {X: 55, Y: 445}}, // rotated-ish
	}
	sides := []image.Point{{X: 21, Y: 21}, {X: 15, Y: 23}, {X: 33, Y: 33}}
	deltas := [][3]core.PointF{
		{},
		{{X: 0.7, Y: -0.4}, {X: -0.5, Y: 0.6}, {X: 0.2, Y: 0.3}},
	}

	frame := core.NewBitmap(480, 480, 4)
	for i := range frame.Pix {
		frame.Pix[i] = byte(rng.Uint32N(256))
	}

	for qi, quad := range quads {
		for _, side := range sides {
			pt := core.PerspectiveTransform(quad[0], quad[1], quad[2], quad[3], side)
			modW, modH := moduleExtent(pt, side)
			if min(modW, modH) < legacySampleBelowPx {
				t.Fatalf("quad %d side %v: module extent %.2fx%.2f below the footprint regime", qi, side, modW, modH)
			}
			for di, delta := range deltas {
				got := sampleSymbolFootprint(frame, pt, side, modW, modH, delta)
				want := sampleSymbolFootprintDirect(frame, pt, side, modW, modH, delta)
				if (got == nil) != (want == nil) {
					t.Fatalf("quad %d side %v delta %d: nil mismatch got=%v want=%v", qi, side, di, got == nil, want == nil)
				}
				if got != nil && !bytes.Equal(got.Pix, want.Pix) {
					t.Fatalf("quad %d side %v delta %d: hoisted sampler differs from the direct form", qi, side, di)
				}
			}
		}
	}
}
