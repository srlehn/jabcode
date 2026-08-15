package detect

import (
	"image"
	"math"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/tables"
)

// apGroundTruth returns the true image-space centres of the interior alignment
// patterns, derived from the encoder's own placement rule rather than from
// either locator.
//
// This is the measurement both locators have to be judged against. Comparing
// them to each other only says which one the other disagrees with, and the row
// walk is not an oracle: it differs from the directional walk in its
// colour-walk arithmetic, its diagonal lengths and its candidate enumeration,
// so "the row walk says 58.5" is not evidence that 58.5 is right.
//
// placeAlignmentPatterns writes the core of the pattern at grid cell (x, y) to
// matrix module (APPos[vx][x]-1, APPos[vy][y]-1), and module (mx, my) has its
// centre at module coordinate (mx+0.5, my+0.5).
func apGroundTruth(r encode.Rendered, fwd core.Perspective) []core.PointF {
	vx := spec.SizeToVersion(r.SideSize.X) - 1
	vy := spec.SizeToVersion(r.SideSize.Y) - 1
	nx, ny := tables.APNum[vx], tables.APNum[vy]
	var out []core.PointF
	for y := range ny {
		for x := range nx {
			if (x == 0 || x == nx-1) && (y == 0 || y == ny-1) {
				continue // corners carry finders, not alignment patterns
			}
			out = append(out, fwd.Warp(core.Pt(
				float64(tables.APPos[vx][x]-1)+0.5,
				float64(tables.APPos[vy][y]-1)+0.5)))
		}
	}
	return out
}

// TestAPLocatorCentreAccuracy measures both alignment-pattern locators against
// the encoder's ground truth at the module scale where the two disagree.
//
// It exists because four attempts to recover two low-resolution version rows by
// making the directional locator mechanically closer to the row walk all came
// out neutral or negative. The question it answers is the prior one: which
// locator is right, and by how much.
func TestAPLocatorCentreAccuracy(t *testing.T) {
	// The two versions whose 3px rows regress, plus one that does not.
	for _, v := range []image.Point{{X: 24, Y: 24}, {X: 8, Y: 32}, {X: 12, Y: 12}} {
		r, err := encode.Render(encode.Config{
			Colors: 8, ModuleSize: 1, SymbolNumber: 1,
			SymbolVersions: []image.Point{v},
		}, []byte("alignment pattern centre accuracy at low module scale"))
		if err != nil {
			t.Fatalf("encode %v: %v", v, err)
		}

		for _, modulePx := range []float64{3, 12} {
			img, fwd := renderRotatedRGBA(r, modulePx, 0)
			bm := core.BitmapFromImage(img)
			BalanceRGB(bm)
			ch := BinarizerRGB(bm, nil)

			truth := apGroundTruth(r, fwd)
			if len(truth) == 0 {
				t.Fatalf("module %g: fixture has no interior alignment patterns", modulePx)
			}

			// Sweep the seed error. The real caller seeds from an extrapolated
			// grid position, which on a low-resolution symbol drifts by more
			// than a module, so accuracy from a near-perfect seed says little.
			for _, offModules := range []float64{0.5, 1, 1.5, 2, 3} {
				var rowsErr, basisErr, rowsMS, basisMS float64
				var rowsMiss, basisMiss int
				for _, want := range truth {
					seedX := want.X + offModules*modulePx
					seedY := want.Y + offModules*modulePx

					got := findAlignmentPatternRows(ch, seedX, seedY, modulePx, apx)
					if got.FoundCount == 0 {
						rowsMiss++
					} else {
						rowsErr += math.Hypot(got.Center.X-want.X, got.Center.Y-want.Y)
						rowsMS += got.ModuleSize
					}

					got = findAlignmentPatternBasis(ch, seedX, seedY, modulePx, apx, uprightAPBasis())
					if got.FoundCount == 0 {
						basisMiss++
					} else {
						basisErr += math.Hypot(got.Center.X-want.X, got.Center.Y-want.Y)
						basisMS += got.ModuleSize
					}
				}
				n := float64(len(truth))
				t.Logf("v%dx%d %gpx n=%d seedOff=%.1f  rows: miss=%d err=%.3f ms=%.3f  basis: miss=%d err=%.3f ms=%.3f",
					v.X, v.Y, modulePx, len(truth), offModules,
					rowsMiss, rowsErr/n, rowsMS/n, basisMiss, basisErr/n, basisMS/n)

				// The directional locator must find every pattern and land well
				// inside a module. It measures about a tenth of a pixel; the
				// bound is a tenth of a module, tight enough to catch the two
				// defects that produced this test - a refinement measured from
				// the fractional seed, and a walk overwriting the coordinate a
				// previous walk refined - each worth half a pixel or more at
				// three pixels per module.
				if basisMiss != 0 {
					t.Errorf("v%dx%d %gpx seedOff=%.1f: directional missed %d of %d",
						v.X, v.Y, modulePx, offModules, basisMiss, len(truth))
				}
				if e := basisErr / n; e > modulePx/10 {
					t.Errorf("v%dx%d %gpx seedOff=%.1f: directional centre error %.3fpx, want <= %.3fpx",
						v.X, v.Y, modulePx, offModules, e, modulePx/10)
				}
				if ms := basisMS / n; math.Abs(ms-modulePx) > modulePx/10 {
					t.Errorf("v%dx%d %gpx seedOff=%.1f: directional module size %.3f, want %.3f",
						v.X, v.Y, modulePx, offModules, ms, modulePx)
				}
			}
		}
	}
}
