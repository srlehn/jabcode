package detect

import (
	"math"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/spec"
)

// pooledCornerNoiseModules is how far a genuine finder centre may sit from an
// exact prediction on a frontal capture. Cross-check refinement moves a centre
// by a fraction of a module, so two modules is slack, not a search.
const pooledCornerNoiseModules = 2.0

// pooledCornerRadius bounds how far from the interpolated corner a real
// candidate may sit and still be that corner.
//
// Three corners determine the fourth exactly under rotation, uniform scale or
// any other affine transform, so on a frontal capture the only slack is centre
// noise; perspective is the whole reason the neighbourhood opens up at all. For
// a keystone whose near modules are s times the far ones, completing the
// parallelogram overshoots the shortened edge by edge*(s-1)/s, and the module
// sizes of the three present corners measure s directly. The search therefore
// follows the capture's own foreshortening rather than a fixed pixel or module
// budget, and collapses to nothing on captures where the construction is exact.
func pooledCornerRadius(fps []FinderPattern, miss int) float64 {
	msMin, msMax := math.Inf(1), 0.0
	for i := range 4 {
		if i != miss {
			msMin = math.Min(msMin, fps[i].ModuleSize)
			msMax = math.Max(msMax, fps[i].ModuleSize)
		}
	}
	if msMin <= 0 || fps[miss].ModuleSize <= 0 {
		return 0
	}
	edge := math.Max(
		dist(fps[miss].Center, fps[(miss+1)&3].Center),
		dist(fps[miss].Center, fps[(miss+3)&3].Center),
	)
	skew := msMax / msMin
	return pooledCornerNoiseModules*fps[miss].ModuleSize + edge*(skew-1)/skew
}

// pickPooledCorner replaces the interpolated corner with one the scan actually
// found, when a candidate is compatible with the partial quad on every side at
// once: it carries the missing type, sits within the prediction's uncertainty,
// agrees on module scale with all three present corners, and completes a quad
// whose side geometry is legal. Selection runs per scan direction while the
// candidate pool is unioned across directions and passes, so the corner one
// direction loses is routinely one another direction found.
//
// The bound has to be two-sided. Cluttered backgrounds produce plenty of
// candidates that passed the entire cross-check chain, so proximity alone would
// trade an exact affine construction for a confident mistake. Finding nothing
// is the expected answer whenever the true corner was never generated, and the
// caller keeps the construction in that case.
func pickPooledCorner(pool, fps []FinderPattern, miss int) (FinderPattern, bool) {
	radius := pooledCornerRadius(fps, miss)
	if radius <= 0 {
		return FinderPattern{}, false
	}
	var quad [4]FinderPattern
	copy(quad[:], fps[:4])
	best, bestScore, found := FinderPattern{}, math.Inf(1), false
	for _, c := range pool {
		// The support floor is the selection's own: a candidate too weakly
		// confirmed to enter a type group is not evidence that can outrank a
		// construction which is exact whenever the capture is affine. Measured
		// on a clean synthetic docking, where a single stray crossing sat about
		// a module from the true corner and displacing the quad onto it cost the
		// read.
		if c.Typ != miss || c.FoundCount < minFinderCrossings || c.ModuleSize <= 0 {
			continue
		}
		off := dist(c.Center, fps[miss].Center)
		if off > radius {
			continue
		}
		scale := 1.0
		for i := range 4 {
			if i != miss {
				scale = math.Max(scale, ratio(c.ModuleSize, fps[i].ModuleSize))
			}
		}
		if scale > quadModuleTol {
			continue
		}
		// The coarse gate, never ScoreFinderQuad: that one divides each edge by
		// the full side size where the finder centres span side-7 modules, so it
		// rejects even an exact small-version quad and would reinstate the
		// construction over a correct detection.
		quad[miss] = c
		if !ConsistentFinderQuad(quad[:]) {
			continue
		}
		if ss := CalculateSideSize(nil, quad[:]); ss.X <= 0 || ss.Y <= 0 {
			continue
		}
		// Distance is normalized by the radius so the two disagreements are
		// comparable, and neither the frame-wide FoundCount nor proximity alone
		// decides: a nearer candidate at the wrong scale loses to a further one
		// the partial quad agrees with.
		if score := off/radius + (scale - 1); score < bestScore {
			best, bestScore, found = c, score, true
		}
	}
	if !found {
		return FinderPattern{}, false
	}
	best.direction = fps[miss].direction
	return best, true
}

// seekMissingFinderPattern searches a local area around the estimated position
// of a single missing finder pattern and, if found, replaces the estimate with
// the detected pattern.
func seekMissingFinderPattern(bm *core.Bitmap, fps []FinderPattern, missIndex int) {
	// Ports seekMissingFinderPattern in detector.c.
	radius := fps[missIndex].ModuleSize * 5
	startX := max(int(fps[missIndex].Center.X-radius), 0)
	startY := max(int(fps[missIndex].Center.Y-radius), 0)
	endX := min(int(fps[missIndex].Center.X+radius), bm.Width-1)
	endY := min(int(fps[missIndex].Center.Y+radius), bm.Height-1)
	areaW := endX - startX
	areaH := endY - startY
	if areaW <= 0 || areaH <= 0 {
		return
	}

	var rgb [3]*core.Bitmap
	for i := range rgb {
		rgb[i] = core.NewBitmap(areaW, areaH, 1)
	}

	bpp := bm.Channels
	bytesPerRow := bm.Width * bpp
	var sum [3]int64
	for i := startY; i < endY; i++ {
		for j := startX; j < endX; j++ {
			off := i*bytesPerRow + j*bpp
			sum[0] += int64(bm.Pix[off+0])
			sum[1] += int64(bm.Pix[off+1])
			sum[2] += int64(bm.Pix[off+2])
		}
	}
	area := int64(areaW) * int64(areaH)

	// Quantize the search area to black, cyan and yellow.
	for i, y := startY, 0; i < endY; i, y = i+1, y+1 {
		for j, x := startX, 0; j < endX; j, x = j+1, x+1 {
			off := i*bytesPerRow + j*bpp
			r, g, b := bm.Pix[off+0], bm.Pix[off+1], bm.Pix[off+2]
			idx := y*areaW + x
			switch {
			case int64(r)*area < sum[0] && int64(g)*area < sum[1] && int64(b)*area < sum[2]: // black
			case r < b: // cyan
				rgb[1].Pix[idx] = 255
				rgb[2].Pix[idx] = 255
			default: // yellow
				rgb[0].Pix[idx] = 255
				rgb[1].Pix[idx] = 255
			}
		}
	}

	var expR, expG, expB int
	switch missIndex {
	case fp2:
		expR, expG, expB = 255, 255, 0
	case fp3:
		expR, expG, expB = 0, 255, 255
	} // fp0/fp1 expect 0,0,0

	fpsMiss := make([]FinderPattern, maxFinderPatterns)
	total := 0
	fpTypeCount := make([]int, 4)
	done := false

	for i := 0; i < areaH && !done; i++ {
		rowR := rgb[0].Pix[i*areaW : (i+1)*areaW]
		rowG := rgb[1].Pix[i*areaW : (i+1)*areaW]
		rowB := rgb[2].Pix[i*areaW : (i+1)*areaW]
		startx, endx, skip := 0, areaW, 0
		for first := true; first || (startx < areaW && endx < areaW); {
			first = false
			startx += skip
			endx = areaW
			ps := seekPatternHorizontal(rowG, startx, endx)
			startx, endx = ps.start, ps.end
			if !ps.ok {
				continue
			}
			skip = ps.skip
			centerxG, moduleSizeG := ps.Center, ps.ModuleSize
			if core.BoolColor(rowG[int(centerxG)] > 0) != expG {
				continue
			}
			centerxR, centerxB := centerxG, centerxG
			var moduleSizeR, moduleSizeB float64
			found := false
			var fp FinderPattern

			switch missIndex {
			case fp0, fp3:
				if crossCheckPatternHorizontal(rgb[2], moduleSizeG*2, &centerxB, float64(i), &moduleSizeB, 3) {
					if core.BoolColor(rowB[int(centerxB)] > 0) != expB {
						continue
					}
					moduleSizeR = moduleSizeG
					if crossCheckColor(rgb[0], int(palette.Default[spec.FP3CoreColor*3+0]), int(moduleSizeR), 5, int(centerxR), i, 0, 3) {
						found = true
					}
				}
				if found {
					if !checkModuleSize2(moduleSizeG, moduleSizeB) {
						continue
					}
					fp.Center.X = (centerxG + centerxB) / 2.0
					fp.ModuleSize = (moduleSizeG + moduleSizeB) / 2.0
				}
			case fp1, fp2:
				if crossCheckPatternHorizontal(rgb[0], moduleSizeG*2, &centerxR, float64(i), &moduleSizeR, 3) {
					if core.BoolColor(rowR[int(centerxR)] > 0) != expR {
						continue
					}
					moduleSizeB = moduleSizeG
					if crossCheckColor(rgb[2], int(palette.Default[spec.FP2CoreColor*3+2]), int(moduleSizeB), 5, int(centerxB), i, 0, 3) {
						found = true
					}
				}
				if found {
					if !checkModuleSize2(moduleSizeR, moduleSizeG) {
						continue
					}
					fp.Center.X = (centerxR + centerxG) / 2.0
					fp.ModuleSize = (moduleSizeR + moduleSizeG) / 2.0
				}
			}

			if found {
				fp.Center.Y = float64(i)
				fp.FoundCount = 1
				fp.Typ = missIndex
				if crossCheckPattern(rgb, &fp, 0, 3) {
					saveFinderPattern(&fp, fpsMiss, &total, fpTypeCount)
					if total >= maxFinderPatterns-1 {
						done = true
						break
					}
				}
			}
		}
	}

	if total > 0 {
		maxFound, maxIdx := 0, 0
		for i := 0; i < total; i++ {
			if fpsMiss[i].FoundCount > maxFound {
				maxFound = fpsMiss[i].FoundCount
				maxIdx = i
			}
		}
		fps[missIndex] = fpsMiss[maxIdx]
		fps[missIndex].Center.X += float64(startX)
		fps[missIndex].Center.Y += float64(startY)
	}
}
