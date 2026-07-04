package detect

import (
	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/spec"
)

// Detection modes. The shared status codes live in the core package.
const (
	quickDetect     = 0
	normalDetect    = 1
	IntensiveDetect = 2

	maxModules        = 145 // modules in side-version 32
	maxSymbolRows     = 3
	maxFinderPatterns = 500
)

// classifyFinderPattern sets fp.typ from the detected core color, returning
// false if the color triple matches no finder-pattern type.
func classifyFinderPattern(fp *FinderPattern, candidates []int, typeR, typeG, typeB int) bool {
	for _, t := range candidates {
		core := fpCoreColorIndex(t)
		if typeR == int(palette.Default[core*3]) && typeG == int(palette.Default[core*3+1]) && typeB == int(palette.Default[core*3+2]) {
			fp.Typ = t
			return true
		}
	}
	return false
}

// fpCoreColorIndex returns the default-palette color index of a finder pattern's
// core.
func fpCoreColorIndex(t int) int {
	switch t {
	case fp0:
		return spec.FP0CoreColor
	case fp1:
		return spec.FP1CoreColor
	case fp2:
		return spec.FP2CoreColor
	default:
		return spec.FP3CoreColor
	}
}

// findPrimarySymbol scans the binarized channels for the four finder patterns of
// the primary symbol, leaving the working list (with the four selected patterns
// in [0:4]) in d.fps and returning a status. It records this pass's counters in
// d.stats.
func (d *PrimaryDetector) findPrimarySymbol() int {
	// Ports findPrimarySymbol in detector.c.
	d.Stats.Passes = append(d.Stats.Passes, FinderPassStats{})
	ch := d.Ch

	minModuleSize := ch[0].Height / (2 * maxSymbolRows * maxModules)
	if minModuleSize < 1 || d.Mode == IntensiveDetect {
		minModuleSize = 1
	}

	fps := make([]FinderPattern, maxFinderPatterns)
	d.FPs = fps
	totalFP := 0
	fpTypeCount := make([]int, 4)
	done := false

	w, h := ch[0].Width, ch[0].Height
	for i := 0; i < h && !done; i += minModuleSize {
		rowR := ch[0].Pix[i*w : (i+1)*w]
		rowG := ch[1].Pix[i*w : (i+1)*w]
		rowB := ch[2].Pix[i*w : (i+1)*w]

		startx, endx, skip := 0, w, 0
		for first := true; first || (startx < w && endx < w); {
			first = false
			startx += skip
			endx = w
			ps := seekPatternHorizontal(rowG, startx, endx)
			startx, endx = ps.start, ps.end
			if !ps.ok {
				continue
			}
			d.pass().RawHits++
			d.seedModules = append(d.seedModules, ps.ModuleSize)
			skip = ps.skip
			centerxG, moduleSizeG := ps.Center, ps.ModuleSize

			typeG := core.BoolColor(rowG[int(centerxG)] > 0)
			centerxR, centerxB := centerxG, centerxG
			var typeR, typeB int
			var moduleSizeR, moduleSizeB float64
			fp1found, fp2found := false, false

			if crossCheckPatternHorizontal(ch[2], moduleSizeG*2, &centerxB, float64(i), &moduleSizeB) {
				d.pass().BranchBlue++
				typeB = core.BoolColor(rowB[int(centerxB)] > 0)
				moduleSizeR = moduleSizeG
				coreRed := int(palette.Default[spec.FP3CoreColor*3+0])
				if crossCheckColor(ch[0], coreRed, int(moduleSizeR), 5, int(centerxR), i, 0) {
					typeR = 0
					fp1found = true
				}
			} else if crossCheckPatternHorizontal(ch[0], moduleSizeG*2, &centerxR, float64(i), &moduleSizeR) {
				d.pass().BranchRed++
				typeR = core.BoolColor(rowR[int(centerxR)] > 0)
				moduleSizeB = moduleSizeG
				coreBlue := int(palette.Default[spec.FP2CoreColor*3+2])
				if crossCheckColor(ch[2], coreBlue, int(moduleSizeB), 5, int(centerxB), i, 0) {
					typeB = 0
					fp2found = true
					d.pass().RedColor++
				}
			}

			if !(fp1found || fp2found) {
				continue
			}
			fp := FinderPattern{Center: core.PointF{Y: float64(i)}, FoundCount: 1}
			if fp1found {
				if !checkModuleSize2(moduleSizeG, moduleSizeB) {
					continue
				}
				fp.Center.X = (centerxG + centerxB) / 2.0
				fp.ModuleSize = (moduleSizeG + moduleSizeB) / 2.0
				if !classifyFinderPattern(&fp, []int{fp0, fp3}, typeR, typeG, typeB) {
					continue
				}
			} else {
				if !checkModuleSize2(moduleSizeR, moduleSizeG) {
					continue
				}
				fp.Center.X = (centerxR + centerxG) / 2.0
				fp.ModuleSize = (moduleSizeR + moduleSizeG) / 2.0
				if !classifyFinderPattern(&fp, []int{fp1, fp2}, typeR, typeG, typeB) {
					continue
				}
				d.pass().RedClassified++
			}
			if crossCheckPattern(ch, &fp, 0) {
				d.pass().CrossSurvivors[fp.Typ]++
				saveFinderPattern(&fp, fps, &totalFP, fpTypeCount)
				if totalFP >= maxFinderPatterns-1 {
					done = true
					break
				}
			}
		}
	}

	// If only FP0+FP1 or only FP2+FP3 were found, also scan vertically.
	if (fpTypeCount[0] != 0 && fpTypeCount[1] != 0 && fpTypeCount[2] == 0 && fpTypeCount[3] == 0) ||
		(fpTypeCount[0] == 0 && fpTypeCount[1] == 0 && fpTypeCount[2] != 0 && fpTypeCount[3] != 0) {
		d.scanPatternVertical(minModuleSize, fps, fpTypeCount, &totalFP)
	}

	d.Candidates = append([]FinderPattern(nil), fps[:totalFP]...)
	d.pass().Candidates = d.Candidates

	for i := 0; i < totalFP; i++ {
		if fps[i].direction >= 0 {
			fps[i].direction = 1
		} else {
			fps[i].direction = -1
		}
	}

	missing := d.selectBestPatterns(fps, totalFP, fpTypeCount)
	if missing > 1 {
		d.pass().Status = core.Failure
		return core.Failure
	}
	if missing == 1 {
		if !estimateMissingPattern(d.BM, d.Ch, fps) {
			d.pass().Status = core.Failure
			return core.Failure
		}
		d.pass().Interpolated = true
	}
	d.pass().Status = core.Success
	return core.Success
}

// estimateMissingPattern interpolates the position of the single missing finder
// pattern from the other three and confirms it by local search. Returns false if
// the estimate falls outside the image (findPrimarySymbol missing-pattern branch).
func estimateMissingPattern(bm *core.Bitmap, ch [3]*core.Bitmap, fps []FinderPattern) bool {
	miss := -1
	switch {
	case fps[0].FoundCount == 0:
		miss = 0
		s23 := (fps[2].ModuleSize + fps[3].ModuleSize) / 2.0
		s13 := (fps[1].ModuleSize + fps[3].ModuleSize) / 2.0
		fps[0].Center.X = (fps[3].Center.X-fps[2].Center.X)/s23*s13 + fps[1].Center.X
		fps[0].Center.Y = (fps[3].Center.Y-fps[2].Center.Y)/s23*s13 + fps[1].Center.Y
		fps[0].Typ, fps[0].FoundCount, fps[0].direction = fp0, 1, -fps[1].direction
		fps[0].ModuleSize = (fps[1].ModuleSize + fps[2].ModuleSize + fps[3].ModuleSize) / 3.0
	case fps[1].FoundCount == 0:
		miss = 1
		s23 := (fps[2].ModuleSize + fps[3].ModuleSize) / 2.0
		s02 := (fps[0].ModuleSize + fps[2].ModuleSize) / 2.0
		fps[1].Center.X = (fps[2].Center.X-fps[3].Center.X)/s23*s02 + fps[0].Center.X
		fps[1].Center.Y = (fps[2].Center.Y-fps[3].Center.Y)/s23*s02 + fps[0].Center.Y
		fps[1].Typ, fps[1].FoundCount, fps[1].direction = fp1, 1, -fps[0].direction
		fps[1].ModuleSize = (fps[0].ModuleSize + fps[2].ModuleSize + fps[3].ModuleSize) / 3.0
	case fps[2].FoundCount == 0:
		miss = 2
		s01 := (fps[0].ModuleSize + fps[1].ModuleSize) / 2.0
		s13 := (fps[1].ModuleSize + fps[3].ModuleSize) / 2.0
		fps[2].Center.X = (fps[1].Center.X-fps[0].Center.X)/s01*s13 + fps[3].Center.X
		fps[2].Center.Y = (fps[1].Center.Y-fps[0].Center.Y)/s01*s13 + fps[3].Center.Y
		fps[2].Typ, fps[2].FoundCount, fps[2].direction = fp2, 1, fps[3].direction
		fps[2].ModuleSize = (fps[0].ModuleSize + fps[1].ModuleSize + fps[3].ModuleSize) / 3.0
	case fps[3].FoundCount == 0:
		miss = 3
		s01 := (fps[0].ModuleSize + fps[1].ModuleSize) / 2.0
		s02 := (fps[0].ModuleSize + fps[2].ModuleSize) / 2.0
		fps[3].Center.X = (fps[0].Center.X-fps[1].Center.X)/s01*s02 + fps[2].Center.X
		fps[3].Center.Y = (fps[0].Center.Y-fps[1].Center.Y)/s01*s02 + fps[2].Center.Y
		fps[3].Typ, fps[3].FoundCount, fps[3].direction = fp3, 1, fps[2].direction
		fps[3].ModuleSize = (fps[0].ModuleSize + fps[1].ModuleSize + fps[2].ModuleSize) / 3.0
	}
	if fps[miss].Center.X < 0 || fps[miss].Center.X > float64(ch[0].Width-1) ||
		fps[miss].Center.Y < 0 || fps[miss].Center.Y > float64(ch[0].Height-1) {
		fps[miss].FoundCount = 0
		return false
	}
	seekMissingFinderPattern(bm, fps, miss)
	return true
}

// scanPatternVertical scans the image column-wise for finder patterns, used when
// only a top or bottom pair was found horizontally (scanPatternVertical). It
// records its hits in the current pass's d.stats.
func (d *PrimaryDetector) scanPatternVertical(minModuleSize int, fps []FinderPattern, fpTypeCount []int, totalFP *int) {
	ch := d.Ch
	w, h := ch[0].Width, ch[0].Height
	done := false
	for j := 0; j < w && !done; j += minModuleSize {
		starty, endy, skip := 0, h, 0
		for first := true; first || (starty < h && endy < h); {
			first = false
			starty += skip
			endy = h
			ps := seekPattern(ch[1], -1, j, starty, endy)
			starty, endy = ps.start, ps.end
			if !ps.ok {
				continue
			}
			d.pass().RawHits++
			d.seedModules = append(d.seedModules, ps.ModuleSize)
			skip = ps.skip
			centeryG, moduleSizeG := ps.Center, ps.ModuleSize

			typeG := core.BoolColor(ch[1].Pix[int(centeryG)*w+j] > 0)
			centeryR, centeryB := centeryG, centeryG
			var typeR, typeB int
			var moduleSizeR, moduleSizeB float64
			fp1found, fp2found := false, false

			if crossCheckPatternVertical(ch[2], int(moduleSizeG*2), float64(j), &centeryB, &moduleSizeB) {
				typeB = core.BoolColor(ch[2].Pix[int(centeryB)*w+j] > 0)
				moduleSizeR = moduleSizeG
				coreRed := int(palette.Default[spec.FP3CoreColor*3+0])
				if crossCheckColor(ch[0], coreRed, int(moduleSizeR), 5, j, int(centeryR), 1) {
					typeR = 0
					fp1found = true
				}
			} else if crossCheckPatternVertical(ch[0], int(moduleSizeG*2), float64(j), &centeryR, &moduleSizeR) {
				typeR = core.BoolColor(ch[0].Pix[int(centeryR)*w+j] > 0)
				moduleSizeB = moduleSizeG
				coreBlue := int(palette.Default[spec.FP2CoreColor*3+2])
				if crossCheckColor(ch[2], coreBlue, int(moduleSizeB), 5, j, int(centeryB), 1) {
					typeB = 0
					fp2found = true
				}
			}

			if !(fp1found || fp2found) {
				continue
			}
			fp := FinderPattern{Center: core.PointF{X: float64(j)}, FoundCount: 1}
			if fp1found {
				if !checkModuleSize2(moduleSizeG, moduleSizeB) {
					continue
				}
				fp.Center.Y = (centeryG + centeryB) / 2.0
				fp.ModuleSize = (moduleSizeG + moduleSizeB) / 2.0
				if !classifyFinderPattern(&fp, []int{fp0, fp3}, typeR, typeG, typeB) {
					continue
				}
			} else {
				if !checkModuleSize2(moduleSizeR, moduleSizeG) {
					continue
				}
				fp.Center.Y = (centeryR + centeryG) / 2.0
				fp.ModuleSize = (moduleSizeR + moduleSizeG) / 2.0
				if !classifyFinderPattern(&fp, []int{fp1, fp2}, typeR, typeG, typeB) {
					continue
				}
			}
			if crossCheckPattern(ch, &fp, 1) {
				d.pass().CrossSurvivors[fp.Typ]++
				saveFinderPattern(&fp, fps, totalFP, fpTypeCount)
				if *totalFP >= maxFinderPatterns-1 {
					done = true
					break
				}
			}
		}
	}
}
