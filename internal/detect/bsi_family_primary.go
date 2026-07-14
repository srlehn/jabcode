//go:build jabcode_bsi || jabcode_legacy

package detect

import (
	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/palette"
)

var bsiFamilyFinderCoreColors = [4]int{1, 2, 5, 6}

// LocateBSIFamilyFinders locates the primary finder set defined by
// BSI TR-03137 and retained by pre-v2.0 releases of the C reference
// implementation. It is available only in builds that enable one of those
// wire variants.
func (d *PrimaryDetector) LocateBSIFamilyFinders() bool {
	d.seedModules = d.seedModules[:0]
	d.printDetected = false
	if d.Trace != nil {
		d.Trace.PassInputs = d.Trace.PassInputs[:0]
		d.Trace.PassChannels = d.Trace.PassChannels[:0]
	}
	if d.quitting() {
		return false
	}
	status := d.findPrimarySymbolBSIFamily()
	d.pass().Label = "BSI-family JAB Code raw"
	d.recordTracePass(d.BM)
	return status == core.Success
}

func (fp *FinderPattern) classifyBSIFamily(r, g, b int) bool {
	for typ, colorIndex := range bsiFamilyFinderCoreColors {
		off := colorIndex * 3
		if r == int(palette.Default[off]) &&
			g == int(palette.Default[off+1]) &&
			b == int(palette.Default[off+2]) {
			fp.Typ = typ
			return true
		}
	}
	return false
}

func crossCheckPatternBSIFamily(ch [3]*core.Bitmap, fp *FinderPattern, hv int) bool {
	moduleSizeMax := fp.ModuleSize * 2
	var moduleSize [3]float64
	var centerX, centerY [3]float64
	var direction, diagonal [3]int
	for c := range 3 {
		centerX[c], centerY[c] = fp.Center.X, fp.Center.Y
		if !crossCheckPatternCh(ch[c], fp.Typ, hv, moduleSizeMax,
			&moduleSize[c], &centerX[c], &centerY[c],
			&direction[c], &diagonal[c], 3) {
			return false
		}
	}
	if !checkModuleSize3(moduleSize[0], moduleSize[1], moduleSize[2]) {
		return false
	}
	fp.ModuleSize = (moduleSize[0] + moduleSize[1] + moduleSize[2]) / 3
	fp.Center.X = (centerX[0] + centerX[1] + centerX[2]) / 3
	fp.Center.Y = (centerY[0] + centerY[1] + centerY[2]) / 3
	if diagonal[0] == 2 || diagonal[1] == 2 || diagonal[2] == 2 {
		fp.direction = 2
	} else if direction[0]+direction[1]+direction[2] > 0 {
		fp.direction = 1
	} else {
		fp.direction = -1
	}
	return true
}

func (d *PrimaryDetector) findPrimarySymbolBSIFamily() int {
	d.Stats.Passes = append(d.Stats.Passes, FinderPassStats{})
	ch := d.Ch
	minModuleSize := ch[0].Height / (2 * maxSymbolRows * maxModules)
	if minModuleSize < 1 || d.Mode == IntensiveDetect {
		minModuleSize = 1
	}

	fps := make([]FinderPattern, maxFinderPatterns)
	d.FPs = fps
	total := 0
	typeCount := make([]int, 4)

	w, h := ch[0].Width, ch[0].Height
	for y := 0; y < h && total < maxFinderPatterns-1; y += minModuleSize {
		row := [3][]byte{
			ch[0].Pix[y*w : (y+1)*w],
			ch[1].Pix[y*w : (y+1)*w],
			ch[2].Pix[y*w : (y+1)*w],
		}
		start, end, skip := 0, w, 0
		for first := true; first || (start < w && end < w); {
			first = false
			start += skip
			end = w
			red := seekPatternHorizontal(row[0], start, end)
			start, end = red.start, red.end
			if !red.ok {
				continue
			}
			d.pass().RawHits++
			d.seedModules = append(d.seedModules, red.ModuleSize)
			skip = red.skip

			center := [3]float64{red.Center, red.Center, red.Center}
			moduleSize := [3]float64{red.ModuleSize}
			if !crossCheckPatternHorizontal(ch[1], red.ModuleSize*2,
				&center[1], float64(y), &moduleSize[1], 3) ||
				!crossCheckPatternHorizontal(ch[2], red.ModuleSize*2,
					&center[2], float64(y), &moduleSize[2], 3) ||
				!checkModuleSize3(moduleSize[0], moduleSize[1], moduleSize[2]) {
				continue
			}

			fp := FinderPattern{
				Center: core.PointF{
					X: (center[0] + center[1] + center[2]) / 3,
					Y: float64(y),
				},
				ModuleSize: (moduleSize[0] + moduleSize[1] + moduleSize[2]) / 3,
				FoundCount: 1,
			}
			if !fp.classifyBSIFamily(
				core.BoolColor(row[0][int(center[0])] > 0),
				core.BoolColor(row[1][int(center[1])] > 0),
				core.BoolColor(row[2][int(center[2])] > 0),
			) || !crossCheckPatternBSIFamily(ch, &fp, 0) {
				continue
			}
			d.pass().CrossSurvivors[fp.Typ]++
			saveFinderPattern(&fp, fps, &total, typeCount)
		}
	}

	if (typeCount[0] != 0 && typeCount[1] != 0 && typeCount[2] == 0 && typeCount[3] == 0) ||
		(typeCount[0] == 0 && typeCount[1] == 0 && typeCount[2] != 0 && typeCount[3] != 0) {
		d.scanPatternVerticalBSIFamily(minModuleSize, fps, typeCount, &total)
	}

	d.Candidates = append([]FinderPattern(nil), fps[:total]...)
	d.pass().Candidates = d.Candidates
	for i := range total {
		if fps[i].direction >= 0 {
			fps[i].direction = 1
		} else {
			fps[i].direction = -1
		}
	}
	missing := d.selectBestPatterns(fps, total, typeCount)
	if missing > 1 || (missing == 1 && !estimateMissingBSIFamily(fps, ch[0].Width, ch[0].Height)) {
		d.pass().Status = core.Failure
		return core.Failure
	}
	if missing == 1 {
		d.pass().Interpolated = true
	}
	d.pass().Status = core.Success
	return core.Success
}

func (d *PrimaryDetector) scanPatternVerticalBSIFamily(minModuleSize int, fps []FinderPattern, typeCount []int, total *int) {
	ch := d.Ch
	w, h := ch[0].Width, ch[0].Height
	for x := 0; x < w && *total < maxFinderPatterns-1; x += minModuleSize {
		start, end, skip := 0, h, 0
		for first := true; first || (start < h && end < h); {
			first = false
			start += skip
			end = h
			red := seekPattern(ch[0], -1, x, start, end)
			start, end = red.start, red.end
			if !red.ok {
				continue
			}
			d.pass().RawHits++
			d.seedModules = append(d.seedModules, red.ModuleSize)
			skip = red.skip

			center := [3]float64{red.Center, red.Center, red.Center}
			moduleSize := [3]float64{red.ModuleSize}
			if !crossCheckPatternVertical(ch[1], int(red.ModuleSize*2),
				float64(x), &center[1], &moduleSize[1], 3) ||
				!crossCheckPatternVertical(ch[2], int(red.ModuleSize*2),
					float64(x), &center[2], &moduleSize[2], 3) ||
				!checkModuleSize3(moduleSize[0], moduleSize[1], moduleSize[2]) {
				continue
			}

			fp := FinderPattern{
				Center: core.PointF{
					X: float64(x),
					Y: (center[0] + center[1] + center[2]) / 3,
				},
				ModuleSize: (moduleSize[0] + moduleSize[1] + moduleSize[2]) / 3,
				FoundCount: 1,
			}
			if !fp.classifyBSIFamily(
				core.BoolColor(ch[0].Pix[int(center[0])*w+x] > 0),
				core.BoolColor(ch[1].Pix[int(center[1])*w+x] > 0),
				core.BoolColor(ch[2].Pix[int(center[2])*w+x] > 0),
			) || !crossCheckPatternBSIFamily(ch, &fp, 1) {
				continue
			}
			d.pass().CrossSurvivors[fp.Typ]++
			saveFinderPattern(&fp, fps, total, typeCount)
		}
	}
}

func estimateMissingBSIFamily(fps []FinderPattern, width, height int) bool {
	missing := -1
	switch {
	case fps[0].FoundCount == 0:
		missing = 0
		s23 := (fps[2].ModuleSize + fps[3].ModuleSize) / 2
		s13 := (fps[1].ModuleSize + fps[3].ModuleSize) / 2
		fps[0].Center.X = (fps[3].Center.X-fps[2].Center.X)/s23*s13 + fps[1].Center.X
		fps[0].Center.Y = (fps[3].Center.Y-fps[2].Center.Y)/s23*s13 + fps[1].Center.Y
		fps[0].Typ, fps[0].FoundCount, fps[0].direction = fp0, 1, -fps[1].direction
		fps[0].ModuleSize = (fps[1].ModuleSize + fps[2].ModuleSize + fps[3].ModuleSize) / 3
	case fps[1].FoundCount == 0:
		missing = 1
		s23 := (fps[2].ModuleSize + fps[3].ModuleSize) / 2
		s02 := (fps[0].ModuleSize + fps[2].ModuleSize) / 2
		fps[1].Center.X = (fps[2].Center.X-fps[3].Center.X)/s23*s02 + fps[0].Center.X
		fps[1].Center.Y = (fps[2].Center.Y-fps[3].Center.Y)/s23*s02 + fps[0].Center.Y
		fps[1].Typ, fps[1].FoundCount, fps[1].direction = fp1, 1, -fps[0].direction
		fps[1].ModuleSize = (fps[0].ModuleSize + fps[2].ModuleSize + fps[3].ModuleSize) / 3
	case fps[2].FoundCount == 0:
		missing = 2
		s01 := (fps[0].ModuleSize + fps[1].ModuleSize) / 2
		s13 := (fps[1].ModuleSize + fps[3].ModuleSize) / 2
		fps[2].Center.X = (fps[1].Center.X-fps[0].Center.X)/s01*s13 + fps[3].Center.X
		fps[2].Center.Y = (fps[1].Center.Y-fps[0].Center.Y)/s01*s13 + fps[3].Center.Y
		fps[2].Typ, fps[2].FoundCount, fps[2].direction = fp2, 1, fps[3].direction
		fps[2].ModuleSize = (fps[0].ModuleSize + fps[1].ModuleSize + fps[3].ModuleSize) / 3
	case fps[3].FoundCount == 0:
		missing = 3
		s01 := (fps[0].ModuleSize + fps[1].ModuleSize) / 2
		s02 := (fps[0].ModuleSize + fps[2].ModuleSize) / 2
		fps[3].Center.X = (fps[0].Center.X-fps[1].Center.X)/s01*s02 + fps[2].Center.X
		fps[3].Center.Y = (fps[0].Center.Y-fps[1].Center.Y)/s01*s02 + fps[2].Center.Y
		fps[3].Typ, fps[3].FoundCount, fps[3].direction = fp3, 1, fps[2].direction
		fps[3].ModuleSize = (fps[0].ModuleSize + fps[1].ModuleSize + fps[2].ModuleSize) / 3
	}
	if missing < 0 {
		return true
	}
	return fps[missing].Center.X >= 0 && fps[missing].Center.X <= float64(width-1) &&
		fps[missing].Center.Y >= 0 && fps[missing].Center.Y <= float64(height-1)
}
