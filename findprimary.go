package jabcode

// Detection modes (jab_detect_mode) and status codes (jabcode.h, decoder.h).
const (
	quickDetect     = 0
	normalDetect    = 1
	intensiveDetect = 2

	jabFailure = 0
	jabSuccess = 1
	fatalError = -2

	maxModules        = 145 // modules in side-version 32
	maxSymbolRows     = 3
	maxFinderPatterns = 500

	fp0CoreColor = 0 // black
	fp1CoreColor = 0 // black
)

// classifyFinderPattern sets fp.typ from the detected core color, returning
// false if the color triple matches no finder-pattern type.
func classifyFinderPattern(fp *finderPattern, candidates []int, typeR, typeG, typeB int) bool {
	for _, t := range candidates {
		core := fpCoreColorIndex(t)
		if typeR == int(defaultPalette[core*3]) && typeG == int(defaultPalette[core*3+1]) && typeB == int(defaultPalette[core*3+2]) {
			fp.typ = t
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
		return fp0CoreColor
	case fp1:
		return fp1CoreColor
	case fp2:
		return fp2CoreColor
	default:
		return fp3CoreColor
	}
}

// findPrimarySymbol scans the binarized channels for the four finder patterns of
// the primary symbol, leaving the working list (with the four selected patterns
// in [0:4]) in d.fps and returning a status (findPrimarySymbol in detector.c).
// It records this pass's counters in d.stats.
func (d *primaryDetector) findPrimarySymbol() int {
	d.stats.passes = append(d.stats.passes, finderPassStats{})
	ch := d.ch

	minModuleSize := ch[0].height / (2 * maxSymbolRows * maxModules)
	if minModuleSize < 1 || d.mode == intensiveDetect {
		minModuleSize = 1
	}

	fps := make([]finderPattern, maxFinderPatterns)
	d.fps = fps
	totalFP := 0
	fpTypeCount := make([]int, 4)
	done := false

	w, h := ch[0].width, ch[0].height
	for i := 0; i < h && !done; i += minModuleSize {
		rowR := ch[0].pix[i*w : (i+1)*w]
		rowG := ch[1].pix[i*w : (i+1)*w]
		rowB := ch[2].pix[i*w : (i+1)*w]

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
			d.pass().rawHits++
			skip = ps.skip
			centerxG, moduleSizeG := ps.center, ps.moduleSize

			typeG := boolColor(rowG[int(centerxG)] > 0)
			centerxR, centerxB := centerxG, centerxG
			var typeR, typeB int
			var moduleSizeR, moduleSizeB float64
			fp1found, fp2found := false, false

			if crossCheckPatternHorizontal(ch[2], moduleSizeG*2, &centerxB, float64(i), &moduleSizeB) {
				typeB = boolColor(rowB[int(centerxB)] > 0)
				moduleSizeR = moduleSizeG
				coreRed := int(defaultPalette[fp3CoreColor*3+0])
				if crossCheckColor(ch[0], coreRed, int(moduleSizeR), 5, int(centerxR), i, 0) {
					typeR = 0
					fp1found = true
				}
			} else if crossCheckPatternHorizontal(ch[0], moduleSizeG*2, &centerxR, float64(i), &moduleSizeR) {
				typeR = boolColor(rowR[int(centerxR)] > 0)
				moduleSizeB = moduleSizeG
				coreBlue := int(defaultPalette[fp2CoreColor*3+2])
				if crossCheckColor(ch[2], coreBlue, int(moduleSizeB), 5, int(centerxB), i, 0) {
					typeB = 0
					fp2found = true
				}
			}

			if !(fp1found || fp2found) {
				continue
			}
			fp := finderPattern{center: pointF{y: float64(i)}, foundCount: 1}
			if fp1found {
				if !checkModuleSize2(moduleSizeG, moduleSizeB) {
					continue
				}
				fp.center.x = (centerxG + centerxB) / 2.0
				fp.moduleSize = (moduleSizeG + moduleSizeB) / 2.0
				if !classifyFinderPattern(&fp, []int{fp0, fp3}, typeR, typeG, typeB) {
					continue
				}
			} else {
				if !checkModuleSize2(moduleSizeR, moduleSizeG) {
					continue
				}
				fp.center.x = (centerxR + centerxG) / 2.0
				fp.moduleSize = (moduleSizeR + moduleSizeG) / 2.0
				if !classifyFinderPattern(&fp, []int{fp1, fp2}, typeR, typeG, typeB) {
					continue
				}
			}
			if crossCheckPattern(ch, &fp, 0) {
				d.pass().crossSurvivors[fp.typ]++
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

	d.candidates = append([]finderPattern(nil), fps[:totalFP]...)

	for i := 0; i < totalFP; i++ {
		if fps[i].direction >= 0 {
			fps[i].direction = 1
		} else {
			fps[i].direction = -1
		}
	}

	missing := d.selectBestPatterns(fps, totalFP, fpTypeCount)
	if missing > 1 {
		d.pass().status = jabFailure
		return jabFailure
	}
	if missing == 1 {
		if !estimateMissingPattern(d.bm, d.ch, fps) {
			d.pass().status = jabFailure
			return jabFailure
		}
		d.pass().interpolated = true
	}
	d.pass().status = jabSuccess
	return jabSuccess
}

// estimateMissingPattern interpolates the position of the single missing finder
// pattern from the other three and confirms it by local search. Returns false if
// the estimate falls outside the image (findPrimarySymbol missing-pattern branch).
func estimateMissingPattern(bm *bitmap, ch [3]*bitmap, fps []finderPattern) bool {
	miss := -1
	switch {
	case fps[0].foundCount == 0:
		miss = 0
		s23 := (fps[2].moduleSize + fps[3].moduleSize) / 2.0
		s13 := (fps[1].moduleSize + fps[3].moduleSize) / 2.0
		fps[0].center.x = (fps[3].center.x-fps[2].center.x)/s23*s13 + fps[1].center.x
		fps[0].center.y = (fps[3].center.y-fps[2].center.y)/s23*s13 + fps[1].center.y
		fps[0].typ, fps[0].foundCount, fps[0].direction = fp0, 1, -fps[1].direction
		fps[0].moduleSize = (fps[1].moduleSize + fps[2].moduleSize + fps[3].moduleSize) / 3.0
	case fps[1].foundCount == 0:
		miss = 1
		s23 := (fps[2].moduleSize + fps[3].moduleSize) / 2.0
		s02 := (fps[0].moduleSize + fps[2].moduleSize) / 2.0
		fps[1].center.x = (fps[2].center.x-fps[3].center.x)/s23*s02 + fps[0].center.x
		fps[1].center.y = (fps[2].center.y-fps[3].center.y)/s23*s02 + fps[0].center.y
		fps[1].typ, fps[1].foundCount, fps[1].direction = fp1, 1, -fps[0].direction
		fps[1].moduleSize = (fps[0].moduleSize + fps[2].moduleSize + fps[3].moduleSize) / 3.0
	case fps[2].foundCount == 0:
		miss = 2
		s01 := (fps[0].moduleSize + fps[1].moduleSize) / 2.0
		s13 := (fps[1].moduleSize + fps[3].moduleSize) / 2.0
		fps[2].center.x = (fps[1].center.x-fps[0].center.x)/s01*s13 + fps[3].center.x
		fps[2].center.y = (fps[1].center.y-fps[0].center.y)/s01*s13 + fps[3].center.y
		fps[2].typ, fps[2].foundCount, fps[2].direction = fp2, 1, fps[3].direction
		fps[2].moduleSize = (fps[0].moduleSize + fps[1].moduleSize + fps[3].moduleSize) / 3.0
	case fps[3].foundCount == 0:
		miss = 3
		s01 := (fps[0].moduleSize + fps[1].moduleSize) / 2.0
		s02 := (fps[0].moduleSize + fps[2].moduleSize) / 2.0
		fps[3].center.x = (fps[0].center.x-fps[1].center.x)/s01*s02 + fps[2].center.x
		fps[3].center.y = (fps[0].center.y-fps[1].center.y)/s01*s02 + fps[2].center.y
		fps[3].typ, fps[3].foundCount, fps[3].direction = fp3, 1, fps[2].direction
		fps[3].moduleSize = (fps[0].moduleSize + fps[1].moduleSize + fps[2].moduleSize) / 3.0
	}
	if fps[miss].center.x < 0 || fps[miss].center.x > float64(ch[0].width-1) ||
		fps[miss].center.y < 0 || fps[miss].center.y > float64(ch[0].height-1) {
		fps[miss].foundCount = 0
		return false
	}
	seekMissingFinderPattern(bm, fps, miss)
	return true
}

// scanPatternVertical scans the image column-wise for finder patterns, used when
// only a top or bottom pair was found horizontally (scanPatternVertical). It
// records its hits in the current pass's d.stats.
func (d *primaryDetector) scanPatternVertical(minModuleSize int, fps []finderPattern, fpTypeCount []int, totalFP *int) {
	ch := d.ch
	w, h := ch[0].width, ch[0].height
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
			d.pass().rawHits++
			skip = ps.skip
			centeryG, moduleSizeG := ps.center, ps.moduleSize

			typeG := boolColor(ch[1].pix[int(centeryG)*w+j] > 0)
			centeryR, centeryB := centeryG, centeryG
			var typeR, typeB int
			var moduleSizeR, moduleSizeB float64
			fp1found, fp2found := false, false

			if crossCheckPatternVertical(ch[2], int(moduleSizeG*2), float64(j), &centeryB, &moduleSizeB) {
				typeB = boolColor(ch[2].pix[int(centeryB)*w+j] > 0)
				moduleSizeR = moduleSizeG
				coreRed := int(defaultPalette[fp3CoreColor*3+0])
				if crossCheckColor(ch[0], coreRed, int(moduleSizeR), 5, j, int(centeryR), 1) {
					typeR = 0
					fp1found = true
				}
			} else if crossCheckPatternVertical(ch[0], int(moduleSizeG*2), float64(j), &centeryR, &moduleSizeR) {
				typeR = boolColor(ch[0].pix[int(centeryR)*w+j] > 0)
				moduleSizeB = moduleSizeG
				coreBlue := int(defaultPalette[fp2CoreColor*3+2])
				if crossCheckColor(ch[2], coreBlue, int(moduleSizeB), 5, j, int(centeryB), 1) {
					typeB = 0
					fp2found = true
				}
			}

			if !(fp1found || fp2found) {
				continue
			}
			fp := finderPattern{center: pointF{x: float64(j)}, foundCount: 1}
			if fp1found {
				if !checkModuleSize2(moduleSizeG, moduleSizeB) {
					continue
				}
				fp.center.y = (centeryG + centeryB) / 2.0
				fp.moduleSize = (moduleSizeG + moduleSizeB) / 2.0
				if !classifyFinderPattern(&fp, []int{fp0, fp3}, typeR, typeG, typeB) {
					continue
				}
			} else {
				if !checkModuleSize2(moduleSizeR, moduleSizeG) {
					continue
				}
				fp.center.y = (centeryR + centeryG) / 2.0
				fp.moduleSize = (moduleSizeR + moduleSizeG) / 2.0
				if !classifyFinderPattern(&fp, []int{fp1, fp2}, typeR, typeG, typeB) {
					continue
				}
			}
			if crossCheckPattern(ch, &fp, 1) {
				d.pass().crossSurvivors[fp.typ]++
				saveFinderPattern(&fp, fps, totalFP, fpTypeCount)
				if *totalFP >= maxFinderPatterns-1 {
					done = true
					break
				}
			}
		}
	}
}

// boolColor maps a binary channel test to the 0/255 color value used for type
// classification.
func boolColor(b bool) int {
	if b {
		return 255
	}
	return 0
}
