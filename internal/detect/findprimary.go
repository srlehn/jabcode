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

// classify sets fp.Typ from the detected core color, returning
// false if the color triple matches no finder-pattern type.
func (fp *FinderPattern) classify(candidates []int, typeR, typeG, typeB int) bool {
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

type primaryFamilyScan struct {
	fps       []FinderPattern
	total     int
	typeCount [4]int
	done      bool
}

func newPrimaryFamilyScan() primaryFamilyScan {
	return primaryFamilyScan{fps: make([]FinderPattern, maxFinderPatterns)}
}

// findPrimarySymbol scans the binarized channels for the four current-family
// primary finder patterns, leaves the working list with the four selected
// patterns in d.FPs[0:4], and returns the current signature's status. It
// records the pass counters in d.Stats. This compatibility entry is used by
// focused detector tests; production additive reads call findPrimaryFamilies.
func (d *PrimaryDetector) findPrimarySymbol() int {
	d.findPrimaryFamilies(true, false)
	return d.familyResults[FinderFamilyCurrent].status
}

// findPrimaryFamilies scans the binarized channels once per prepared image
// pass and classifies every enabled physical finder signature during that
// traversal. Each result retains its selected four-pattern working list and
// all pre-prune candidates, while the shared FinderPassStats entry records the
// per-signature counters from the same input and channel set.
func (d *PrimaryDetector) findPrimaryFamilies(wantCurrent, wantBSI bool) FinderFamilySet {
	// Ports findPrimarySymbol in detector.c and the BSI-era equivalent.
	d.Stats.Passes = append(d.Stats.Passes, FinderPassStats{})
	d.passFamilies = 0
	if wantCurrent {
		d.passFamilies |= FinderFamilyCurrent.Mask()
	}
	if wantBSI {
		d.passFamilies |= FinderFamilyBSI.Mask()
		d.pass().startBSIFamily()
	}
	ch := d.Ch
	minModuleSize := ch[0].Height / (2 * maxSymbolRows * maxModules)
	if minModuleSize < 1 || d.Mode == IntensiveDetect {
		minModuleSize = 1
	}

	var current primaryFamilyScan
	if wantCurrent {
		current = newPrimaryFamilyScan()
	}
	var bsi primaryFamilyScan

	// A device row scan for a channel replaces that family's CPU row walk:
	// its hits are bit-identical to the walk's raw seeds and the device chain
	// already ran the per-hit cross-check processing, so the consumer only
	// replays counters and survivors in the walk's order. Families without
	// device hits (no session, an overflowed record buffer) keep the walk.
	hits := d.rowHits
	d.rowHits = nil
	hitsCurrent := wantCurrent && hits.scanned(1)
	hitsBSI := wantBSI && hits.scanned(0)
	if hitsCurrent {
		d.consumeCurrentFamilyHits(hits, minModuleSize, &current)
	}
	if hitsBSI {
		d.consumeBSIFamilyHits(hits, minModuleSize, &bsi)
	}

	walkCurrent := wantCurrent && !hitsCurrent
	walkBSI := wantBSI && !hitsBSI
	w, h := ch[0].Width, ch[0].Height
	for y := 0; y < h && ((walkCurrent && !current.done) || (walkBSI && !bsi.done)); y += minModuleSize {
		rows := [3][]byte{
			ch[0].Pix[y*w : (y+1)*w],
			ch[1].Pix[y*w : (y+1)*w],
			ch[2].Pix[y*w : (y+1)*w],
		}
		if walkCurrent && !current.done {
			d.scanCurrentFamilyRow(rows, y, &current)
		}
		if walkBSI && !bsi.done {
			d.scanBSIFamilyRow(rows, y, &bsi)
		}
	}

	if wantCurrent {
		if needsVerticalScan(current.typeCount) {
			d.scanPatternVertical(minModuleSize, current.fps, current.typeCount[:], &current.total)
		}
		d.familyResults[FinderFamilyCurrent] = d.finishCurrentFamilyScan(&current)
	} else {
		d.familyResults[FinderFamilyCurrent] = finderFamilyResult{status: core.Failure}
	}

	if wantBSI {
		if needsVerticalScan(bsi.typeCount) {
			d.scanPatternVerticalBSIFamily(minModuleSize, &bsi)
		}
		d.familyResults[FinderFamilyBSI] = d.finishBSIFamilyScan(&bsi)
	} else {
		d.familyResults[FinderFamilyBSI] = finderFamilyResult{status: core.Failure}
	}
	if wantCurrent {
		d.SelectFinderFamily(FinderFamilyCurrent)
	} else if wantBSI {
		d.SelectFinderFamily(FinderFamilyBSI)
	}

	var found FinderFamilySet
	for family := FinderFamily(0); family < finderFamilyCount; family++ {
		if d.familyResults[family].status == core.Success {
			found |= 1 << family
		}
	}
	return found
}

func needsVerticalScan(typeCount [4]int) bool {
	// If only FP0+FP1 or only FP2+FP3 were found, also scan vertically.
	return (typeCount[0] != 0 && typeCount[1] != 0 && typeCount[2] == 0 && typeCount[3] == 0) ||
		(typeCount[0] == 0 && typeCount[1] == 0 && typeCount[2] != 0 && typeCount[3] != 0)
}

func (d *PrimaryDetector) scanCurrentFamilyRow(rows [3][]byte, y int, state *primaryFamilyScan) {
	w := d.Ch[0].Width
	rowG := rows[1]
	startX, endX, skip := 0, w, 0
	for first := true; first || (startX < w && endX < w); {
		first = false
		startX += skip
		endX = w
		ps := seekPatternHorizontal(rowG, startX, endX)
		startX, endX = ps.start, ps.end
		if !ps.ok {
			continue
		}
		skip = ps.skip
		d.processCurrentFamilyHit(y, ps.Center, ps.ModuleSize, rows, state)
		if state.done {
			return
		}
	}
}

// consumeCurrentFamilyHits replays the device row scan's raw hits in the CPU
// row walk's order. When the pass also ran the device chain, each outcome
// record replays its counters and surviving finder pattern without touching
// the mask channels; before the background chain kernel is compiled, the
// bit-identical CPU per-hit chain processes the same hits instead.
func (d *PrimaryDetector) consumeCurrentFamilyHits(hits *finderPassRowHits, minModuleSize int, state *primaryFamilyScan) {
	replay := hits.chained(1)
	ch := d.Ch
	w := ch[0].Width
	for _, hit := range hits.channels[1] {
		if state.done {
			return
		}
		if minModuleSize > 1 && hit.y%minModuleSize != 0 {
			continue
		}
		if !replay {
			rows := [3][]byte{
				ch[0].Pix[hit.y*w : (hit.y+1)*w],
				ch[1].Pix[hit.y*w : (hit.y+1)*w],
				ch[2].Pix[hit.y*w : (hit.y+1)*w],
			}
			d.processCurrentFamilyHit(hit.y, hit.center(), hit.moduleSize(), rows, state)
			continue
		}
		d.pass().RawHits++
		d.seedModules = append(d.seedModules, hit.moduleSize())
		outcome := hits.outcomes[hit.rec]
		if outcome.flags&chainFlagBranchBlue != 0 {
			d.pass().BranchBlue++
		}
		if outcome.flags&chainFlagBranchRed != 0 {
			d.pass().BranchRed++
		}
		if outcome.flags&chainFlagRedColor != 0 {
			d.pass().RedColor++
		}
		if outcome.flags&chainFlagRedClassified != 0 {
			d.pass().RedClassified++
		}
		if outcome.flags&chainFlagSurvivor == 0 {
			continue
		}
		fp := FinderPattern{
			Typ:        outcome.typ,
			ModuleSize: outcome.moduleSize,
			Center:     core.PointF{X: outcome.centerX, Y: outcome.centerY},
			FoundCount: 1,
			direction:  outcome.direction,
		}
		d.pass().CrossSurvivors[fp.Typ]++
		saveFinderPattern(&fp, state.fps, &state.total, state.typeCount[:])
		if state.total >= maxFinderPatterns-1 {
			state.done = true
		}
	}
}

// processCurrentFamilyHit runs the cross-check and classification chain of one
// raw n-1-1-1-m green-row hit, saving a surviving finder pattern into state.
func (d *PrimaryDetector) processCurrentFamilyHit(
	y int,
	centerG, moduleG float64,
	rows [3][]byte,
	state *primaryFamilyScan,
) {
	ch := d.Ch
	rowR, rowG, rowB := rows[0], rows[1], rows[2]
	d.pass().RawHits++
	d.seedModules = append(d.seedModules, moduleG)

	typeG := core.BoolColor(rowG[int(centerG)] > 0)
	centerR, centerB := centerG, centerG
	var typeR, typeB int
	var moduleR, moduleB float64
	blueBranch, redBranch := false, false
	slack := d.ccSlack(moduleG)

	if crossCheckPatternHorizontal(ch[2], moduleG*2, &centerB, float64(y), &moduleB, slack) {
		d.pass().BranchBlue++
		typeB = core.BoolColor(rowB[int(centerB)] > 0)
		moduleR = moduleG
		coreRed := int(palette.Default[spec.FP3CoreColor*3])
		if crossCheckColor(ch[0], coreRed, int(moduleR), 5, int(centerR), y, 0, slack) {
			typeR = 0
			blueBranch = true
		}
	} else if crossCheckPatternHorizontal(ch[0], moduleG*2, &centerR, float64(y), &moduleR, slack) {
		d.pass().BranchRed++
		typeR = core.BoolColor(rowR[int(centerR)] > 0)
		moduleB = moduleG
		coreBlue := int(palette.Default[spec.FP2CoreColor*3+2])
		if crossCheckColor(ch[2], coreBlue, int(moduleB), 5, int(centerB), y, 0, slack) {
			typeB = 0
			redBranch = true
			d.pass().RedColor++
		}
	}

	if !(blueBranch || redBranch) {
		return
	}
	fp := FinderPattern{Center: core.PointF{Y: float64(y)}, FoundCount: 1}
	if blueBranch {
		if !checkModuleSize2(moduleG, moduleB) {
			return
		}
		fp.Center.X = (centerG + centerB) / 2
		fp.ModuleSize = (moduleG + moduleB) / 2
		if !fp.classify([]int{fp0, fp3}, typeR, typeG, typeB) {
			return
		}
	} else {
		if !checkModuleSize2(moduleR, moduleG) {
			return
		}
		fp.Center.X = (centerR + centerG) / 2
		fp.ModuleSize = (moduleR + moduleG) / 2
		if !fp.classify([]int{fp1, fp2}, typeR, typeG, typeB) {
			return
		}
		d.pass().RedClassified++
	}
	if crossCheckPattern(ch, &fp, 0, d.ccSlack(fp.ModuleSize)) {
		d.pass().CrossSurvivors[fp.Typ]++
		saveFinderPattern(&fp, state.fps, &state.total, state.typeCount[:])
		if state.total >= maxFinderPatterns-1 {
			state.done = true
		}
	}
}

func (d *PrimaryDetector) finishCurrentFamilyScan(state *primaryFamilyScan) finderFamilyResult {
	candidates := append([]FinderPattern(nil), state.fps[:state.total]...)
	d.pass().Candidates = candidates
	for i := range state.total {
		if state.fps[i].direction >= 0 {
			state.fps[i].direction = 1
		} else {
			state.fps[i].direction = -1
		}
	}

	missing := d.selectBestPatterns(state.fps, state.total, state.typeCount[:])
	status := core.Success
	if missing > 1 {
		status = core.Failure
	} else if missing == 1 {
		if !d.ensureBitmap() || !estimateMissingPattern(d.BM, d.Ch, state.fps) {
			status = core.Failure
		} else {
			d.pass().Interpolated = true
		}
	}
	d.pass().Status = status
	return finderFamilyResult{
		fps: state.fps, candidates: candidates, channels: d.Ch,
		status: status, printDetected: d.printPass,
	}
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
			slack := d.ccSlack(moduleSizeG)

			if crossCheckPatternVertical(ch[2], int(moduleSizeG*2), float64(j), &centeryB, &moduleSizeB, slack) {
				typeB = core.BoolColor(ch[2].Pix[int(centeryB)*w+j] > 0)
				moduleSizeR = moduleSizeG
				coreRed := int(palette.Default[spec.FP3CoreColor*3+0])
				if crossCheckColor(ch[0], coreRed, int(moduleSizeR), 5, j, int(centeryR), 1, slack) {
					typeR = 0
					fp1found = true
				}
			} else if crossCheckPatternVertical(ch[0], int(moduleSizeG*2), float64(j), &centeryR, &moduleSizeR, slack) {
				typeR = core.BoolColor(ch[0].Pix[int(centeryR)*w+j] > 0)
				moduleSizeB = moduleSizeG
				coreBlue := int(palette.Default[spec.FP2CoreColor*3+2])
				if crossCheckColor(ch[2], coreBlue, int(moduleSizeB), 5, j, int(centeryB), 1, slack) {
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
				if !fp.classify([]int{fp0, fp3}, typeR, typeG, typeB) {
					continue
				}
			} else {
				if !checkModuleSize2(moduleSizeR, moduleSizeG) {
					continue
				}
				fp.Center.Y = (centeryR + centeryG) / 2.0
				fp.ModuleSize = (moduleSizeR + moduleSizeG) / 2.0
				if !fp.classify([]int{fp1, fp2}, typeR, typeG, typeB) {
					continue
				}
			}
			if crossCheckPattern(ch, &fp, 1, d.ccSlack(fp.ModuleSize)) {
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
