//go:build jabcode_bsi || jabcode_legacy

package detect

import (
	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/palette"
)

const bsiFamilyFinderEnabled = true

type optionalFinderPassStats struct {
	bsiAttempted bool
	bsi          FinderFamilyPassStats
}

func (p *FinderPassStats) startBSIFamily() { p.bsiAttempted = true }

func (p *FinderPassStats) bsiFamily() *FinderFamilyPassStats { return &p.bsi }

// familyStats returns the counters a signature records into, so callers that
// work across both can stay signature-agnostic.
func (d *PrimaryDetector) familyStats(family FinderFamily) *FinderFamilyPassStats {
	if family == FinderFamilyBSI {
		return d.pass().bsiFamily()
	}
	return &d.pass().FinderFamilyPassStats
}

// BSIFamilyStats returns the optional BSI/pre-v2.0 signature counters when
// that signature was requested in this pass.
func (p FinderPassStats) BSIFamilyStats() (FinderFamilyPassStats, bool) {
	return p.bsi, p.bsiAttempted
}

// classifyBSIFamily maps a binarized core color to the primary finder type
// defined by BSI TR-03137 and retained by pre-v2.0 C-reference releases.
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

func crossCheckPatternBSIFamily(ch [3]*core.Bitmap, fp *FinderPattern, hv, slack int) bool {
	moduleSizeMax := fp.ModuleSize * 2
	var moduleSize [3]float64
	var centerX, centerY [3]float64
	var direction, diagonal [3]int
	for c := range 3 {
		centerX[c], centerY[c] = fp.Center.X, fp.Center.Y
		if !crossCheckPatternCh(ch[c], fp.Typ, hv, moduleSizeMax,
			&moduleSize[c], &centerX[c], &centerY[c],
			&direction[c], &diagonal[c], slack) {
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

// scanBSIFamilyRow adds BSI-era candidates from one row to the same prepared
// image pass that scans the current-family signature.
func (d *PrimaryDetector) scanBSIFamilyRow(rows [3][]byte, y int, state *primaryFamilyScan) {
	w := d.Ch[0].Width
	start, end, skip := 0, w, 0
	for first := true; first || (start < w && end < w); {
		first = false
		start += skip
		end = w
		red := seekPatternHorizontal(rows[0], start, end)
		start, end = red.start, red.end
		if !red.ok {
			continue
		}
		skip = red.skip
		d.processBSIFamilyHit(y, red.Center, red.ModuleSize, rows, state)
		if state.done {
			return
		}
	}
}

// consumeBSIFamilyHits replays the device row scan's raw red-row hits in the
// CPU row walk's order. When the pass also ran the device BSI chain, each
// outcome record replays without touching the mask channels; before the
// background chain kernel is compiled, the bit-identical CPU per-hit chain
// processes the same hits instead.
func (d *PrimaryDetector) consumeBSIFamilyHits(hits *finderPassRowHits, minModuleSize int, state *primaryFamilyScan) {
	replay := hits.chained(0)
	if !replay && !d.ensureChannels() {
		return
	}
	ch := d.Ch
	w := ch[0].Width
	for _, hit := range hits.channels[0] {
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
			d.processBSIFamilyHit(hit.y, hit.center(), hit.moduleSize(), rows, state)
			continue
		}
		stats := d.pass().bsiFamily()
		stats.RawHits++
		d.bsiFamilySeedModules = append(d.bsiFamilySeedModules, hit.moduleSize())
		outcome := hits.outcomes[hit.rec]
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
		stats.CrossSurvivors[fp.Typ]++
		if state.fps == nil {
			state.fps = make([]FinderPattern, maxFinderPatterns)
		}
		saveFinderPattern(&fp, state.fps, &state.total, state.typeCount[:])
		if state.total >= maxFinderPatterns-1 {
			state.done = true
		}
	}
}

// processBSIFamilyHit runs the cross-check and classification chain of one
// raw n-1-1-1-m red-row hit, saving a surviving finder pattern into state.
func (d *PrimaryDetector) processBSIFamilyHit(
	y int,
	center0, module0 float64,
	rows [3][]byte,
	state *primaryFamilyScan,
) {
	ch := d.Ch
	stats := d.pass().bsiFamily()
	stats.RawHits++
	d.bsiFamilySeedModules = append(d.bsiFamilySeedModules, module0)
	slack := d.ccSlack(module0)

	center := [3]float64{center0, center0, center0}
	moduleSize := [3]float64{module0}
	if !crossCheckPatternHorizontal(ch[1], module0*2,
		&center[1], float64(y), &moduleSize[1], slack) ||
		!crossCheckPatternHorizontal(ch[2], module0*2,
			&center[2], float64(y), &moduleSize[2], slack) ||
		!checkModuleSize3(moduleSize[0], moduleSize[1], moduleSize[2]) {
		return
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
		core.BoolColor(rows[0][int(center[0])] > 0),
		core.BoolColor(rows[1][int(center[1])] > 0),
		core.BoolColor(rows[2][int(center[2])] > 0),
	) || !crossCheckPatternBSIFamily(ch, &fp, 0, d.ccSlack(fp.ModuleSize)) {
		return
	}
	stats.CrossSurvivors[fp.Typ]++
	if state.fps == nil {
		state.fps = make([]FinderPattern, maxFinderPatterns)
	}
	saveFinderPattern(&fp, state.fps, &state.total, state.typeCount[:])
	if state.total >= maxFinderPatterns-1 {
		state.done = true
	}
}

func (d *PrimaryDetector) scanPatternVerticalBSIFamily(minModuleSize int, state *primaryFamilyScan) {
	ch := d.Ch
	w, h := ch[0].Width, ch[0].Height
	for x := 0; x < w && state.total < maxFinderPatterns-1; x += minModuleSize {
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
			stats := d.pass().bsiFamily()
			stats.RawHits++
			d.bsiFamilySeedModules = append(d.bsiFamilySeedModules, red.ModuleSize)
			skip = red.skip
			slack := d.ccSlack(red.ModuleSize)

			center := [3]float64{red.Center, red.Center, red.Center}
			moduleSize := [3]float64{red.ModuleSize}
			if !crossCheckPatternVertical(ch[1], int(red.ModuleSize*2),
				float64(x), &center[1], &moduleSize[1], slack) ||
				!crossCheckPatternVertical(ch[2], int(red.ModuleSize*2),
					float64(x), &center[2], &moduleSize[2], slack) ||
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
			) || !crossCheckPatternBSIFamily(ch, &fp, 1, d.ccSlack(fp.ModuleSize)) {
				continue
			}
			stats.CrossSurvivors[fp.Typ]++
			saveFinderPattern(&fp, state.fps, &state.total, state.typeCount[:])
		}
	}
}

func (d *PrimaryDetector) finishBSIFamilyScan(state *primaryFamilyScan, degrees float64) finderFamilyResult {
	stats := d.pass().bsiFamily()
	if state.total == 0 {
		stats.Scans = append(stats.Scans, FinderFamilyScanStats{
			Degrees: degrees, Missing: 4, Status: core.Failure,
		})
		return finderFamilyResult{status: core.Failure, scan: len(stats.Scans) - 1}
	}
	candidates := append([]FinderPattern(nil), state.fps[:state.total]...)
	stats.Candidates = candidates
	d.accumulateFamilyCandidates(FinderFamilyBSI, candidates)
	for i := range state.total {
		if state.fps[i].direction >= 0 {
			state.fps[i].direction = 1
		} else {
			state.fps[i].direction = -1
		}
	}

	scan := FinderFamilyScanStats{Degrees: degrees}
	var pre [4]FinderPattern
	missing := d.selectBestPatternsFor(state.fps, state.total, state.typeCount[:], &scan, &pre)
	status := core.Success
	if missing > 1 {
		status = core.Failure
	} else if missing == 1 {
		src, ok := estimateMissingBSIFamily(state.fps,
			d.familyPassCandidates[FinderFamilyBSI], d.Ch[0].Width, d.Ch[0].Height)
		if !ok {
			status = core.Failure
		} else {
			scan.Corner = src
		}
	}
	scan.Status = status
	scan.Consistent = status == core.Success && ConsistentFinderQuad(state.fps)
	stats.Scans = append(stats.Scans, scan)
	d.recordScanQuad(FinderFamilyBSI, len(stats.Scans)-1, state.fps, pre)
	return finderFamilyResult{
		fps: state.fps, candidates: candidates, channels: d.Ch,
		status: status, printDetected: d.printPass, scan: len(stats.Scans) - 1,
	}
}

// estimateMissingBSIFamily completes a BSI quad that lost exactly one corner,
// preferring a candidate the scan found over the construction. The geometry is
// the current family's: both signatures are the same joined-squares pattern and
// differ only in core colour, so the completion and its pooled confirmation are
// shared rather than duplicated. What is not shared is the image seek, which
// the BSI-era reference never had.
//
// It reports where the corner came from, and whether the completed quad is
// usable at all.
func estimateMissingBSIFamily(fps, pool []FinderPattern, width, height int) (CornerSource, bool) {
	miss, missing := interpolateMissingPattern(fps)
	if !missing {
		return CornerFound, true
	}
	if fps[miss].Center.X < 0 || fps[miss].Center.X > float64(width-1) ||
		fps[miss].Center.Y < 0 || fps[miss].Center.Y > float64(height-1) {
		return CornerConstructed, false
	}
	if c, ok := pickPooledCorner(pool, fps, miss); ok {
		fps[miss] = c
		return CornerPooled, true
	}
	return CornerConstructed, true
}
