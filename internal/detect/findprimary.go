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

	maxModules               = 145 // modules in side-version 32
	maxSymbolRows            = 3
	maxFinderPatterns        = 500
	maxContextualFinderSeeds = 32768
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
	weak      []FinderPattern
	total     int
	typeCount [4]int
	done      bool
}

func newPrimaryFamilyScan() primaryFamilyScan {
	return primaryFamilyScan{
		fps:  make([]FinderPattern, maxFinderPatterns),
		weak: make([]FinderPattern, 0, 1024),
	}
}

func (state *primaryFamilyScan) retainContextualSeed(fp FinderPattern) {
	if len(state.weak) < maxContextualFinderSeeds {
		state.weak = append(state.weak, fp)
	}
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
	if (walkCurrent || walkBSI) && !d.ensureChannels() {
		walkCurrent, walkBSI = false, false
	}
	w, h := ch[0].Width, ch[0].Height
	for y := 0; y < h && ((walkCurrent && !current.done) || (walkBSI && !bsi.done)); y += minModuleSize {
		// A cancelled route abandons the walk here rather than at the next
		// pass boundary, because a whole-frame walk is the largest single
		// piece of work in a pass. Returning before the selection below is
		// what makes that safe: a half-walked frame has seen only part of the
		// candidate population, so letting it select would risk publishing a
		// quad assembled from corners that a full walk would have rejected -
		// and a cancelled route's geometry is still read, by the seeded route.
		// Reporting no family instead leaves the previous pass's results,
		// which are a genuine failure or absent.
		if d.Quitting() {
			return 0
		}
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

	// The same abort as in the walk, for the two ways of reaching this point
	// that the walk's own poll does not cover: a pass whose row hits came from
	// the device skips the walk entirely, and a walk that ran to completion
	// still has the vertical rescans and the selection ahead of it.
	if d.Quitting() {
		return 0
	}

	if wantCurrent {
		if needsVerticalScan(current.typeCount) && d.ensureChannels() {
			d.scanPatternVertical(minModuleSize, current.fps, current.typeCount[:], &current.total)
		}
		d.familyResults[FinderFamilyCurrent] = d.finishCurrentFamilyScan(&current, 0)
	} else {
		d.familyResults[FinderFamilyCurrent] = finderFamilyResult{status: core.Failure, scan: -1}
	}

	if wantBSI {
		if needsVerticalScan(bsi.typeCount) && d.ensureChannels() {
			d.scanPatternVerticalBSIFamily(minModuleSize, &bsi)
		}
		d.familyResults[FinderFamilyBSI] = d.finishBSIFamilyScan(&bsi, 0)
	} else {
		d.familyResults[FinderFamilyBSI] = finderFamilyResult{status: core.Failure, scan: -1}
	}
	if wantCurrent {
		d.SelectFinderFamily(FinderFamilyCurrent)
	} else if wantBSI {
		d.SelectFinderFamily(FinderFamilyBSI)
	}

	// The row walk is the first of six scan directions and gets no special
	// standing: a quad it assembles from mismatched corners must not stop the
	// remaining directions any more than a directional one may.
	var picks [finderFamilyCount]familyPick
	found := d.locatedFamilies()
	for family := FinderFamily(0); family < finderFamilyCount; family++ {
		if found.Has(family) {
			picks[family].offer(d.familyResults[family], d.familyPassCandidates[family])
		}
	}
	if found != 0 && d.settled(picks, wantCurrent, wantBSI) {
		d.markPublishedScans(found, &picks)
		return found
	}
	return d.retryScanDirections(wantCurrent, wantBSI, minModuleSize, &picks)
}

// familyPick holds the quad one wire family will publish: the first consistent
// one any scan direction produced, or the first one that merely located if no
// direction produced a consistent quad.
//
// Corner provenance does not rank directions here. The retry scans a family
// only while its retained quad is inconsistent, so preferring a later detected
// corner would be unreachable without also changing that retry policy.
//
// The gate is ConsistentFinderQuad, not ScoreFinderQuad. Only a coarse validity
// predicate belongs here - convexity, module-size agreement, opposite-edge
// agreement - because the question is whether four patterns can be one symbol's
// corners at all. ScoreFinderQuad derives a wire size and applies the stricter
// edge-span gate used by the bounded consensus fallback; making it terminate
// the direction sweep would turn a fallback plausibility score into an early
// rejection of perspective quads.
type familyPick struct {
	result     finderFamilyResult
	candidates []FinderPattern
	consistent bool
	have       bool
}

// offer records r if it improves on what this family already has. candidates is
// snapshotted because the accumulation keeps growing as later directions scan,
// and a fallback must restore the candidate set the winning direction saw, not
// a union across directions that first-wins would never have assembled.
func (p *familyPick) offer(r finderFamilyResult, candidates []FinderPattern) {
	if r.status != core.Success {
		return
	}
	consistent := ConsistentFinderQuad(r.fps)
	if p.have && (p.consistent || !consistent) {
		return
	}
	p.result = r
	p.candidates = append([]FinderPattern(nil), candidates...)
	p.consistent, p.have = consistent, true
}

// settled reports whether the direction sweep can stop.
//
// The current family decides it whenever it is wanted: the historical signature
// is an additive fallback, so a consistent quad of its own must not end the
// search for the wire format the caller actually asked for. Letting it would
// reintroduce family coupling at termination after it was removed from
// selection, and a spurious historical quad at an early direction would hide a
// sound current one at a later direction.
//
// Requiring *every* wanted family is equally wrong in the other direction: it
// would sweep the remaining directions on the historical signature's behalf
// after the current one was already sound, spending time no untagged build
// spends. Only the current family's own state ends the sweep, and a family that
// has not settled keeps whatever quad it located, so this decides when to stop
// looking and never which quad a family publishes.
// This must stay the same predicate the retry loop guards each family's scan
// with. Making termination stricter than that guard does not extend the search:
// the family stops scanning on its own condition, and the loop only keeps
// spinning for the other family.
func (d *PrimaryDetector) settled(picks [finderFamilyCount]familyPick, wantCurrent, wantBSI bool) bool {
	if wantCurrent {
		return picks[FinderFamilyCurrent].consistent
	}
	return wantBSI && picks[FinderFamilyBSI].consistent
}

// retryScanDirections re-runs the enabled family scans along the remaining
// probe directions when the row walk found no symbol. The rows are only the
// first of six directions; a symbol whose axes are more than about ten degrees
// off the image axes leaves a straight row scan crossing the finder's data
// quadrants instead of its pattern, and turning the scan line recovers it
// without turning the frame. All six read the channels prepared for this pass,
// which is what makes the ladder's per-angle canvases unnecessary.
//
// Every wire family gets this, not just the current one: the BSI-era finder is
// the same joined-squares construction and fails off-axis for the same reason.
//
// A located quad only stops the search once it is a consistent one. Locating
// four typed patterns does not require them to be four *different* corners: an
// off-angle direction on a rotated capture readily picks two real corners plus
// two near-duplicates a few hundred pixels away, and the sliver that assembles
// from them locates like any other quad. Returning on it hides the direction
// that would have found the symbol, which is what made those captures need the
// frame turned instead.
//
// The consistency test does not detect duplicate corners as such, and must not
// be described as if it did. It rejects quads whose opposite edges or whose
// finder module sizes disagree, and a sliver assembled from patterns that all
// measured the same scale passes it. What it reliably rejects is the case where
// the duplicated corners were measured at different scales, which is why it
// catches the ones seen so far. Anything stronger needs a real duplicate test.
//
// Consistency decides, never a ranking between consistent quads: a geometry
// score orders plausibility, not decodability, and preferring a better-scoring
// later quad over an already-sound earlier one costs real captures.
//
// A family stops contributing once it has a consistent quad, but the families
// are never split into separate traversals: they share one scan per direction
// and one set of prepared channels, and one family's spurious quad must not
// discard another's sound one.
//
// Cost falls only on frames whose row walk produced nothing consistent, and
// only until every wanted family has a consistent quad.
func (d *PrimaryDetector) retryScanDirections(
	wantCurrent, wantBSI bool,
	step int,
	picks *[finderFamilyCount]familyPick,
) FinderFamilySet {
	if d.AxisAlignedScan {
		return d.publishPicks(picks, wantCurrent, wantBSI)
	}
	for _, deg := range scanDirections[1:] {
		if d.Quitting() {
			return 0
		}
		dir := newScanDirection(deg)
		if wantCurrent && !picks[FinderFamilyCurrent].consistent {
			state := newPrimaryFamilyScan()
			d.sweepDirectionalFamily(dir, step, &state, directionalFamily{
				channel:   currentFamilySeekChannel,
				onSummary: d.applyDirectionalSummary,
				onHit:     d.processDirectionalFamilyHit,
				onHits:    d.processDirectionalFamilyHits,
				walk:      d.scanDirectionalFamily,
			})
			if d.Quitting() {
				return 0
			}
			r := d.finishCurrentFamilyScan(&state, deg)
			picks[FinderFamilyCurrent].offer(r, d.familyPassCandidates[FinderFamilyCurrent])
			// Each family walks the frame itself; only the prepared channels
			// are shared. So a current family that just settled must stop the
			// direction here, before the historical signature walks the same
			// frame again for a result nothing will read.
			if d.settled(*picks, wantCurrent, wantBSI) {
				break
			}
		}
		if wantBSI && !picks[FinderFamilyBSI].consistent {
			state := newPrimaryFamilyScan()
			d.sweepDirectionalBSIFamily(dir, step, &state)
			if d.Quitting() {
				return 0
			}
			r := d.finishBSIFamilyScan(&state, deg)
			picks[FinderFamilyBSI].offer(r, d.familyPassCandidates[FinderFamilyBSI])
		}
		if d.settled(*picks, wantCurrent, wantBSI) {
			break
		}
	}
	return d.publishPicks(picks, wantCurrent, wantBSI)
}

// publishPicks installs each family's chosen quad and its candidate set as the
// detector's result for this pass, and marks the scan direction it came from so
// the pass summary describes that direction rather than the last one tried.
func (d *PrimaryDetector) publishPicks(
	picks *[finderFamilyCount]familyPick,
	wantCurrent, wantBSI bool,
) FinderFamilySet {
	var found FinderFamilySet
	for family := FinderFamily(0); family < finderFamilyCount; family++ {
		if !picks[family].have {
			continue
		}
		d.familyResults[family] = picks[family].result
		d.familyPassCandidates[family] = picks[family].candidates
		found |= family.Mask()
	}
	d.markPublishedScans(found, picks)
	if found == 0 {
		return 0
	}
	if wantCurrent {
		d.SelectFinderFamily(FinderFamilyCurrent)
	} else if wantBSI {
		d.SelectFinderFamily(FinderFamilyBSI)
	}
	return found
}

// markPublishedScans mirrors each published pick's selection up to its family's
// pass summary. Both exits from the direction sweep go through here, so a pass
// that settled on the row walk reports the row walk rather than nothing.
func (d *PrimaryDetector) markPublishedScans(found FinderFamilySet, picks *[finderFamilyCount]familyPick) {
	if len(d.Stats.Passes) == 0 {
		return
	}
	for family := FinderFamily(0); family < finderFamilyCount; family++ {
		if !found.Has(family) {
			continue
		}
		stats := d.familyStats(family)
		stats.publishScan(picks[family].result.scan)
		// Candidates is the one field the scan record does not hold, because a
		// per-direction copy of every candidate is far larger than the rest of
		// the stats put together. Restoring the published direction's own list
		// here keeps the summary whole without paying for the other five.
		if c := picks[family].result.candidates; c != nil {
			stats.Candidates = c
		}
	}
}

func (d *PrimaryDetector) locatedFamilies() FinderFamilySet {
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
	if !replay && !d.ensureChannels() {
		return
	}
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
		d.seedModules.add(hit.moduleSize())
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
		// The device decides the source-colour signal when the balanced image
		// was bound to its chain. Answering it on the host is what forced the
		// whole image across the bus during a locate, so the host only steps in
		// for a pass whose kernel could not evaluate it.
		if fp.Typ == fp1 || fp.Typ == fp2 {
			if outcome.flags&chainFlagColorEvaluated != 0 {
				if outcome.flags&chainFlagColorOK == 0 {
					continue
				}
			} else if !d.ensureBitmap() ||
				!finderPatternHasColorSignal(d.BM, fp, newScanDirection(0)) {
				continue
			}
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
	d.seedModules.add(moduleG)

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
	if (fp.Typ == fp1 || fp.Typ == fp2) &&
		(!d.ensureBitmap() || !finderPatternHasColorSignal(d.BM, fp, newScanDirection(0))) {
		return
	}
	seed := fp
	if crossCheckPattern(ch, &fp, 0, d.ccSlack(fp.ModuleSize)) {
		d.pass().CrossSurvivors[fp.Typ]++
		saveFinderPattern(&fp, state.fps, &state.total, state.typeCount[:])
		if state.total >= maxFinderPatterns-1 {
			state.done = true
		}
	} else {
		state.retainContextualSeed(seed)
	}
}

func (d *PrimaryDetector) finishCurrentFamilyScan(state *primaryFamilyScan, degrees float64) finderFamilyResult {
	candidates := append([]FinderPattern(nil), state.fps[:state.total]...)
	d.pass().Candidates = candidates
	d.accumulateFamilyCandidates(FinderFamilyCurrent, candidates)
	contextual := contextualFinderCandidates(state.weak)
	d.accumulateContextualFinderCandidates(contextual)
	var contextualTypes [4]bool
	for _, candidate := range d.contextualCandidates {
		contextualTypes[candidate.Typ] = true
	}
	for i := range state.total {
		if state.fps[i].direction >= 0 {
			state.fps[i].direction = 1
		} else {
			state.fps[i].direction = -1
		}
	}

	scan := FinderFamilyScanStats{Degrees: degrees}
	var pre [4]FinderPattern
	missing := d.selectBestPatternsFor(
		state.fps, state.total, state.typeCount[:], contextualTypes, &scan, &pre,
	)
	status := core.Success
	var alternatives []FinderQuadHypothesis
	if missing > 1 {
		status = core.Failure
	} else if missing == 1 {
		if src, miss, ok := estimateMissingPattern(d.Balanced, d.Ch, state.fps,
			d.familyPassCandidates[FinderFamilyCurrent]); !ok {
			status = core.Failure
		} else {
			scan.Corner = src
			if src == CornerConstructed {
				alternatives = contextualFinderQuads(state.fps, miss, d.contextualCandidates)
			}
		}
	}
	scan.Status = status
	scan.Consistent = status == core.Success && ConsistentFinderQuad(state.fps)
	stats := &d.pass().FinderFamilyPassStats
	stats.Scans = append(stats.Scans, scan)
	d.recordScanQuad(FinderFamilyCurrent, len(stats.Scans)-1, state.fps, pre)
	return finderFamilyResult{
		fps: state.fps, candidates: candidates, alternatives: alternatives, channels: d.Ch,
		status: status, corner: scan.Corner, printDetected: d.printPass, scan: len(stats.Scans) - 1,
	}
}

// estimateMissingPattern interpolates the position of the single missing finder
// pattern from the other three and confirms it against the image, reporting
// where the corner it leaves behind came from. ok is false when the estimate
// falls outside the image.
//
// Confirmation prefers a candidate the scan already found over a fresh search:
// pool is the family's candidate union over the directions and passes run so
// far, so it holds corners this direction's own selection never saw, and a real
// detection beats re-searching a box in image rows around the estimate - which
// is what seekMissingFinderPattern does, and why it cannot confirm a corner on
// an obliquely captured symbol whose rows cross data quadrants.
// balanced supplies the image only for the seek, which is the last of four
// outcomes and the only one that reads a pixel. Asking for it up front would
// download a whole resident frame for an interpolation that usually never
// looks at one, and a nil return simply leaves the corner constructed.
func estimateMissingPattern(balanced func() *core.Bitmap, ch [3]*core.Bitmap, fps, pool []FinderPattern) (CornerSource, int, bool) {
	miss, missing := interpolateMissingPattern(fps)
	if !missing {
		return CornerFound, -1, false
	}
	if fps[miss].Center.X < 0 || fps[miss].Center.X > float64(ch[0].Width-1) ||
		fps[miss].Center.Y < 0 || fps[miss].Center.Y > float64(ch[0].Height-1) {
		fps[miss].FoundCount = 0
		return CornerConstructed, miss, false
	}
	if c, ok := pickPooledCorner(pool, fps, miss); ok {
		fps[miss] = c
		return CornerPooled, miss, true
	}
	if bm := balanced(); bm != nil && seekMissingFinderPattern(bm, fps, miss) {
		return CornerSought, miss, true
	}
	return CornerConstructed, miss, true
}

func interpolateMissingPattern(fps []FinderPattern) (int, bool) {
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
	if miss < 0 {
		return 0, false
	}
	return miss, true
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
			d.seedModules.add(ps.ModuleSize)
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
			if (fp.Typ == fp1 || fp.Typ == fp2) &&
				(!d.ensureBitmap() || !finderPatternHasColorSignal(d.BM, fp, newScanDirection(90))) {
				continue
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
