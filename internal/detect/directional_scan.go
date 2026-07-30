package detect

import (
	"math"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/spec"
)

// The directional family scan: the same traversal, classification and
// cross-check chain as the row walk in findprimary.go, with the row replaced by
// a line at an arbitrary direction. One prepared frame serves every direction,
// which is what makes it a replacement for the rotation ladder rather than a
// reordering of it.

// clipScanLine restricts a scan line to the frame, returning the first in-frame
// sample index and how many samples follow it. Without this a sweep spends most
// of its samples outside the frame, because a line's perpendicular offset says
// nothing about where along it the frame begins.
func clipScanLine(w, h int, p0 core.PointF, d scanDirection) (start, count int, ok bool) {
	lo, hi := math.Inf(-1), math.Inf(1)
	// Each axis contributes the interval of i keeping that coordinate in range;
	// the walk is the intersection. A zero component means the coordinate never
	// moves, so it either holds throughout or the line misses the frame.
	clip := func(p, step, limit float64) bool {
		if step == 0 {
			return p >= 0 && p <= limit
		}
		t0, t1 := -p/step, (limit-p)/step
		if t0 > t1 {
			t0, t1 = t1, t0
		}
		lo, hi = math.Max(lo, t0), math.Min(hi, t1)
		return true
	}
	if !clip(p0.X, d.dx, float64(w-1)) || !clip(p0.Y, d.dy, float64(h-1)) {
		return 0, 0, false
	}
	start = int(math.Ceil(lo))
	end := int(math.Floor(hi))
	if end < start {
		return 0, 0, false
	}
	return start, end - start + 1, true
}

// scanDirectionalFamily walks every line at direction base, spaced step apart
// perpendicular to it, and runs the per-hit chain on each raw green signature.
// It is scanCurrentFamilyRow generalized from a row to a line.
func (d *PrimaryDetector) scanDirectionalFamily(base scanDirection, step int, state *primaryFamilyScan) {
	d.sweepDirection(d.Ch[1], base, step, state, d.processDirectionalFamilyHit)
}

// sweepDirection covers the frame with lines at base and reports every raw
// run-length signature in seek to onHit. The seek channel is the family's own:
// green for the current signature, red for the BSI-era one, exactly as their
// row walks choose it.
func (d *PrimaryDetector) sweepDirection(
	seek *core.Bitmap,
	base scanDirection,
	step int,
	state *primaryFamilyScan,
	onHit func(base scanDirection, centre core.PointF, moduleSize float64, state *primaryFamilyScan),
) {
	ch := d.Ch
	w, h := ch[0].Width, ch[0].Height

	// Lines are indexed by their signed perpendicular offset from the origin.
	// The frame's four corners bound that offset, so the sweep covers the frame
	// exactly rather than a bounding disc.
	perp := base.perpendicular()
	nx, ny := perp.dx/perp.pxPerSample, perp.dy/perp.pxPerSample
	qLo, qHi := math.Inf(1), math.Inf(-1)
	for _, c := range [4][2]float64{{0, 0}, {float64(w - 1), 0}, {0, float64(h - 1)}, {float64(w - 1), float64(h - 1)}} {
		q := c[0]*nx + c[1]*ny
		qLo, qHi = math.Min(qLo, q), math.Max(qHi, q)
	}

	for q := qLo; q <= qHi; q += float64(step) {
		if d.Quitting() {
			return
		}
		p0 := core.PointF{X: q * nx, Y: q * ny}
		start, count, ok := clipScanLine(w, h, p0, base)
		if !ok {
			continue
		}
		for count > 0 {
			centre, moduleSize, next, hit := seekPatternAlong(seek, base, p0.X, p0.Y, start, count)
			if !hit {
				break
			}
			count -= next - start
			start = next
			onHit(base, centre, moduleSize, state)
			if state.done {
				return
			}
		}
	}
}

// processDirectionalFamilyHit runs the cross-check and classification chain of
// one raw n-1-1-1-m green signature found along base, saving a surviving finder
// pattern into state. It is processCurrentFamilyHit with the row index and row
// slices replaced by a point and the scan basis.
func (d *PrimaryDetector) processDirectionalFamilyHit(base scanDirection, centre core.PointF, moduleG float64, state *primaryFamilyScan) {
	ch := d.Ch
	d.pass().RawHits++
	d.seedModules = append(d.seedModules, moduleG)

	gx, gy := int(centre.X), int(centre.Y)
	if gx < 0 || gx >= ch[1].Width || gy < 0 || gy >= ch[1].Height {
		return
	}
	typeG := core.BoolColor(ch[1].Pix[gy*ch[1].Width+gx] > 0)
	centreR, centreB := centre, centre
	var typeR, typeB int
	var moduleR, moduleB float64
	blueBranch, redBranch := false, false
	slack := d.ccSlack(moduleG)

	var branchWalk walkWindow
	// branchWalkPtr stays nil unless a trace is attached, so an untraced read is
	// unchanged.
	var branchWalkPtr *walkWindow
	if d.Trace != nil {
		branchWalkPtr = &branchWalk
	}
	// colourCh records the channel the core-colour check was made on, which is
	// the *other* channel from the pattern walk that confirmed: the blue branch
	// tests the red core, the red branch tests the blue one. Recording the
	// pattern channel here would name a walk that passed as the rejecter.
	colourCh := -1
	// The branch order is the row walk's: blue first, red second, and a
	// candidate needs the green signature plus one of the two.
	if crossCheckPatternAlong(ch[2], base, moduleG*2, &centreB, &moduleB, slack, branchWalkPtr) {
		d.pass().BranchBlue++
		colourCh = 0
		if !inFrame(ch[2], centreB) {
			return
		}
		typeB = core.BoolColor(ch[2].Pix[int(centreB.Y)*ch[2].Width+int(centreB.X)] > 0)
		moduleR = moduleG
		coreRed := int(palette.Default[spec.FP3CoreColor*3])
		if crossCheckColorAlong(ch[0], coreRed, moduleR, 5, centreR, base, slack) {
			typeR = 0
			blueBranch = true
		}
	} else if crossCheckPatternAlong(ch[0], base, moduleG*2, &centreR, &moduleR, slack, branchWalkPtr) {
		d.pass().BranchRed++
		colourCh = 2
		if !inFrame(ch[0], centreR) {
			return
		}
		typeR = core.BoolColor(ch[0].Pix[int(centreR.Y)*ch[0].Width+int(centreR.X)] > 0)
		moduleB = moduleG
		coreBlue := int(palette.Default[spec.FP2CoreColor*3+2])
		if crossCheckColorAlong(ch[2], coreBlue, moduleB, 5, centreB, base, slack) {
			typeB = 0
			redBranch = true
			d.pass().RedColor++
		}
	}

	passIndex := len(d.Stats.Passes) - 1
	// The branch walks all start from the seek centre, so here the walk start and
	// the candidate centre coincide and WalkDeg is the base direction.
	reject := func(stage FinderStage, typ, channel int, w walkWindow) {
		d.Trace.reject(ch[1], FinderRejection{
			Stage: stage, Pass: passIndex, Typ: typ, Channel: channel,
			BaseDeg: base.deg, WalkDeg: base.deg,
			Centre: centre, Module: moduleG, Runs: w.runs, Reason: w.reason,
		})
	}
	if !(blueBranch || redBranch) {
		// A confirmed pattern walk means the core-colour check is what rejected
		// this candidate, and the run window belongs to a walk that passed.
		if colourCh >= 0 {
			reject(StageBranchColor, -1, colourCh, walkWindow{})
		} else {
			// Both walks failed, and the window that survives is always the red
			// one's: blue runs first and red only runs because blue failed.
			reject(StageBranchPattern, -1, 0, branchWalk)
		}
		return
	}
	fp := FinderPattern{FoundCount: 1}
	if blueBranch {
		if !checkModuleSize2(moduleG, moduleB) {
			reject(StageBranchModuleSize, -1, -1, walkWindow{})
			return
		}
		fp.Center = midpoint(centre, centreB)
		fp.ModuleSize = (moduleG + moduleB) / 2
		if !fp.classify([]int{fp0, fp3}, typeR, typeG, typeB) {
			reject(StageClassify, -1, -1, walkWindow{})
			return
		}
	} else {
		if !checkModuleSize2(moduleR, moduleG) {
			reject(StageBranchModuleSize, -1, -1, walkWindow{})
			return
		}
		fp.Center = midpoint(centreR, centre)
		fp.ModuleSize = (moduleR + moduleG) / 2
		if !fp.classify([]int{fp1, fp2}, typeR, typeG, typeB) {
			reject(StageClassify, -1, -1, walkWindow{})
			return
		}
		d.pass().RedClassified++
	}
	if fp.Typ == fp1 || fp.Typ == fp2 {
		if !d.ensureBitmap() {
			reject(StageChainSignal, fp.Typ, -1, walkWindow{})
			return
		}
		if channel, ok := finderPatternColorSignal(d.BM, fp, base); !ok {
			reject(StageChainSignal, fp.Typ, channel, walkWindow{})
			return
		}
	}
	seed := fp
	if crossCheckPatternAlongCh(ch, &fp, base, d.ccSlack(fp.ModuleSize), d.Trace, passIndex) {
		d.pass().CrossSurvivors[fp.Typ]++
		saveFinderPattern(&fp, state.fps, &state.total, state.typeCount[:])
		if state.total >= maxFinderPatterns-1 {
			state.done = true
		}
	} else {
		state.retainContextualSeed(seed)
	}
}

func inFrame(img *core.Bitmap, p core.PointF) bool {
	x, y := int(p.X), int(p.Y)
	return x >= 0 && x < img.Width && y >= 0 && y < img.Height
}

func midpoint(a, b core.PointF) core.PointF {
	return core.PointF{X: (a.X + b.X) / 2, Y: (a.Y + b.Y) / 2}
}
