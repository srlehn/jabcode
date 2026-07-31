package detect

import (
	"math"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/spec"
)

// Directional finder scanning: reading a rotated symbol by turning the scan
// line rather than the image.
//
// The axis-aligned walks in finderpattern.go can only see a finder whose
// symbol axis is within roughly ten degrees of an image axis, because beyond
// that a straight scan through the core drifts out of the core's own module row
// and starts crossing the two quadrants that are data rather than pattern (the
// finder is two square references joined at a core, not concentric rings; see
// the fp0..fp3 block). Pre-rotating the whole frame fixes that by moving the
// pixels. Scanning along the symbol's own direction fixes it without touching
// them, and one prepared frame then serves every direction.
//
// A scan line is undirected, so directions fold modulo 180. The chain then
// walks a basis of two perpendicular module axes, base and its perpendicular,
// so a probe at d covers the axis pair {d, d+90} and [0,90) suffices. That is a
// property of the axis pair, not of any one finder: an individual finder has no
// quarter-turn symmetry (see the fp0..fp3 block). scanDirections tiles the range
// at a 15-degree step: a 7.5-degree worst-case residual sits inside the measured
// survival band, while the 11.25 degrees a four-direction set would leave does
// not.

// scanDirection is one probe direction. The walk advances its major axis by one
// pixel per sample, which keeps the layer-crossing ratios of the finder
// signature exact - every layer boundary scales by the same factor - at the
// cost of sampling the line more coarsely the further it tilts.
type scanDirection struct {
	deg    float64 // the direction itself, kept so the basis turns are exact
	dx, dy float64 // unit step along the major axis, one component being +/-1

	// pxPerSample is the physical distance one sample covers, 1/max(|cos|,|sin|).
	// Run lengths are measured in samples, so converting by this factor is what
	// makes a directional module size comparable with an axis-aligned one
	// rather than up to 41 percent short.
	pxPerSample float64
}

// newScanDirection builds the walk parameters for a scan at angleDeg.
func newScanDirection(angleDeg float64) scanDirection {
	a := angleDeg * math.Pi / 180
	c, s := math.Cos(a), math.Sin(a)
	major := math.Max(math.Abs(c), math.Abs(s))
	return scanDirection{deg: angleDeg, dx: c / major, dy: s / major, pxPerSample: 1 / major}
}

// scanDirectionFromVector builds the walk parameters for the direction of
// (dx, dy), which need not be unit length. It is what a basis measured from
// located pattern positions needs, where the direction arrives as a vector
// difference and forcing it through an angle constant would only lose
// precision.
func scanDirectionFromVector(dx, dy float64) scanDirection {
	return newScanDirection(math.Atan2(dy, dx) * 180 / math.Pi)
}

// unit returns the direction as a unit vector. The walk step is normalized on
// its major axis instead, so it is pxPerSample times as long as this.
func (d scanDirection) unit() (x, y float64) {
	return d.dx / d.pxPerSample, d.dy / d.pxPerSample
}

// turn returns the direction rotated by delta degrees. The cross-check basis is
// built from turns of the scan direction rather than from image axes, which is
// the whole substitution: perpendicular replaces vertical and the two 45-degree
// turns replace the image diagonals.
func (d scanDirection) turn(delta float64) scanDirection {
	return newScanDirection(d.deg + delta)
}

// perpendicular returns the direction turned a quarter turn, which is the
// directional counterpart of the vertical cross-check.
func (d scanDirection) perpendicular() scanDirection { return d.turn(90) }

// scanDirections are the probe directions covering [0,90). Kept at six for the
// residual-angle reason above; reducing the count is not where the saving is,
// since all of them read one prepared frame.
var scanDirections = []float64{0, 15, 30, 45, 60, 75}

const finderMinChannelContrast = 0.1

// finderPatternColorSignal verifies that an FP1/FP2 mask signature represents
// a source-level yellow/black transition in both colour-bearing channels. The
// palette classifier gives yellow and black identical red and green masks, so
// walking both masks repeats one decision made by the shared black gate. Signed
// Michelson contrast over the expected five-module band restores two
// independent source-channel observations without introducing a pixel-scale
// threshold.
func finderPatternColorSignal(bm *core.Bitmap, fp FinderPattern, base scanDirection) (int, bool) {
	if fp.Typ != fp1 && fp.Typ != fp2 {
		return -1, true
	}
	if bm == nil || bm.Width <= 0 || bm.Height <= 0 || bm.Channels < 3 ||
		len(bm.Pix) < bm.Width*bm.Height*bm.Channels || fp.ModuleSize <= 0 {
		return -1, false
	}

	samples := max(5, int(math.Ceil(5*fp.ModuleSize/base.pxPerSample)))
	coreBit := 0
	if fp.Typ == fp2 {
		coreBit = 1
	}
	var sums [2][2]float64
	var counts [2]int
	for i := range samples {
		offset := (float64(i)+0.5)/float64(samples)*5 - 2.5
		bit := coreBit
		if distance := math.Abs(offset); distance >= 0.5 && distance < 1.5 {
			bit = 1 - bit
		}
		x := int(fp.Center.X + offset*fp.ModuleSize*base.dx/base.pxPerSample)
		y := int(fp.Center.Y + offset*fp.ModuleSize*base.dy/base.pxPerSample)
		if x < 0 || x >= bm.Width || y < 0 || y >= bm.Height {
			continue
		}
		pixel := (y*bm.Width + x) * bm.Channels
		for channel := range 2 {
			sums[channel][bit] += float64(bm.Pix[pixel+channel])
		}
		counts[bit]++
	}
	if counts[0] == 0 || counts[1] == 0 {
		return -1, false
	}
	for channel := range 2 {
		black := sums[channel][0] / float64(counts[0])
		yellow := sums[channel][1] / float64(counts[1])
		if (yellow-black)/max(yellow+black, 1) < finderMinChannelContrast {
			return channel, false
		}
	}
	return -1, true
}

func finderPatternHasColorSignal(bm *core.Bitmap, fp FinderPattern, base scanDirection) bool {
	_, ok := finderPatternColorSignal(bm, fp, base)
	return ok
}

// crossCheckPatternAlong validates a finder candidate along an arbitrary
// direction and refines its centre along that direction, reporting whether the
// five-run signature holds. It is the directional counterpart of
// crossCheckPatternHorizontal.
//
// Two differences from that function are forced rather than chosen. The word
// scans in runlength.go read eight contiguous bytes, which a strided walk
// cannot use, so runs are measured a sample at a time. And the module size is
// converted from samples to physical pixels before returning, because every
// consumer of ModuleSize expects the true module size.
//
// centre and moduleSize are outputs of a positive verdict only, matching the
// axis-aligned walks.
func crossCheckPatternAlong(img *core.Bitmap, dir scanDirection, moduleSizeMax float64, centre *core.PointF, moduleSize *float64, slack int, w *walkWindow) bool {
	return crossCheckAlong(img, dir, moduleSizeMax, dir.pxPerSample, centre, moduleSize, slack, w)
}

// walkWindow is what a cross-check walk leaves behind for a trace: the five-run
// window it ended on and why it stopped. A rejection is only interpretable
// against both, and neither can be recovered afterwards from the centre alone.
type walkWindow struct {
	runs   [5]int
	reason WalkReject
}

// fail records why the walk stopped and returns its verdict, keeping every
// early exit a single return. A nil receiver is the untraced path.
func (w *walkWindow) fail(r WalkReject) bool {
	if w != nil {
		w.reason = r
	}
	return false
}

// crossCheckAlong is crossCheckPatternAlong with the sample-to-pixel factor
// supplied, because the diagonal walks do not use dir.pxPerSample: a 45-degree
// line crosses a module over sqrt(2) times its width, so the extra distance and
// the coarser sample cancel and the factor is pxPerSample/sqrt(2). At an
// upright scan that is exactly 1, which is why the axis-aligned diagonal walk
// needs no conversion at all.
// w, when non-nil, receives the five-run window the walk ended on and why it
// stopped, whatever the verdict.
func crossCheckAlong(img *core.Bitmap, dir scanDirection, moduleSizeMax, pxPerRun float64, centre *core.PointF, moduleSize *float64, slack int, w *walkWindow) bool {
	const stateMiddle = 2
	// The walk counts straight into the caller's window when one is supplied, so
	// every early return leaves the runs behind without a deferred copy. w is
	// nil unless a trace is attached, so the untraced path keeps counting in a
	// plain local on a walk taken millions of times per read.
	var local [5]int
	stateCount := &local
	if w != nil {
		*w = walkWindow{}
		stateCount = &w.runs
	}

	sx := *centre
	x0, y0 := int(sx.X), int(sx.Y)
	if x0 < 0 || x0 >= img.Width || y0 < 0 || y0 >= img.Height {
		return w.fail(WalkIncomplete)
	}

	// The same bound the axis-aligned walks take: the module size is one third
	// of the three middle state counts, those counts never shrink, so once
	// their sum passes the limit no continuation recovers the candidate. In
	// samples rather than pixels, since that is what the counts are in.
	insideLimit := img.Width + img.Height + 1
	if lim := moduleSizeMax * 3 / pxPerRun; lim >= 0 && lim < float64(insideLimit) {
		insideLimit = int(lim) + 2
	}
	inside := 1

	at := func(i int, sign float64) (byte, bool) {
		x := int(sx.X + sign*float64(i)*dir.dx)
		y := int(sx.Y + sign*float64(i)*dir.dy)
		if x < 0 || x >= img.Width || y < 0 || y >= img.Height {
			return 0, false
		}
		return img.Pix[y*img.Width+x], true
	}

	stateCount[stateMiddle]++
	prev := img.Pix[y0*img.Width+x0]
	stateIndex := 0
	for i := 1; stateIndex <= stateMiddle; i++ {
		cur, ok := at(i, -1)
		if !ok {
			break
		}
		if cur == prev {
			state := stateMiddle - stateIndex
			stateCount[state]++
			if state > 0 {
				if inside++; inside >= insideLimit {
					return w.fail(WalkTooWide)
				}
			}
			continue
		}
		if stateIndex > 0 && stateCount[stateMiddle-stateIndex] < slack {
			stateCount[stateMiddle-(stateIndex-1)] += stateCount[stateMiddle-stateIndex]
			stateCount[stateMiddle-stateIndex] = 0
			stateIndex--
			stateCount[stateMiddle-stateIndex]++
			if inside = stateCount[1] + stateCount[2] + stateCount[3]; inside >= insideLimit {
				return w.fail(WalkTooWide)
			}
		} else {
			stateIndex++
			if stateIndex > stateMiddle {
				break
			}
			stateCount[stateMiddle-stateIndex]++
			if stateMiddle-stateIndex > 0 {
				if inside++; inside >= insideLimit {
					return w.fail(WalkTooWide)
				}
			}
		}
		prev = cur
	}
	if stateIndex < stateMiddle {
		return w.fail(WalkIncomplete)
	}

	stateIndex = 0
	prev = img.Pix[y0*img.Width+x0]
	last := 0
	for i := 1; stateIndex <= stateMiddle; i++ {
		cur, ok := at(i, +1)
		if !ok {
			break
		}
		last = i
		if cur == prev {
			state := stateMiddle + stateIndex
			stateCount[state]++
			if state < 4 {
				if inside++; inside >= insideLimit {
					return w.fail(WalkTooWide)
				}
			}
			continue
		}
		if stateIndex > 0 && stateCount[stateMiddle+stateIndex] < slack {
			stateCount[stateMiddle+(stateIndex-1)] += stateCount[stateMiddle+stateIndex]
			stateCount[stateMiddle+stateIndex] = 0
			stateIndex--
			stateCount[stateMiddle+stateIndex]++
			if inside = stateCount[1] + stateCount[2] + stateCount[3]; inside >= insideLimit {
				return w.fail(WalkTooWide)
			}
		} else {
			stateIndex++
			if stateIndex > stateMiddle {
				break
			}
			stateCount[stateMiddle+stateIndex]++
			if stateMiddle+stateIndex < 4 {
				if inside++; inside >= insideLimit {
					return w.fail(WalkTooWide)
				}
			}
		}
		prev = cur
	}
	if stateIndex < stateMiddle {
		return w.fail(WalkIncomplete)
	}

	ms, ok := checkPatternCross(*stateCount)
	if !ok {
		return w.fail(WalkSignature)
	}
	ms *= pxPerRun
	if ms > moduleSizeMax {
		return w.fail(WalkModuleSize)
	}
	*moduleSize = ms
	// The refined centre is the midpoint of the middle run, expressed back in
	// image coordinates along the walk direction.
	offset := float64(last-stateCount[4]-stateCount[3]) - float64(stateCount[2])/2.0
	centre.X = sx.X + offset*dir.dx
	centre.Y = sx.Y + offset*dir.dy
	return true
}

// seekPatternAlong scans one line at direction dir, passing through (x0,y0), for
// the next finder run-length signature. It is the directional counterpart of
// seekPatternHorizontal: same five-state window and the same checkPatternCross
// acceptance, walking a line rather than a contiguous row slice.
//
// Sample indices run from start for at most limit steps. A line generally meets
// the frame partway along its span, so leading samples outside the frame are
// skipped rather than ending the scan; the walk ends at the first sample that
// leaves the frame again. That lets a caller sweep lines by perpendicular
// offset without solving each one's entry point.
//
// It reports the signature's centre in image coordinates, its module size in
// physical pixels, and the sample index to resume from. That resume point is
// the matched window's start plus its first run, not the window's end: finder
// signatures overlap along a line, and resuming past the whole window drops
// every one that begins inside the one just matched.
func seekPatternAlong(img *core.Bitmap, dir scanDirection, x0, y0 float64, start, limit int) (centre core.PointF, moduleSize float64, next int, ok bool) {
	const stateNumber = 5
	curState := 0
	var stateCount [5]int

	sample := func(i int) (byte, bool) {
		x := int(x0 + float64(i)*dir.dx)
		y := int(y0 + float64(i)*dir.dy)
		if x < 0 || x >= img.Width || y < 0 || y >= img.Height {
			return 0, false
		}
		return img.Pix[y*img.Width+x], true
	}

	end := start + limit
	first := start
	var prev byte
	entered := false
	for ; first < end; first++ {
		v, inFrame := sample(first)
		if inFrame {
			prev, entered = v, true
			break
		}
	}
	if !entered {
		return centre, 0, end, false
	}
	// resume tracks where the current five-run window begins, which is what the
	// caller must step past. It moves with the window, not with the walk.
	resume := first
	stateCount[curState]++
	for i := first + 1; i < end; i++ {
		cur, inFrame := sample(i)
		if !inFrame {
			return centre, 0, i, false
		}
		// The last in-frame sample closes the window it is in, so a signature
		// whose final run reaches the frame edge is still matched.
		last := i == end-1
		same := cur == prev
		if same {
			stateCount[curState]++
			if !last {
				continue
			}
		}
		prev = cur
		if curState < stateNumber-1 {
			if stateCount[curState] < 3 {
				if curState == 0 {
					stateCount[curState] = 1
					resume = i
				} else {
					stateCount[curState-1] += stateCount[curState]
					stateCount[curState] = 0
					curState--
					stateCount[curState]++
				}
			} else {
				curState++
				stateCount[curState]++
			}
			continue
		}
		if stateCount[curState] < 3 {
			stateCount[curState-1] += stateCount[curState]
			stateCount[curState] = 0
			curState--
			stateCount[curState]++
			continue
		}
		if ms, hit := checkPatternCross(stateCount); hit {
			endPos := i
			if last && same {
				endPos = i + 1
			}
			offset := float64(endPos-stateCount[4]-stateCount[3]) - float64(stateCount[2])/2.0
			centre = core.PointF{X: x0 + offset*dir.dx, Y: y0 + offset*dir.dy}
			return centre, ms * dir.pxPerSample, max(resume+stateCount[0], first+1), true
		}
		resume += stateCount[0]
		for k := range stateNumber - 1 {
			stateCount[k] = stateCount[k+1]
		}
		stateCount[stateNumber-1] = 1
		curState = stateNumber - 1
	}
	return centre, 0, end, false
}

// diagPxPerRun is the sample-to-pixel factor for a diagonal of the scan basis.
func diagPxPerRun(d scanDirection) float64 { return d.pxPerSample / math.Sqrt2 }

// crossCheckPatternDiagonalAlong validates a candidate along the two 45-degree
// turns of the scan basis, the directional counterpart of
// crossCheckPatternDiagonal.
//
// Which turn is tried first is not arbitrary and not a search order: the finder
// is two squares joined at their corner along one diagonal, so the five-run
// signature exists along that diagonal and not the other. fp0 and fp1 are joined
// along the +45 turn, fp2 and fp3 along the -45 turn. dir carries the confirmed
// turn out as +1, -1, or 2 when both hold.
func crossCheckPatternDiagonalAlong(img *core.Bitmap, typ int, base scanDirection, moduleSizeMax float64,
	centre *core.PointF, moduleSize *float64, dir *int, bothDir bool, slack int, fail *chainFail,
) int {
	var delta float64
	fixDir := false
	switch {
	case *dir != 0:
		if *dir > 0 {
			delta, *dir = 45, 1
		} else {
			delta, *dir = -45, -1
		}
		fixDir = true
	case typ == fp0 || typ == fp1:
		delta, *dir = 45, 1
	default:
		delta, *dir = -45, -1
	}

	// Two confirmations of the same turn, taken from the centre the first one
	// refined, is what the axis-aligned walk means by dcc==2, and the branch in
	// crossCheckPatternChAlong that accepts a candidate on diagonal evidence
	// alone is calibrated against that. Flipping to the other turn is the
	// response to a failure, not a second opinion.
	confirmed := 0
	tmpModuleSize := 0.0
	for tryCount := 1; ; tryCount++ {
		turn := base.turn(delta)
		probe := *centre
		start := probe
		var ms float64
		var walk walkWindow
		var walkPtr *walkWindow
		if fail != nil {
			walkPtr = &walk
		}
		ok := crossCheckAlong(img, turn, moduleSizeMax, diagPxPerRun(turn), &probe, &ms, slack, walkPtr)
		// Only a failing walk may be reported. The shared buffer would otherwise
		// hand a rejection the window of the *other* turn that succeeded, since
		// dcc==1 is still a rejection when the perpendicular did not confirm.
		if !ok && fail != nil {
			fail.walk = walkRecord{walk: walk, centre: start, deg: turn.deg, valid: true}
		}
		if ok {
			if tmpModuleSize > 0 {
				ms = (ms + tmpModuleSize) / 2.0
			} else {
				tmpModuleSize = ms
			}
			*moduleSize = ms
			*centre = probe
			confirmed++
			if !bothDir || tryCount == 2 || fixDir {
				if confirmed == 2 {
					*dir = 2
				}
				return confirmed
			}
			continue
		}
		if tryCount == 2 || fixDir {
			return confirmed
		}
		delta, *dir = -delta, -(*dir)
	}
}

// crossCheckColorAlong verifies the finder core's colour over moduleNumber
// modules centred on centre and running along d. It is crossCheckColor with the
// walk direction supplied, so the same three checks can be made in the scan
// basis rather than on image axes.
func crossCheckColorAlong(img *core.Bitmap, colour int, moduleSize float64, moduleNumber int, centre core.PointF, d scanDirection, tol int) bool {
	n := int(moduleSize * float64(moduleNumber-1) / d.pxPerSample)
	if n <= 0 {
		return false
	}
	x0 := centre.X - float64(n)/2*d.dx
	y0 := centre.Y - float64(n)/2*d.dy
	unmatch := 0
	for i := range n {
		x, y := int(x0+float64(i)*d.dx), int(y0+float64(i)*d.dy)
		if x < 0 || x >= img.Width || y < 0 || y >= img.Height {
			break
		}
		// Pixel rather than a Pix index: the alignment path shares this walk and
		// runs against channel bitmaps that may read back lazily, with no plane
		// materialized at all.
		if int(img.Pixel(x, y)) != colour {
			unmatch++
		} else if unmatch <= tol {
			unmatch = 0
		}
		if unmatch > tol {
			return false
		}
	}
	return true
}

// crossCheckColorBasis runs the three core-colour walks of the scan basis, the
// directional counterpart of crossCheckColor's three directions.
func crossCheckColorBasis(img *core.Bitmap, colour int, moduleSize float64, moduleNumber int, centre core.PointF, base scanDirection, tol int) bool {
	if !crossCheckColorAlong(img, colour, moduleSize, moduleNumber, centre, base, tol) ||
		!crossCheckColorAlong(img, colour, moduleSize, moduleNumber, centre, base.perpendicular(), tol) {
		return false
	}
	// The core is where the two square references overlap, so only one of the
	// two 45-degree turns runs along the join and stays inside the pattern;
	// which one depends on the finder type. The axis-aligned check accepts
	// either for the same reason, so this is not a widened tolerance.
	return crossCheckColorAlong(img, colour, moduleSize, moduleNumber, centre, base.turn(45), tol) ||
		crossCheckColorAlong(img, colour, moduleSize, moduleNumber, centre, base.turn(-45), tol)
}

// crossCheckPatternChAlong validates a candidate in one channel across the
// three directions of the scan basis. It is crossCheckPatternCh's hv==0 branch
// with perpendicular in place of vertical and along in place of horizontal;
// there is no hv==1 counterpart because a directional sweep has only one
// traversal direction, the scan's own.
func crossCheckPatternChAlong(ch *core.Bitmap, typ int, base scanDirection, moduleSizeMax float64,
	moduleSize *float64, centre *core.PointF, dir, dcc *int, slack int, fail *chainFail,
) bool {
	var msP, msA, msD float64
	var walk walkWindow
	// walkPtr stays nil unless a trace is attached, so an untraced read walks
	// exactly the code it walked before the diagnostic existed.
	var walkPtr *walkWindow
	if fail != nil {
		walkPtr = &walk
	}
	// The preceding walks move centre, so each walk's own start has to be taken
	// before it runs. Reporting the candidate's original centre would name a
	// position the failing walk never sampled, and at this scale one pixel
	// changes which lattice line the walk lands on.
	pcc := false
	if crossCheckPatternAlong(ch, base.perpendicular(), moduleSizeMax, centre, &msP, slack, walkPtr) {
		pcc = true
		start := *centre
		if !crossCheckPatternAlong(ch, base, moduleSizeMax, centre, &msA, slack, walkPtr) {
			fail.setWalk(StageChainBase, walkRecord{walk: walk, centre: start, deg: base.deg, valid: true}, 0)
			return false
		}
	}
	*dcc = crossCheckPatternDiagonalAlong(ch, typ, base, moduleSizeMax, centre, &msD, dir, !pcc, slack, fail)
	switch {
	case pcc && *dcc > 0:
		// Diagonal walks confirm the finder join. The two symbol axes define
		// module scale without importing the blurred join's shorter centre run.
		*moduleSize = (msP + msA) / 2.0
		return true
	case *dcc == 2:
		start := *centre
		if !crossCheckPatternAlong(ch, base, moduleSizeMax, centre, &msA, slack, walkPtr) {
			fail.setWalk(StageChainBase, walkRecord{walk: walk, centre: start, deg: base.deg, valid: true}, 0)
			return false
		}
		*moduleSize = msA
		return true
	}
	if fail != nil {
		fail.setWalk(StageChainDiagonal, fail.walk, *dcc)
	}
	return false
}

// walkRecord is one cross-check walk as it was actually performed: where it
// started, which direction it ran, and the window it ended on. A rejection is
// only interpretable against all three, because the walks refine centre in
// place and the diagonal turns away from the scan direction.
type walkRecord struct {
	walk   walkWindow
	centre core.PointF
	deg    float64
	valid  bool
}

// chainFail carries the stage and the failing walk out of a cross-check chain.
// A nil chainFail means no trace is attached and disables the recording.
//
// It is call-scoped rather than package state for reentrancy: the direction
// sweep itself is sequential, but several decode attempts can run at once, and
// a package-level flag has already produced one unattributable trace here.
type chainFail struct {
	stage    FinderStage
	walk     walkRecord
	confirms int
}

func (f *chainFail) setWalk(stage FinderStage, w walkRecord, confirms int) {
	if f == nil {
		return
	}
	f.stage, f.walk, f.confirms = stage, w, confirms
}

// crossCheckPatternAlongCh validates a finder candidate across the colour
// channels its type constrains and refines its centre, module size and
// direction. It is crossCheckPattern in the scan basis.
func crossCheckPatternAlongCh(ch [3]*core.Bitmap, fp *FinderPattern, base scanDirection, slack int, trace *DetectorTrace, pass int) bool {
	moduleSizeMax := fp.ModuleSize * 2.0
	var failStore chainFail
	var fail *chainFail
	if trace != nil {
		fail = &failStore
	}
	report := func(channel int) bool {
		r := FinderRejection{
			Stage: failStore.stage, Pass: pass, Typ: fp.Typ, Channel: channel,
			BaseDeg: base.deg, Centre: fp.Center, Module: fp.ModuleSize,
			Confirms: failStore.confirms,
		}
		// A walk-based stage reports where that walk started and which way it
		// ran; the module-size and colour stages did not walk, so they keep the
		// candidate centre and carry no window.
		if w := failStore.walk; w.valid {
			r.Centre, r.WalkDeg = w.centre, w.deg
			r.Runs, r.Reason = w.walk.runs, w.walk.reason
		} else {
			r.WalkDeg = base.deg
		}
		trace.reject(ch[1], r)
		return false
	}

	var msG float64
	centreG := fp.Center
	dirG, dccG := 0, 0
	if !crossCheckPatternChAlong(ch[1], fp.Typ, base, moduleSizeMax, &msG, &centreG, &dirG, &dccG, slack, fail) {
		return report(1)
	}

	if fp.Typ == fp1 || fp.Typ == fp2 {
		var msR float64
		centreR := fp.Center
		dirR, dccR := 0, 0
		if !crossCheckPatternChAlong(ch[0], fp.Typ, base, moduleSizeMax, &msR, &centreR, &dirR, &dccR, slack, fail) {
			return report(0)
		}
		if !checkModuleSize2(msR, msG) {
			fail.setWalk(StageChainModuleSize, walkRecord{}, 0)
			return report(-1)
		}
		fp.ModuleSize = (msR + msG) / 2.0
		fp.Center = core.PointF{X: (centreR.X + centreG.X) / 2.0, Y: (centreR.Y + centreG.Y) / 2.0}
		coreBlue := int(palette.Default[spec.FP2CoreColor*3+2])
		if !crossCheckColorBasis(ch[2], coreBlue, fp.ModuleSize, 5, fp.Center, base, slack) {
			fail.setWalk(StageChainColor, walkRecord{}, 0)
			return report(2)
		}
		fp.direction = diagonalDirection(dccR, dccG, dirR, dirG)
	}

	if fp.Typ == fp0 || fp.Typ == fp3 {
		var msB float64
		centreB := fp.Center
		dirB, dccB := 0, 0
		if !crossCheckPatternChAlong(ch[2], fp.Typ, base, moduleSizeMax, &msB, &centreB, &dirB, &dccB, slack, fail) {
			return report(2)
		}
		if !checkModuleSize2(msG, msB) {
			fail.setWalk(StageChainModuleSize, walkRecord{}, 0)
			return report(-1)
		}
		fp.ModuleSize = (msG + msB) / 2.0
		fp.Center = core.PointF{X: (centreG.X + centreB.X) / 2.0, Y: (centreG.Y + centreB.Y) / 2.0}
		coreRed := int(palette.Default[spec.FP3CoreColor*3])
		if !crossCheckColorBasis(ch[0], coreRed, fp.ModuleSize, 5, fp.Center, base, slack) {
			fail.setWalk(StageChainColor, walkRecord{}, 0)
			return report(0)
		}
		fp.direction = diagonalDirection(dccG, dccB, dirG, dirB)
	}
	return true
}

// diagonalDirection folds the two channels' diagonal verdicts into the sign the
// quad assembly reads, matching crossCheckPattern's choice.
func diagonalDirection(dccA, dccB, dirA, dirB int) int {
	switch {
	case dccA == 2 || dccB == 2:
		return 2
	case dirA+dirB > 0:
		return 1
	default:
		return -1
	}
}
