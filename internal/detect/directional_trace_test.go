package detect

import (
	"math"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
)

// TestFinderStageNamesAreComplete keeps String in step with the stage list, so
// a stage added later cannot silently report as "unknown" in a funnel.
func TestFinderStageNamesAreComplete(t *testing.T) {
	for s := FinderStage(0); s < FinderStageCount; s++ {
		if s.String() == "unknown" {
			t.Errorf("stage %d has no name", s)
		}
	}
}

// TestRejectionRetentionSeparatesPasses pins the retention policy rather than a
// decode outcome: retries re-binarize the same frame, so samples from the first
// binarization must not exhaust the bucket a later pass would use. Everything
// else about the two rejections below is identical, which is exactly the case a
// pass-blind bucket collapses.
func TestRejectionRetentionSeparatesPasses(t *testing.T) {
	img := &core.Bitmap{Width: 640, Height: 480}
	var tr DetectorTrace
	sample := func(pass int) FinderRejection {
		return FinderRejection{
			Stage: StageChainDiagonal, Pass: pass, Typ: fp1, Channel: 1,
			BaseDeg: 45, WalkDeg: 90, Centre: core.PointF{X: 100, Y: 100}, Module: 9,
		}
	}
	for range maxRejectionSamplesPerCell * 4 {
		tr.reject(img, sample(0))
	}
	tr.reject(img, sample(1))

	if got := tr.RejectCounts[StageChainDiagonal]; got != maxRejectionSamplesPerCell*4+1 {
		t.Errorf("counted %d rejections, want %d", got, maxRejectionSamplesPerCell*4+1)
	}
	if len(tr.Rejections) != maxRejectionSamplesPerCell+1 {
		t.Fatalf("retained %d samples, want %d", len(tr.Rejections), maxRejectionSamplesPerCell+1)
	}
	if last := tr.Rejections[len(tr.Rejections)-1]; last.Pass != 1 {
		t.Errorf("the second pass was not retained: last sample is from pass %d", last.Pass)
	}
}

// TestRetainsRareCases pins the retention key on the distinctions a record is
// kept for. Common failures arrive first and in bulk, so a key blind to any of
// these lets them spend the bucket while the rare case at the same spot - the
// other diagonal turn, an oversized module, a single diagonal confirmation - is
// never sampled at all.
func TestRetainsRareCases(t *testing.T) {
	common := FinderRejection{
		Stage: StageChainDiagonal, Typ: fp1, Channel: 1,
		BaseDeg: 45, WalkDeg: 90, Centre: core.PointF{X: 100, Y: 100},
		Module: 9, Reason: WalkSignature,
	}
	for _, tc := range []struct {
		name string
		rare func(FinderRejection) FinderRejection
	}{
		{"the other turn", func(r FinderRejection) FinderRejection { r.WalkDeg = 0; return r }},
		{"an oversized module", func(r FinderRejection) FinderRejection { r.Reason = WalkModuleSize; return r }},
		{"one confirmation", func(r FinderRejection) FinderRejection { r.Confirms = 1; return r }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			img := &core.Bitmap{Width: 640, Height: 480}
			var tr DetectorTrace
			for range maxRejectionSamplesPerCell * 4 {
				tr.reject(img, common)
			}
			rare := tc.rare(common)
			tr.reject(img, rare)

			if len(tr.Rejections) != maxRejectionSamplesPerCell+1 {
				t.Fatalf("retained %d samples, want %d: the rare case shared the common one's bucket",
					len(tr.Rejections), maxRejectionSamplesPerCell+1)
			}
			if last := tr.Rejections[len(tr.Rejections)-1]; last != rare {
				t.Errorf("last retained sample is %+v, want %+v", last, rare)
			}
		})
	}
}

// TestRejectionRetentionSpreadsOverTheFrame guards the reason the coarse cell
// exists: rejections arrive in scan order, so a bucket that ignores position
// fills entirely from wherever the sweep starts.
func TestRejectionRetentionSpreadsOverTheFrame(t *testing.T) {
	img := &core.Bitmap{Width: 640, Height: 480}
	var tr DetectorTrace
	for x := range 8 {
		for range maxRejectionSamplesPerCell * 3 {
			tr.reject(img, FinderRejection{
				Stage: StageBranchPattern, Typ: -1, Channel: -1, BaseDeg: 0,
				Centre: core.PointF{X: float64(x) * 80, Y: 10},
			})
		}
	}
	cells := map[int]bool{}
	for _, r := range tr.Rejections {
		cells[cellIndex(r.Centre.X, img.Width)] = true
	}
	if len(cells) < 8 {
		t.Errorf("samples cover %d cells, want the 8 that were fed", len(cells))
	}
}

// TestRejectionRetentionIsHardBounded makes the ceiling a property of the type
// rather than of the caller, since a pathological frame produces rejections in
// the hundreds of thousands.
func TestRejectionRetentionIsHardBounded(t *testing.T) {
	img := &core.Bitmap{Width: 4096, Height: 4096}
	var tr DetectorTrace
	for i := range maxRejectionSamples * 2 {
		tr.reject(img, FinderRejection{
			Stage: StageClassify, Typ: -1, Channel: -1, BaseDeg: float64(i % 90),
			Centre: core.PointF{X: float64(i % 4096), Y: float64((i * 7) % 4096)},
		})
	}
	if len(tr.Rejections) > maxRejectionSamples {
		t.Errorf("retained %d samples, over the %d ceiling", len(tr.Rejections), maxRejectionSamples)
	}
	if tr.RejectCounts[StageClassify] != maxRejectionSamples*2 {
		t.Errorf("the ceiling suppressed counting: %d", tr.RejectCounts[StageClassify])
	}
}

// TestBranchRejectionAttribution is the attribution test an ordinary decode test
// cannot be: it runs the real directional scan over a real rendered symbol and
// checks that every recorded rejection describes the check that actually
// rejected it. A colour rejection carrying the run window of a pattern walk that
// passed is the specific defect being excluded.
func TestBranchRejectionAttribution(t *testing.T) {
	r := directionalTestSymbol(t)
	const modulePx = 12.0
	img, _ := renderRotatedRGBA(r, modulePx, 30*math.Pi/180)
	bm := core.BitmapFromImage(img)
	BalanceRGB(bm)
	ch := BinarizerRGB(bm, nil)

	var tr DetectorTrace
	d := &PrimaryDetector{BM: bm, Ch: ch, Mode: IntensiveDetect, Trace: &tr}
	d.Stats.Passes = append(d.Stats.Passes, FinderPassStats{})
	state := newPrimaryFamilyScan()
	d.scanDirectionalFamily(newScanDirection(30), 1, &state)

	if len(tr.Rejections) == 0 {
		t.Fatal("the scan recorded no rejections, so nothing is being attributed")
	}
	seen := map[FinderStage]int{}
	for _, rj := range tr.Rejections {
		seen[rj.Stage]++
	}
	for s := FinderStage(0); s < FinderStageCount; s++ {
		t.Logf("%-18s %d retained", s, seen[s])
	}
	// The reason histogram says which arms of checkWalkReason this fixture
	// actually exercises; TestWalkRejectReasons covers all of them by hand.
	reasons := map[WalkReject]int{}
	for _, rj := range tr.Rejections {
		reasons[rj.Reason]++
	}
	t.Logf("walk reasons: %v", reasons)
	// Without this the branch-color arm below is vacuous, and the defect it
	// guards against is precisely a colour rejection filed as a pattern one.
	if seen[StageBranchColor] == 0 {
		t.Error("no branch-color rejection was retained, so its attribution is untested")
	}
	for i, rj := range tr.Rejections {
		if rj.Pass != 0 {
			t.Errorf("sample %d: pass %d, want 0", i, rj.Pass)
		}
		if rj.Stage >= FinderStageCount {
			t.Errorf("sample %d: stage %d out of range", i, rj.Stage)
		}
		switch rj.Stage {
		case StageBranchColor:
			// The pattern walk passed, so its window says nothing about the
			// colour check that rejected the candidate.
			if rj.Runs != [5]int{} {
				t.Errorf("sample %d: branch-color carries runs %v from a walk that passed", i, rj.Runs)
			}
			// Accepting "0 or 2" would pass just as well with the two swapped,
			// which is the bug this guards. So check the record against the
			// image instead: whichever pattern walk confirms at this centre
			// identifies the branch, and the colour check always runs on the
			// other channel. Blue pattern tests the red core, red tests blue.
			base := newScanDirection(rj.BaseDeg)
			slack := d.ccSlack(rj.Module)
			probe, wantColour := rj.Centre, 2
			var ms float64
			if crossCheckPatternAlong(ch[2], base, rj.Module*2, &probe, &ms, slack, nil) {
				wantColour = 0
			}
			if rj.Channel != wantColour {
				t.Errorf("sample %d at (%.1f,%.1f): branch-color names channel %d, but the branch taken there checks the core colour on channel %d",
					i, rj.Centre.X, rj.Centre.Y, rj.Channel, wantColour)
			}
		case StageBranchPattern:
			// Blue is walked first and red only because blue failed, so the
			// window that survives to here is always red's.
			if rj.Channel != 0 {
				t.Errorf("sample %d: branch-pattern channel %d, want the red walk's 0", i, rj.Channel)
			}
			if rj.Runs == [5]int{} {
				t.Errorf("sample %d: branch-pattern kept no run window", i)
			}
			checkWalkReason(t, i, rj)
		case StageChainBase, StageChainDiagonal:
			if rj.Runs == [5]int{} {
				t.Errorf("sample %d: %s kept no run window", i, rj.Stage)
			}
			checkWalkReason(t, i, rj)
			// The decisive one. The chain refines the centre in place and the
			// diagonal turns away from the base, so replaying the walk the record
			// describes is the only thing that proves the record describes it: a
			// sample naming the candidate's own centre, or the scan direction
			// instead of the turn, replays into a different window. The two
			// diagonal turns also share a walk buffer, so this is equally what
			// excludes a rejection carrying the window of the turn that passed.
			if rj.Channel < 0 || rj.Channel > 2 {
				t.Errorf("sample %d: %s names channel %d, which was never walked", i, rj.Stage, rj.Channel)
				continue
			}
			dir := newScanDirection(rj.WalkDeg)
			pxPerRun := dir.pxPerSample
			if rj.Stage == StageChainDiagonal {
				pxPerRun = diagPxPerRun(dir)
			}
			probe := rj.Centre
			var ms float64
			var w walkWindow
			if crossCheckAlong(ch[rj.Channel], dir, rj.Module*2, pxPerRun, &probe, &ms, d.ccSlack(rj.Module), &w) {
				t.Errorf("sample %d: %s replays as a pass from (%.2f,%.2f) at %g degrees",
					i, rj.Stage, rj.Centre.X, rj.Centre.Y, rj.WalkDeg)
			}
			if w.runs != rj.Runs || w.reason != rj.Reason {
				t.Errorf("sample %d: %s at (%.2f,%.2f) deg %g replays as %v/%s, recorded %v/%s",
					i, rj.Stage, rj.Centre.X, rj.Centre.Y, rj.WalkDeg, w.runs, w.reason, rj.Runs, rj.Reason)
			}
			if rj.Stage == StageChainDiagonal && rj.Confirms >= 2 {
				t.Errorf("sample %d: chain-diagonal rejected with %d confirmations", i, rj.Confirms)
			}
		case StageBranchModuleSize, StageClassify, StageChainModuleSize, StageChainColor:
			if rj.Runs != [5]int{} {
				t.Errorf("sample %d: %s is not a walk but carries runs %v", i, rj.Stage, rj.Runs)
			}
			if rj.Reason != WalkNotWalked {
				t.Errorf("sample %d: %s did not walk but reports %s", i, rj.Stage, rj.Reason)
			}
		}
	}
}

// TestWalkRejectReasons separates the walk's four exits by hand, because a real
// frame cannot be relied on to produce all of them. The pair that matters is
// signature against module size: a window can satisfy checkPatternCross and
// still be rejected for the module size it implies, and a trace that merges the
// two reports "no finder here" about a finder that is there.
func TestWalkRejectReasons(t *testing.T) {
	// Five alternating runs of four, which checkPatternCross reads as module
	// size 4 exactly. The distorted band keeps the same extent but a middle run
	// no ratio test can accept.
	band := func(runs ...int) *core.Bitmap {
		bm := &core.Bitmap{Height: 1, Channels: 1}
		for i, n := range runs {
			for range n {
				bm.Pix = append(bm.Pix, byte(1-i%2))
			}
		}
		bm.Width = len(bm.Pix)
		return bm
	}
	even := band(4, 4, 4, 4, 4)
	skewed := band(4, 1, 12, 1, 4)
	flat := band(20)

	for _, tc := range []struct {
		name   string
		img    *core.Bitmap
		start  float64
		max    float64
		wantOK bool
		want   WalkReject
	}{
		{"accepted", even, 9, 8, true, WalkNotWalked},
		// The interesting bound. Below roughly 3.67 the same ceiling stops the
		// walk early as too-wide, so a size rejection needs a limit loose enough
		// to admit the runs and then refuse the module they imply.
		{"module size", even, 9, 3.7, false, WalkModuleSize},
		{"too wide", even, 9, 2, false, WalkTooWide},
		{"signature", skewed, 10, 8, false, WalkSignature},
		{"no window", flat, 9, 8, false, WalkIncomplete},
	} {
		t.Run(tc.name, func(t *testing.T) {
			centre := core.PointF{X: tc.start}
			var ms float64
			var w walkWindow
			ok := crossCheckAlong(tc.img, newScanDirection(0), tc.max, 1, &centre, &ms, 1, &w)
			if ok != tc.wantOK {
				t.Errorf("verdict %v, want %v (runs %v, ms %.2f)", ok, tc.wantOK, w.runs, ms)
			}
			if w.reason != tc.want {
				t.Errorf("reason %s, want %s (runs %v)", w.reason, tc.want, w.runs)
			}
		})
	}
}

// checkWalkReason requires the recorded reason to agree with the window it is
// filed against. It is what keeps the two late failures apart: a window can
// satisfy checkPatternCross and still be rejected for the module size it
// implies, and reading that record as "the ratios were wrong" is the mistake
// the funnel used to invite. The early exits leave a partial window whose
// ratios mean nothing either way, so only the two late reasons are constrained.
func checkWalkReason(t *testing.T, i int, rj FinderRejection) {
	t.Helper()
	if rj.Reason == WalkNotWalked {
		t.Errorf("sample %d: %s is a walk but records no reason for stopping", i, rj.Stage)
		return
	}
	_, cross := checkPatternCross(rj.Runs)
	if rj.Reason == WalkSignature && cross {
		t.Errorf("sample %d: %s blames the ratios, but %v passes checkPatternCross", i, rj.Stage, rj.Runs)
	}
	if rj.Reason == WalkModuleSize && !cross {
		t.Errorf("sample %d: %s blames the module size, but %v fails checkPatternCross first", i, rj.Stage, rj.Runs)
	}
}

// TestUntracedScanRecordsNothing pins the nil-trace path, which is the one every
// ordinary decode takes.
func TestUntracedScanRecordsNothing(t *testing.T) {
	r := directionalTestSymbol(t)
	const modulePx = 12.0
	img, _ := renderRotatedRGBA(r, modulePx, 30*math.Pi/180)
	bm := core.BitmapFromImage(img)
	BalanceRGB(bm)
	ch := BinarizerRGB(bm, nil)

	d := &PrimaryDetector{BM: bm, Ch: ch, Mode: IntensiveDetect}
	d.Stats.Passes = append(d.Stats.Passes, FinderPassStats{})
	state := newPrimaryFamilyScan()
	d.scanDirectionalFamily(newScanDirection(30), 1, &state)
	if state.total == 0 {
		t.Error("the untraced scan found nothing, so it is not exercising the same path")
	}
}
