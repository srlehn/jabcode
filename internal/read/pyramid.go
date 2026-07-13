package read

import (
	"image"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/wire"
)

// minPyramidSide is the shorter-side floor for the coarsest pyramid level. It
// is a floor on the search schedule, not a measurement scale: below roughly
// this size a symbol occupying a typical fraction of the frame drops under the
// detector's minimum workable module size, so a coarser level could never
// decode anything a finer one cannot. Measured on a 12-megapixel phone
// capture: the 378 px shorter-side level decodes, the 189 px one fails.
const minPyramidSide = 300

// pyramidLevels builds the resolution pyramid Decode searches, coarsest level
// first, or nil when img cannot hold more than one level - the single-level
// path then runs the search directly on img, byte-identical to a pipeline
// without a pyramid.
func pyramidLevels(img image.Image) []*image.NRGBA {
	b := img.Bounds()
	if min(b.Dx(), b.Dy()) < 2*minPyramidSide {
		return nil
	}
	levels := []*image.NRGBA{pyramidBase(img)}
	for {
		last := levels[len(levels)-1]
		if min(last.Rect.Dx(), last.Rect.Dy()) < 2*minPyramidSide {
			break
		}
		levels = append(levels, detect.HalveNRGBA(last))
	}
	slices.Reverse(levels)
	return levels
}

// pyramidBase converts img once into the zero-origin NRGBA frame every level
// derives from - the one conversion all of a level's orientation rungs then
// share (rotatePrep aliases a zero-origin NRGBA instead of re-copying the
// canvas per rung). The pipeline never reads alpha, so it is forced opaque;
// that keeps later bitmap conversions of the base on the verbatim-copy route
// instead of re-premultiplying.
func pyramidBase(img image.Image) *image.NRGBA {
	bm := core.BitmapFromImage(img)
	core.ParallelRows(bm.Height, func(lo, hi int) {
		for i := lo*bm.Width*4 + 3; i < hi*bm.Width*4; i += 4 {
			bm.Pix[i] = 255
		}
	})
	return bm.NRGBA()
}

// decodePyramid searches the pyramid with one goroutine per level, each
// running the level's upright read and then, on failure with finder evidence,
// the level's orientation and region search. The coarsest level additionally
// publishes its detection finding (finder quad, side size, rung angle), and a
// seeded route resumes from that geometry on the finer levels without any
// finder search there (decodeSeeded) - so a capture the coarse route can
// locate never waits for the expensive blind full-resolution ladders.
//
// Results commit in a fixed priority order, never first-done: the coarsest
// upright, then the seeded route, then the finer uprights (coarsest first),
// then the searches (coarsest first). Every route's result is a pure function
// of the input - the seeded route reads only the coarsest level's
// deterministic finding - so the outcome is deterministic regardless of
// scheduling (the residual hazard of a miscorrected decode differing between
// routes is why the order is pinned). Uprights outrank the rest because they
// are the cheap bounded hypothesis; the seeded route outranks the blind
// ladders because its success carries either a cross-scale byte-for-byte
// agreement or the only decode of a geometry the blind ladders missed. Slots
// that can no longer win are told to quit at their next stage boundary and
// are not waited for - each route only touches its own data.
// On success it also reports the winning hypothesis - the shorter side of the
// level a full read succeeds on and the pre-rotation angle (0 for an upright
// win) - which a Stream replays as its first attempt on the next frame.
// Route attempts are collected into tr (nil to skip; see routeTrace for the
// per-slot collection and merge discipline).
func decodePyramid(levels []*image.NRGBA, tr *routeTrace) (data []byte, side int, deg float64, ok bool) {
	return decodePyramidProfile(levels, tr, wire.CReference)
}

func decodePyramidProfile(levels []*image.NRGBA, tr *routeTrace, profile wire.Profile) (data []byte, side int, deg float64, ok bool) {
	if tr != nil && tr.detailed {
		tr.pyramid = make([]image.Point, len(levels))
		tr.pyramidImages = make([]image.Image, len(levels))
		for i, level := range levels {
			tr.pyramid[i] = image.Pt(level.Rect.Dx(), level.Rect.Dy())
			tr.pyramidImages[i] = level
		}
	}
	type result struct {
		data []byte
		side int
		deg  float64
		ok   bool
	}
	// Slot 0 is the coarsest upright, slot 1 the seeded route, 2..n the finer
	// uprights, n+1..2n the searches (coarsest first).
	n := len(levels)
	uprightSlot := func(i int) int {
		if i == 0 {
			return 0
		}
		return i + 1
	}
	searchSlot := func(i int) int { return n + 1 + i }
	results := make([]result, 2*n+1)
	done := make([]chan struct{}, 2*n+1)
	for s := range done {
		done[s] = make(chan struct{})
	}
	// Tracing gives every route slot its own trace - each is written only by
	// the goroutine that owns the slot and read only after its done channel
	// closes - merged into tr in slot order below, so the collected order is
	// deterministic. A nil tr leaves every slot trace nil; add and merge are
	// nil-safe no-ops then.
	traces := make([]*routeTrace, 2*n+1)
	if tr != nil {
		traces[1] = &routeTrace{level: -1, detailed: tr.detailed}
		for i := range levels {
			traces[uprightSlot(i)] = &routeTrace{level: i, detailed: tr.detailed}
			traces[searchSlot(i)] = &routeTrace{level: i, detailed: tr.detailed}
		}
	}
	var winner atomic.Int64
	winner.Store(int64(len(done)))
	quit := func(slot int) func() bool {
		return func() bool { return winner.Load() < int64(slot) }
	}
	commit := func(slot int) {
		for {
			w := winner.Load()
			if int64(slot) >= w || winner.CompareAndSwap(w, int64(slot)) {
				return
			}
		}
	}

	// The coarsest level sends its finding exactly once - after its upright
	// read when that already settles it, otherwise after its search - and the
	// seeded route consumes it. A finding whose route decoded outranks a
	// locate-only one (decodeRetriesFinding); within one outcome class the
	// routes run sequentially in the level goroutine, so the choice is
	// deterministic.
	seed := make(chan finding, 1)

	// The promising orientation angles are scale-invariant, so the probe runs
	// once on the coarsest level and every level's search reuses the rungs -
	// deterministic because the input to the probe is fixed. A dense symbol
	// filling a large frame can starve the bounded probe of module pixels (a
	// 100-module symbol in a 12-megapixel frame holds ~3 px/module at the
	// default bound, below the cross-check floor, even though the symbol reads
	// fine once its orientation is known), so an empty result escalates: the
	// next finer level is probed under a doubled resolution bound, doubling
	// the per-module pixel budget each step, until a level retains an
	// orientation or the finest level found nothing. The escalation order is
	// fixed, so the shared result stays deterministic; the cost is bounded to
	// one probe per level, paid only on frames whose cheaper probes all came
	// up empty - frames that today fail outright. Escalated probes keep every
	// family passing the types floor instead of the top-2 cut: at their finer
	// resolutions texture inflates wrong-angle survivors, and on measured
	// captures the true family ranked third or fourth - the ladder, not the
	// probe amplitude, has to discriminate there.
	type sharedProbeResult struct {
		rungs  []float64
		probes []DiagnosticProbe
	}
	var sharedResult atomic.Pointer[sharedProbeResult]
	sharedRungs := sync.OnceValue(func() []float64 {
		if tr == nil || !tr.detailed {
			if rungs := detect.CoarseOrientationRungs(levels[0]); len(rungs) > 0 {
				return rungs
			}
			for k, lvl := range levels[1:] {
				fams := detect.CoarseProbeFamiliesWithin(lvl, detect.CoarseMaxDim<<(k+1))
				if rungs := detect.FamiliesToRungsUncapped(fams); len(rungs) > 0 {
					return rungs
				}
			}
			return nil
		}
		result := &sharedProbeResult{}
		defer sharedResult.Store(result)
		families, probe := detect.CoarseProbeFamiliesTraced(levels[0])
		result.rungs = detect.FamiliesToRungs(families)
		result.probes = append(result.probes, DiagnosticProbe{
			Level: 0, ROI: -1, Label: "pyramid shared level 0", Probe: probe,
			Rungs: append([]float64(nil), result.rungs...),
		})
		if len(result.rungs) > 0 {
			return result.rungs
		}
		for k, lvl := range levels[1:] {
			families, probe := detect.CoarseProbeFamiliesWithinTraced(lvl, detect.CoarseMaxDim<<(k+1))
			result.rungs = detect.FamiliesToRungsUncapped(families)
			result.probes = append(result.probes, DiagnosticProbe{
				Level: k + 1, ROI: -1, Label: "pyramid shared escalation", Probe: probe,
				Rungs: append([]float64(nil), result.rungs...),
			})
			if len(result.rungs) > 0 {
				return result.rungs
			}
		}
		return nil
	})
	mergeSharedProbe := func() {
		if tr == nil || !tr.detailed {
			return
		}
		if result := sharedResult.Load(); result != nil {
			tr.probes = append(tr.probes, result.probes...)
			sharedResult.Store(nil)
		}
	}

	for i := range levels {
		go func() {
			us := uprightSlot(i)
			fp := &finding{}
			detail := traces[us].beginAttempt("upright", 0, -1)
			data, stage, evidence := decodeBitmapFindingTracedProfile(core.BitmapFromImage(levels[i]), quit(us), fp, detail, profile)
			ok := stage == readDecoded
			traces[us].finishAttempt(routeAttempt{deg: 0, roi: -1, stage: stage, side: fp.side}, detail, data)
			if ok {
				commit(us)
			}
			results[us] = result{data, shorterSide(levels[i]), 0, ok}
			close(done[us])
			ss := searchSlot(i)
			if ok || !evidence || quit(ss)() {
				if i == 0 {
					seed <- *fp
				}
				close(done[ss])
				return
			}
			// An empty shared result is final: the escalation already probed
			// every level, each under a bound no coarser than the old
			// per-level fallback would have used, so a re-probe here could
			// not see more. The non-nil empty slice makes the search skip
			// straight to its region retries.
			rungs := sharedRungs()
			if rungs == nil {
				rungs = []float64{}
			}
			data, deg, ok := decodeRetriesFindingProfile(levels[i], quit(ss), fp, rungs, traces[ss], profile)
			if ok {
				commit(ss)
			}
			if i == 0 {
				seed <- *fp
			}
			results[ss] = result{data, shorterSide(levels[i]), deg, ok}
			close(done[ss])
		}()
	}
	go func() {
		f := <-seed
		if f.located && !quit(1)() {
			if data, side, ok := decodeSeededTracedProfile(levels, f, quit(1), traces[1], profile); ok {
				commit(1)
				results[1] = result{data, side, f.deg, true}
			}
		}
		close(done[1])
	}()

	for s := range done {
		<-done[s]
		tr.merge(traces[s])
		if r := results[s]; r.ok {
			mergeSharedProbe()
			return r.data, r.side, r.deg, true
		}
	}
	mergeSharedProbe()
	return nil, 0, 0, false
}
