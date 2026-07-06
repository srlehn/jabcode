package read

import (
	"image"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/detect"
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
// level a full read succeeds on, the pre-rotation angle (0 for an upright
// win), and the winning route's detection finding in frame coordinates -
// which a Stream replays as its first attempt on the next frame.
func decodePyramid(levels []*image.NRGBA) (data []byte, side int, deg float64, f finding, ok bool) {
	type result struct {
		data []byte
		side int
		deg  float64
		f    finding
		ok   bool
	}
	frame := levels[len(levels)-1].Rect
	// toFrame returns fp's finding normalized from lvl's coordinates to frame
	// coordinates - the one convention findings travel in between routes.
	toFrame := func(fp *finding, lvl *image.NRGBA) finding {
		nf := *fp
		if nf.located {
			nf.scale(float64(frame.Dx())/float64(lvl.Rect.Dx()), float64(frame.Dy())/float64(lvl.Rect.Dy()))
		}
		return nf
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
	// deterministic because the input to the probe is fixed. An empty shared
	// result makes each level fall back to its own probe: a family too weak
	// for the coarsest level's downscale chain may still surface in a finer
	// level's.
	sharedRungs := sync.OnceValue(func() []float64 {
		return detect.CoarseOrientationRungs(levels[0])
	})

	for i := range levels {
		go func() {
			us := uprightSlot(i)
			fp := &finding{}
			data, ok, evidence := decodeBitmapFinding(core.BitmapFromImage(levels[i]), quit(us), fp)
			if ok {
				commit(us)
			}
			results[us] = result{data, shorterSide(levels[i]), 0, toFrame(fp, levels[i]), ok}
			close(done[us])
			ss := searchSlot(i)
			if ok || !evidence || quit(ss)() {
				if i == 0 {
					seed <- toFrame(fp, levels[i])
				}
				close(done[ss])
				return
			}
			rungs := sharedRungs()
			if len(rungs) == 0 {
				rungs = nil // fall back to the level's own probe
			}
			data, deg, ok := decodeRetriesFinding(levels[i], quit(ss), fp, rungs)
			if ok {
				commit(ss)
			}
			if i == 0 {
				seed <- toFrame(fp, levels[i])
			}
			results[ss] = result{data, shorterSide(levels[i]), deg, toFrame(fp, levels[i]), ok}
			close(done[ss])
		}()
	}
	go func() {
		sf := <-seed
		if sf.located && !quit(1)() {
			if data, side, ok := decodeSeeded(levels, sf, quit(1)); ok {
				commit(1)
				results[1] = result{data, side, sf.deg, sf, true}
			}
		}
		close(done[1])
	}()

	for s := range done {
		<-done[s]
		if r := results[s]; r.ok {
			return r.data, r.side, r.deg, r.f, true
		}
	}
	return nil, 0, 0, finding{}, false
}
