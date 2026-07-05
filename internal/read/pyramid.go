package read

import (
	"image"
	"slices"
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
// the level's orientation and region search. Results commit in a fixed
// priority order - every upright (coarsest first) ahead of every search
// (coarsest first), never first-done - so the outcome is deterministic
// regardless of scheduling (any success yields the same payload bytes; the
// residual hazard of a miscorrected decode differing between levels is why
// the order is pinned). Uprights outrank searches because they are the cheap
// bounded hypothesis: a capture that reads upright at some scale never waits
// on an expensive orientation search, and a small-module capture that only
// reads at full resolution waits only for the coarser uprights to fail. The
// searches still start the moment their own level's upright fails, so a
// rotated capture's coarse search overlaps the finer uprights instead of
// queueing behind them. Slots that can no longer win are told to quit at
// their next stage boundary and are not waited for - each level only touches
// its own data.
// On success it also reports the winning hypothesis - the level's shorter
// side and the pre-rotation angle (0 for an upright win) - which a Stream
// replays as its first attempt on the next frame.
func decodePyramid(levels []*image.NRGBA) (data []byte, side int, deg float64, ok bool) {
	type result struct {
		data []byte
		deg  float64
		ok   bool
	}
	// Slots 0..n-1 are the uprights, n..2n-1 the searches, coarsest first.
	n := len(levels)
	results := make([]result, 2*n)
	done := make([]chan struct{}, 2*n)
	for s := range done {
		done[s] = make(chan struct{})
	}
	var winner atomic.Int64
	winner.Store(int64(2 * n))
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

	for i := range levels {
		go func() {
			data, ok, evidence := decodeBitmap(core.BitmapFromImage(levels[i]), quit(i))
			if ok {
				commit(i)
			}
			results[i] = result{data: data, ok: ok}
			close(done[i])
			if ok || !evidence || quit(n+i)() {
				close(done[n+i])
				return
			}
			data, deg, ok := decodeRetries(levels[i], quit(n+i))
			if ok {
				commit(n + i)
			}
			results[n+i] = result{data, deg, ok}
			close(done[n+i])
		}()
	}
	for s := range done {
		<-done[s]
		if r := results[s]; r.ok {
			lvl := levels[s%n]
			return r.data, min(lvl.Rect.Dx(), lvl.Rect.Dy()), r.deg, true
		}
	}
	return nil, 0, 0, false
}
