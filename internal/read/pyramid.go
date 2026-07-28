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

// singleScaleFrame reports whether a frame of this size is too small to carry
// a resolution pyramid, so the search has exactly one scale to work with.
func singleScaleFrame(size image.Point) bool {
	return min(size.X, size.Y) < 2*minPyramidSide
}

// pyramidLevels builds the resolution pyramid Decode searches, coarsest level
// first, or nil when img cannot hold more than one level - the single-level
// path then runs the search directly on img, byte-identical to a pipeline
// without a pyramid.
func pyramidLevels(img image.Image) []*image.NRGBA {
	b := img.Bounds()
	if singleScaleFrame(image.Pt(b.Dx(), b.Dy())) {
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

// pyramid is the lazily materialized resolution pyramid: every level's
// dimensions are derived up front from the frame's, but a level's pixels are
// only built on first CPU consumption - downloaded from the GPU ladder when a
// session retains them (byte-identical to the CPU halving per the ladder
// parity gate), halved from the next finer level otherwise. A decode whose
// routes stay on the device therefore never builds the CPU half-scale chain
// the GPU ladder already holds. Levels are indexed coarsest first; the finest
// level is the base conversion itself, materialized eagerly because both the
// GPU upload and the CPU fallbacks start from it.
type pyramid struct {
	dims []image.Point
	base *image.NRGBA
	// download, when set, reads a level back from the retained GPU ladder.
	// It is installed after the session builds and before any consumer runs,
	// and returns nil to fall back to the CPU halving chain.
	download func(level int) *image.NRGBA
	levels   []pyramidLevelSlot
}

type pyramidLevelSlot struct {
	once sync.Once
	img  *image.NRGBA
}

// newPyramid derives the pyramid schedule for img, or nil when img cannot
// hold more than one level. Only the finest level's pixels are built here.
func newPyramid(img image.Image) *pyramid {
	b := img.Bounds()
	if singleScaleFrame(image.Pt(b.Dx(), b.Dy())) {
		return nil
	}
	dims := []image.Point{{X: b.Dx(), Y: b.Dy()}}
	for {
		last := dims[len(dims)-1]
		if singleScaleFrame(last) {
			break
		}
		dims = append(dims, image.Pt(max((last.X+1)/2, 1), max((last.Y+1)/2, 1)))
	}
	slices.Reverse(dims)
	p := &pyramid{dims: dims, base: pyramidBase(img), levels: make([]pyramidLevelSlot, len(dims))}
	p.levels[len(dims)-1].img = p.base
	return p
}

// eagerPyramid wraps already materialized levels (coarsest first) in the
// pyramid interface, for callers that built their levels up front.
func eagerPyramid(levels []*image.NRGBA) *pyramid {
	p := &pyramid{
		dims:   make([]image.Point, len(levels)),
		base:   levels[len(levels)-1],
		levels: make([]pyramidLevelSlot, len(levels)),
	}
	for i, level := range levels {
		p.dims[i] = image.Pt(level.Rect.Dx(), level.Rect.Dy())
		p.levels[i].img = level
	}
	return p
}

func (p *pyramid) count() int            { return len(p.dims) }
func (p *pyramid) dim(i int) image.Point { return p.dims[i] }
func (p *pyramid) side(i int) int        { return min(p.dims[i].X, p.dims[i].Y) }

// level materializes level i's pixels on first use. Safe for concurrent
// callers; the recursion into the next finer level terminates at the eager
// base.
func (p *pyramid) level(i int) *image.NRGBA {
	slot := &p.levels[i]
	slot.once.Do(func() {
		if slot.img != nil {
			return
		}
		if p.download != nil {
			if img := p.download(i); img != nil {
				slot.img = img
				return
			}
		}
		slot.img = detect.HalveNRGBA(p.level(i + 1))
	})
	return slot.img
}

// levelImage exposes level i to the search ladder without materializing it.
func (p *pyramid) levelImage(i int) levelImage {
	return levelImage{size: p.dim(i), load: func() image.Image { return p.level(i) }}
}

// pyramidBase converts img once into the zero-origin NRGBA frame every level
// derives from - the one conversion all of a level's orientation rungs then
// share (rotatePrep aliases a zero-origin NRGBA instead of re-copying the
// canvas per rung). The single-level search runs on the same conversion, so
// its unjoined route slots read decoder-owned memory, never the caller's
// image. The pipeline never reads alpha, so it is forced opaque;
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

// nrgbaBase returns img as an NRGBA for the enlarged detection scale to
// upsample: the frame itself when it already is one, otherwise the same
// zero-origin opaque conversion the pyramid derives its levels from.
func nrgbaBase(img image.Image) *image.NRGBA {
	if base, ok := img.(*image.NRGBA); ok {
		return base
	}
	return pyramidBase(img)
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
func decodePyramid(p *pyramid, tr *routeTrace) (data []byte, side int, ok bool) {
	message, side, ok := decodePyramidCapabilities(p, tr, compiledCapabilities())
	return messageTransmission(message), side, ok
}

func decodePyramidOnly(p *pyramid, tr *routeTrace, variant wire.Variant) (data []byte, side int, ok bool) {
	message, side, ok := decodePyramidCapabilities(p, tr, variant.Mask())
	return messageTransmission(message), side, ok
}

func decodePyramidCapabilities(p *pyramid, tr *routeTrace, capabilities wire.Capabilities) (data *Message, side int, ok bool) {
	return decodePyramidCapabilitiesWithGPU(
		p,
		tr,
		capabilities,
		detect.NewAutomaticGPUDecodeSession,
	)
}

type gpuDecodeSessionFactory func(
	base *core.Bitmap,
	levelCount int,
) (*detect.GPUDecodeSession, error)

func decodePyramidCapabilitiesWithGPU(
	p *pyramid,
	tr *routeTrace,
	capabilities wire.Capabilities,
	newGPUSession gpuDecodeSessionFactory,
) (data *Message, side int, ok bool) {
	gpuBase := &core.Bitmap{
		Width: p.base.Rect.Dx(), Height: p.base.Rect.Dy(), Channels: 4, Pix: p.base.Pix,
	}
	var gpuSession *detect.GPUDecodeSession
	if newGPUSession != nil {
		gpuSession, _ = newGPUSession(gpuBase, p.count())
	}
	if gpuSession != nil {
		defer gpuSession.Close()
		// The session retains every level on the device; a lazy CPU consumer
		// downloads its level instead of rebuilding the halving chain the
		// ladder already ran (byte-identical either way). The field is
		// written once here, before any consumer goroutine starts, and never
		// cleared - clearing it would race unjoined straggler slots. A
		// straggler materializing after this decode returned finds the
		// session closed and falls back to the CPU halving; its work is
		// discarded either way.
		p.download = func(level int) *image.NRGBA {
			bm, err := gpuSession.DownloadLevel(p.count() - 1 - level)
			if err != nil || bm == nil {
				return nil
			}
			return bm.NRGBA()
		}
	}
	if tr != nil && tr.detailed {
		tr.pyramid = make([]image.Point, p.count())
		tr.pyramidImages = make([]image.Image, p.count())
		for i := range p.count() {
			tr.pyramid[i] = p.dim(i)
			tr.pyramidImages[i] = p.level(i)
		}
	}
	type result struct {
		data *Message
		side int
		ok   bool
	}
	// Slot 0 is the coarsest upright, slot 1 the seeded route, 2..n the finer
	// uprights, n+1..2n the searches (coarsest first).
	n := p.count()
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
		for i := range n {
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

	for i := range n {
		go func() {
			us := uprightSlot(i)
			fp := &finding{}
			detail := traces[us].beginAttempt(-1)
			data, stage, evidence := decodePyramidLevelFindingCapabilities(
				p.levelImage(i).load,
				quit(us),
				fp,
				detail,
				capabilities,
				gpuSession,
				n-1-i,
			)
			ok := stage == readDecoded
			traces[us].finishAttempt(routeAttempt{kind: "upright", roi: -1, stage: stage, side: fp.side}, detail, messageTransmission(data))
			if ok {
				commit(us)
			}
			results[us] = result{data, p.side(i), ok}
			close(done[us])
			ss := searchSlot(i)
			if ok || !evidence || quit(ss)() {
				if i == 0 {
					seed <- *fp
				}
				close(done[ss])
				return
			}
			data, okSearch := decodeRetriesRegionsLevel(
				p.levelImage(i),
				quit(ss),
				fp,
				true,
				traces[ss],
				capabilities,
			)
			if okSearch {
				commit(ss)
			}
			if i == 0 {
				seed <- *fp
			}
			results[ss] = result{data, p.side(i), okSearch}
			close(done[ss])
		}()
	}
	go func() {
		f := <-seed
		if f.located && !quit(1)() {
			if data, side, ok := decodeSeededTracedCapabilities(p, f, quit(1), traces[1], capabilities); ok {
				commit(1)
				results[1] = result{data, side, true}
			}
		}
		close(done[1])
	}()

	for s := range done {
		<-done[s]
		tr.merge(traces[s])
		if r := results[s]; r.ok {
			return r.data, r.side, true
		}
	}
	return nil, 0, false
}
