package read

import (
	"image"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/phaseprobe"
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
// only built on first CPU consumption, halved from the next finer level. A
// decode whose routes all stay on the device therefore never builds the chain
// at all. Levels are indexed coarsest first; the finest level is the base
// conversion itself, materialized eagerly because both the GPU upload and the
// CPU fallbacks start from it.
//
// The host consumers here are CPU route slots, and they halve rather than read
// the retained GPU ladder back. The ladder's levels are the device route's own
// working set: shipping them to the host to spare it the halving chain made the
// device route pay for a host route's input, which is the one thing a decode
// budgeted at one upload and one download cannot do. The halving it costs was
// measured at 22 to 24 ms, and only on decodes where a CPU slot materializes a
// level at all.
type pyramid struct {
	dims   []image.Point
	base   *image.NRGBA
	levels []pyramidLevelSlot
}

type pyramidLevelSlot struct {
	once sync.Once
	img  *image.NRGBA
}

// pyramidDims derives every level's size from the frame's alone, coarsest
// first, or nil for a frame that holds only one scale. It depends on no pixels,
// which is what lets the GPU workspace be built for a frame that has not been
// decoded yet.
func pyramidDims(width, height int) []image.Point {
	if singleScaleFrame(image.Pt(width, height)) {
		return nil
	}
	dims := []image.Point{{X: width, Y: height}}
	for {
		last := dims[len(dims)-1]
		if singleScaleFrame(last) {
			break
		}
		dims = append(dims, image.Pt(max((last.X+1)/2, 1), max((last.Y+1)/2, 1)))
	}
	slices.Reverse(dims)
	return dims
}

// newPyramid derives the pyramid schedule for img, or nil when img cannot
// hold more than one level. Only the finest level's pixels are built here.
func newPyramid(img image.Image) *pyramid {
	b := img.Bounds()
	dims := pyramidDims(b.Dx(), b.Dy())
	if dims == nil {
		return nil
	}
	p := &pyramid{dims: dims, base: pyramidBase(img), levels: make([]pyramidLevelSlot, len(dims))}
	p.levels[len(dims)-1].img = p.base
	return p
}

// WarmGPUForFrame prepares the device decode route for a frame of this size
// without needing its pixels. The route's own acquisition otherwise creates the
// device, the kernel set, the finder chain compilation and the size-matched
// workspace after the image has been decoded, all of it on the critical path of
// a single-shot read. Everything there depends on geometry alone; only the
// pixel upload does not, and that stays where it was. A caller that knows the
// frame size before it decodes the image - which the header gives for free -
// can overlap the whole of it.
func WarmGPUForFrame(width, height int) {
	if dims := pyramidDims(width, height); dims != nil {
		detect.WarmAutomaticGPUDecode(width, height, len(dims))
	}
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
		slot.img = detect.HalveNRGBA(p.level(i + 1))
	})
	return slot.img
}

// levelImage exposes level i to the search ladder without materializing it.
func (p *pyramid) levelImage(i int) levelImage {
	return levelImage{size: p.dim(i), load: func() image.Image { return p.level(i) }}
}

// pyramidBase converts img once into the zero-origin NRGBA frame every level
// derives from. The single-level search runs on the same conversion, so
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
		phaseprobe.Mark("pyramid.session.start")
		gpuSession, _ = newGPUSession(gpuBase, p.count())
		phaseprobe.Markf("pyramid.session.end", "available=%t", gpuSession != nil)
	}
	if gpuSession != nil {
		retireWinner := false
		defer func() {
			if retireWinner {
				_ = gpuSession.Retire()
				return
			}
			_ = gpuSession.Close()
		}()
		defer func() {
			if ok && (tr == nil || !tr.detailed) {
				retireWinner = true
			}
		}()
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
	// Slot 0 is the coarsest whole-frame route, slot 1 the seeded route, 2..n
	// the finer whole-frame routes, n+1..2n the searches (coarsest first).
	n := p.count()
	frameSlot := func(i int) int {
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
			traces[frameSlot(i)] = &routeTrace{level: i, detailed: tr.detailed}
			traces[searchSlot(i)] = &routeTrace{level: i, detailed: tr.detailed}
		}
	}
	// A diagnostic read has to run the whole ladder: its report describes every
	// route, and a route that was cancelled has nothing to describe. An
	// ordinary read stops at the first success instead. LDPC has already
	// accepted the payload by then, so a later route could only reproduce that
	// answer, never improve on it, and waiting for one costs wall time on every
	// read whose winner is not the highest-priority route.
	exhaustive := tr != nil && tr.detailed
	var winner atomic.Int64
	winner.Store(int64(len(done)))
	quit := func(slot int) func() bool {
		if exhaustive {
			return func() bool { return winner.Load() < int64(slot) }
		}
		return func() bool {
			w := winner.Load()
			return w < int64(len(done)) && w != int64(slot)
		}
	}
	// Buffered for every slot, so publishing a success never blocks a route
	// that the reader has already stopped listening to.
	settled := make(chan int, len(done))
	publish := func(slot int) {
		if !exhaustive {
			settled <- slot
		}
	}
	commit := func(slot int) {
		for {
			w := winner.Load()
			if int64(slot) >= w || winner.CompareAndSwap(w, int64(slot)) {
				return
			}
		}
	}

	// The coarsest level sends its finding exactly once - after its whole-frame
	// read when that already settles it, otherwise after its search - and the
	// seeded route consumes it. A finding whose route decoded outranks a
	// locate-only one (decodeRetriesFinding); within one outcome class the
	// routes run sequentially in the level goroutine, so the choice is
	// deterministic.
	seed := make(chan finding, 1)

	for i := range n {
		go func() {
			us := frameSlot(i)
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
			traces[us].finishAttempt(routeAttempt{kind: "frame", roi: -1, stage: stage, side: fp.side, deg: attemptDeg(fp)}, detail, messageTransmission(data))
			results[us] = result{data, p.side(i), ok}
			if ok {
				commit(us)
				publish(us)
			}
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
			results[ss] = result{data, p.side(i), okSearch}
			if okSearch {
				commit(ss)
				publish(ss)
			}
			if i == 0 {
				seed <- *fp
			}
			close(done[ss])
		}()
	}
	go func() {
		f := <-seed
		if f.located && !quit(1)() {
			if data, side, ok := decodeSeededTracedCapabilities(p, f, quit(1), traces[1], capabilities); ok {
				results[1] = result{data, side, true}
				commit(1)
				publish(1)
			}
		}
		close(done[1])
	}()

	if !exhaustive {
		joined := make(chan struct{})
		go func() {
			for s := range done {
				<-done[s]
			}
			close(joined)
		}()
		select {
		case s := <-settled:
			// Only the winner's trace is joined: the routes still running own
			// theirs, and reading one of those would race their goroutines.
			tr.merge(traces[s])
			phaseprobe.Markf("pyramid.return", "slot=%d", s)
			return results[s].data, results[s].side, true
		case <-joined:
		}
		// Every route finished. A success published in the same moment the last
		// route closed is still a success, so take it before reporting failure.
		select {
		case s := <-settled:
			tr.merge(traces[s])
			phaseprobe.Markf("pyramid.return", "slot=%d", s)
			return results[s].data, results[s].side, true
		default:
		}
		for s := range done {
			tr.merge(traces[s])
		}
		return nil, 0, false
	}

	for s := range done {
		<-done[s]
		tr.merge(traces[s])
		if r := results[s]; r.ok {
			phaseprobe.Markf("pyramid.return", "slot=%d", s)
			return r.data, r.side, true
		}
	}
	return nil, 0, false
}
