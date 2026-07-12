package read

import (
	"image"
	"math"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/detect"
)

// Stream decodes successive camera frames of the same scene under a fixed
// per-frame work quota. Unlike the single-image Decode, which escalates
// through rotation rungs, regions of interest and an alignment-pattern
// fallback until everything failed, a Stream frame spends at most one replay
// of a remembered hypothesis, one upright scan, one probe-selected rotated
// attempt and one admission-gated payload correction, then returns and waits
// for the next frame: on a live camera the next frame is cheaper than
// searching this one harder. Hypotheses the budget could not try carry over
// to the following frames in deterministic order, so a hard first lock
// (rotated, or small in the frame) is found across a few frames instead of
// inside one. The zero value is ready to use; a Stream is not safe for
// concurrent use (frames of one camera arrive in order).
//
// Each frame's result is deterministic given the frames decoded before it:
// the ring and the hypothesis queue are pure functions of the sequence, and
// every attempt is deterministic.
type Stream struct {
	ring    []streamPrior // remembered hypotheses, most recent first
	pending []streamHyp   // untried hypotheses carried across frames, FIFO
	gen     uint64        // frame generation, monotonic
	work    streamWork    // work counters of the current frame
}

// streamPrior is a remembered hypothesis: the level scale and angle that
// read (or credibly located) a symbol, plus - when a locate published it -
// the finder geometry in frame coordinates and the frame size it was located
// on. A located quad enables the direct seeded replay; scale and angle alone
// still steer the scan.
type streamPrior struct {
	side int     // the pyramid level's shorter side
	deg  float64 // the pre-rotation, 0 for an upright read
	f    finding // frame-coordinate geometry when located (payload stripped)
	src  image.Point
}

// streamHyp is one untried read hypothesis: a pyramid scale and an angle.
type streamHyp struct {
	side int
	deg  float64
}

// streamWork counts the work one frame actually spent. The scheduler's
// budget is a testable contract: every counter has a hard per-frame cap and
// the exhaustive ladder's stages (regions of interest, alignment-pattern
// fallback) must never run, so their counters stay zero by construction.
type streamWork struct {
	levelsBuilt      int // pyramid levels materialized
	replayAttempts   int // remembered-hypothesis replays (cap 1)
	uprightScans     int // fresh upright scans (cap 1)
	rotatedAttempts  int // probe-selected rotated attempts (cap 1)
	correctionChains int // payload corrections spent (cap 1)
}

const (
	streamRingCap    = 3  // remembered hypotheses kept
	streamPendingCap = 16 // carried hypotheses kept
)

// Decode reads one frame within the per-frame quota. On success the winning
// hypothesis moves to the ring's front; on failure the ring is kept - a
// single blurred or occluded frame should not throw away a working lock -
// and the frame's unspent hypotheses wait in the queue for the next frame.
func (s *Stream) Decode(img image.Image) ([]byte, error) {
	s.gen++
	s.work = streamWork{}

	levels := pyramidLevels(img)
	s.work.levelsBuilt = max(len(levels), 1)
	b := img.Bounds()
	src := image.Pt(b.Dx(), b.Dy())

	type canvasKey struct {
		side int
		deg  float64
	}
	tried := map[canvasKey]bool{}

	attempt := func(hyp streamHyp) ([]byte, bool) {
		lvl := nearestLevelImage(img, levels, hyp.side)
		lb := lvl.Bounds()
		key := canvasKey{min(lb.Dx(), lb.Dy()), hyp.deg}
		if tried[key] {
			return nil, false
		}
		tried[key] = true
		var bm *core.Bitmap
		if hyp.deg != 0 {
			bm = detect.RotateToBitmap(lvl, hyp.deg)
		} else {
			bm = core.BitmapFromImage(lvl)
		}
		var rf finding
		data, _, ok := s.attemptBitmap(bm, &rf)
		if ok && rf.located {
			// Bank the winning geometry, converted to frame coordinates so
			// the next frame can seed from it directly. Locate-only geometry
			// stays out of the ring: promoting an unproven quad over a
			// working lock misdirects the replay (cross-frame evidence
			// banking is the accumulation step's separate store).
			rf.toImage(hyp.deg, bm.Width, bm.Height, lb.Dx(), lb.Dy(), image.Point{})
			rf.scale(float64(src.X)/float64(lb.Dx()), float64(src.Y)/float64(lb.Dy()))
			rf.payload = nil
			s.remember(streamPrior{side: key.side, deg: hyp.deg, f: rf, src: src})
		}
		return data, ok
	}

	// Replay the most recent remembered hypothesis: a located quad seeds the
	// sample directly first (no finder search; the admission gate is the
	// cheap miss), and on a seed miss the same replay slot spends one
	// re-locating scan at the remembered scale and angle - the finder search
	// is what survives hand-held drift the stale quad cannot. A miss keeps
	// the entry: geometry decays on repeated disagreement, not on one bad
	// frame.
	if len(s.ring) > 0 {
		r := s.ring[0]
		s.work.replayAttempts++
		if r.f.located && r.src == src {
			if data, ok := s.replayQuad(img, levels, r); ok {
				return data, nil
			}
		}
		if data, ok := attempt(streamHyp{side: r.side, deg: r.deg}); ok {
			return data, nil
		}
	}

	// One fresh upright scan at the coarsest scale (deduplicated against the
	// replay when that already was the coarse upright).
	s.work.uprightScans++
	if data, ok := attempt(streamHyp{side: coarsestSide(levels, min(src.X, src.Y))}); ok {
		return data, nil
	}

	// One probe-selected rotated (or finer-level) attempt: carried
	// hypotheses first, then a fresh probe of the coarse level. Whatever the
	// budget cannot try now waits for the following frames.
	if len(s.pending) == 0 {
		s.refillPending(img, levels, min(src.X, src.Y))
	}
	if len(s.pending) > 0 {
		hyp := s.pending[0]
		s.pending = s.pending[1:]
		s.work.rotatedAttempts++
		if data, ok := attempt(hyp); ok {
			return data, nil
		}
	}

	return nil, errDecodeFailed
}

// attemptBitmap runs one bounded read on a prepared canvas: locate, sample
// and interpret metadata, then finish through the shared gate. The locate
// geometry is published into f even when the read fails; admitted reports
// whether the observation passed the admission gate, so the caller can bank
// credible geometry without banking phantoms.
func (s *Stream) attemptBitmap(bm *core.Bitmap, f *finding) (data []byte, admitted, ok bool) {
	detect.BalanceRGB(bm)
	ch := detect.BinarizerRGB(bm, nil)
	symbols := make([]core.DecodedSymbol, maxSymbolNumber)
	d := &detect.PrimaryDetector{BM: bm, Ch: ch, Mode: detect.IntensiveDetect}
	obs, stage := observePrimary(d, &symbols[0], f)
	if stage != readSampled || obs == nil {
		return nil, false, false
	}
	admitted = obs.AdmitPayloadCorrection()
	data, ok = s.finishRead(bm, func() [3]*core.Bitmap { return ch }, symbols, obs, admitted)
	return data, admitted, ok
}

// replayQuad seeds a read directly from a remembered frame-coordinate quad
// on the pyramid level nearest the remembered scale: no finder search, one
// direct sample, and the admission gate as the cheap miss - a drifted or
// stale quad is refused before any payload correction. There is no
// alignment-pattern fallback; a miss waits for the scan slots or the next
// frame.
func (s *Stream) replayQuad(img image.Image, levels []*image.NRGBA, r streamPrior) (data []byte, ok bool) {
	lvl := nearestLevelImage(img, levels, r.side)
	lb := lvl.Bounds()

	var bm *core.Bitmap
	if r.deg != 0 {
		bm = detect.RotateToBitmap(lvl, r.deg)
	} else {
		bm = core.BitmapFromImage(lvl)
	}
	detect.BalanceRGB(bm)

	// Scale the frame-coordinate quad to the level, then map it onto the
	// rotation canvas (centred on the level, rotateInto's forward mapping).
	sx := float64(lb.Dx()) / float64(r.src.X)
	sy := float64(lb.Dy()) / float64(r.src.Y)
	rad := r.deg * math.Pi / 180
	cs, sn := math.Cos(rad), math.Sin(rad)
	lcx, lcy := float64(lb.Dx())/2, float64(lb.Dy())/2
	ccx, ccy := float64(bm.Width)/2, float64(bm.Height)/2
	var fps [4]detect.FinderPattern
	for i := range 4 {
		dx, dy := r.f.quad[i].X*sx-lcx, r.f.quad[i].Y*sy-lcy
		fps[i] = detect.FinderPattern{
			Typ:        i,
			Center:     core.PointF{X: cs*dx - sn*dy + ccx, Y: sn*dx + cs*dy + ccy},
			ModuleSize: r.f.sizes[i] * (sx + sy) / 2,
			FoundCount: 1,
		}
	}

	pt := core.PerspectiveTransform(fps[0].Center, fps[1].Center, fps[2].Center, fps[3].Center, r.f.side)
	matrix := detect.SampleSymbol(bm, pt, r.f.side)
	if matrix == nil {
		return nil, false
	}
	symbols := make([]core.DecodedSymbol, maxSymbolNumber)
	symbol := &symbols[0]
	symbol.Index = 0
	symbol.HostIndex = 0
	symbol.SideSize = r.f.side
	symbol.ModuleSize = (fps[0].ModuleSize + fps[1].ModuleSize + fps[2].ModuleSize + fps[3].ModuleSize) / 4.0
	for i := range 4 {
		symbol.PatternPositions[i] = fps[i].Center
	}
	obs, ret := decode.ObservePrimary(matrix, symbol)
	if ret != core.Success || obs == nil {
		return nil, false
	}
	// Validity pre-check, stricter than the general admission gate: a
	// hand-drifted quad still reads clean metadata syndromes (the short
	// metadata codes tolerate misalignment the data modules do not), so
	// without this a stale seed would spend the frame's only correction
	// chain on a doomed sample and starve the re-locating scan behind it.
	// Only a nearly perfect fixed-pattern read may pay for correction here.
	const seedMinFixedAgreement = 0.9
	if agree, checked := obs.FixedPatternAgreement(); checked == 0 || float64(agree) < seedMinFixedAgreement*float64(checked) {
		return nil, false
	}
	// The binarized channels are only needed once a docked secondary has to
	// be detected; the direct sample reads the balanced bitmap.
	return s.finishRead(bm, func() [3]*core.Bitmap { return detect.BinarizerRGB(bm, nil) }, symbols, obs, obs.AdmitPayloadCorrection())
}

// finishRead spends the frame's single admission-gated payload correction on
// an observed primary and, on success, completes the read (docked
// secondaries, message assembly). An inadmissible observation or an
// exhausted budget is a miss, not an error - the evidence waits for a better
// frame.
func (s *Stream) finishRead(bm *core.Bitmap, chFn func() [3]*core.Bitmap, symbols []core.DecodedSymbol, obs *decode.PrimaryObservation, admitted bool) (data []byte, ok bool) {
	if s.work.correctionChains >= 1 || !admitted {
		return nil, false
	}
	s.work.correctionChains++
	if obs.CorrectPayload() != core.Success {
		return nil, false
	}
	var ch [3]*core.Bitmap
	if symbols[0].Meta.DockedPosition != 0 {
		ch = chFn()
	}
	return decodeSymbols(bm, ch, symbols, 1)
}

// remember moves a hypothesis to the ring's front, deduplicated by scale and
// angle, bounded.
func (s *Stream) remember(p streamPrior) {
	kept := s.ring[:0:0]
	kept = append(kept, p)
	for _, r := range s.ring {
		if (r.side != p.side || r.deg != p.deg) && len(kept) < streamRingCap {
			kept = append(kept, r)
		}
	}
	s.ring = kept
}

// refillPending enqueues fresh hypotheses once the carried queue is empty:
// the coarse orientation probe's angles at the coarse scale, then the finer
// levels' uprights - the cross-frame escalation for symbols too small for
// the coarse scan. The queue is bounded; a live stream re-probes on a later
// frame rather than hoarding stale angles.
func (s *Stream) refillPending(img image.Image, levels []*image.NRGBA, baseSide int) {
	probeOn := img
	if levels != nil {
		probeOn = levels[0]
	}
	for _, deg := range detect.CoarseOrientationRungs(probeOn) {
		if deg == 0 {
			continue // the upright scan owns the zero angle
		}
		s.enqueue(streamHyp{side: coarsestSide(levels, baseSide), deg: deg})
	}
	for _, lvl := range levels[min(1, len(levels)):] {
		s.enqueue(streamHyp{side: shorterSide(lvl)})
	}
}

func (s *Stream) enqueue(h streamHyp) {
	if len(s.pending) >= streamPendingCap {
		return
	}
	for _, p := range s.pending {
		if p == h {
			return
		}
	}
	s.pending = append(s.pending, h)
}

// nearestLevelImage picks the pyramid level whose shorter side is closest to
// the wanted scale, or the frame itself when there is no pyramid.
func nearestLevelImage(img image.Image, levels []*image.NRGBA, side int) image.Image {
	if levels == nil {
		return img
	}
	best := levels[0]
	for _, l := range levels[1:] {
		if absInt(shorterSide(l)-side) < absInt(shorterSide(best)-side) {
			best = l
		}
	}
	return best
}

// coarsestSide is the shorter side of the coarsest pyramid level, or of the
// frame itself when there is no pyramid.
func coarsestSide(levels []*image.NRGBA, baseSide int) int {
	if len(levels) == 0 {
		return baseSide
	}
	return shorterSide(levels[0])
}

func shorterSide(img *image.NRGBA) int { return min(img.Rect.Dx(), img.Rect.Dy()) }

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
