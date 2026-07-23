package read

import (
	"image"
	"math"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/wire"
)

// Stream decodes one ordered, coherent frame sequence under a fixed per-frame
// work quota. Unlike the single-image Decode, which escalates
// through rotation rungs, regions of interest and an alignment-pattern
// fallback until everything failed, a Stream frame spends at most one replay
// of a remembered hypothesis, one upright scan, one probe-selected rotated
// attempt and one admission-gated payload correction, then returns and waits
// for the next frame: in a coherent sequence the next frame is usually cheaper
// than searching this one harder. Hypotheses the budget could not try carry over
// to the following frames in deterministic order, so a hard first lock
// (rotated, or small in the frame) is found across a few frames instead of
// inside one. The zero value is ready to use; a Stream is not safe for
// concurrent use (frames are supplied in sequence order).
//
// Each frame's result is deterministic given the frames decoded before it:
// the ring and the hypothesis queue are pure functions of the sequence, and
// every attempt is deterministic.
//
// Emission is per frame. DecodeMessage returns the payload currently on screen
// for every frame that yields one, and an error for a frame that does not,
// including a transition frame showing two codes at once, which never decodes.
// A frame that re-shows an already-returned code returns it again (cheaply,
// without a correction, when the frame's own evidence still confirms it), and a
// code that reappears later in the sequence is decoded afresh: the Stream never
// withholds a payload because it emitted those bytes before. Cross-frame
// deduplication, loop-occurrence identity and whole-message reassembly are the
// caller's responsibility, above this decoder. Frame order is the caller's
// supply order, which the Stream neither reorders nor tags, so a caller
// correlates payloads to frames by the order it supplied them.
type Stream struct {
	capabilities  wire.Capabilities // zero selects every decoder compiled into this build
	forced        bool              // capabilities is an explicit internal oracle selection
	routeCursor   uint8             // next irreducible wire interpretation in deterministic order
	routeFailures uint8             // consecutive corrections spent on the selected route
	ring          []streamPrior     // remembered hypotheses, most recent first
	pending       []streamHyp       // untried hypotheses carried across frames, FIFO
	group         evidenceGroup     // fixed-anchor content evidence, separate from the search ring
	gen           uint64            // frame generation, monotonic
	bankedGen     uint64            // generation whose one canonical observation was offered
	enlargedGen   uint64            // generation that last spent the enlarged attempt, 0 for none
	work          streamWork        // work counters of the current frame
	gpuSession    *detect.GPUDecodeSession
	gpuWidth      int
	gpuHeight     int
	gpuLevelCount int
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

// streamHyp is one untried read hypothesis: a pyramid scale and an angle, or
// the enlarged detection scale when a single-scale frame is too small for a
// finder cross-check at its native pixels.
type streamHyp struct {
	side     int
	deg      float64
	enlarged bool // sample a Catmull-Rom enlargement of the frame, not a native level
}

type streamObservation struct {
	optionalStreamObservation
	channels [3]*core.Bitmap
	symbols  []core.DecodedSymbol
	primary  *decode.PrimaryObservation
	route    streamRoute
}

type streamRoute struct {
	family  detect.FinderFamily
	variant wire.Variant
}

type streamSample struct {
	matrix   *core.Bitmap
	base     core.DecodedSymbol
	channels [3]*core.Bitmap
}

// preparedFrame owns the balanced pixels and lazily materialized detector
// channels for one Stream hypothesis. Keeping both behind one frame seam lets
// observation, correction and a future resident backend share preparation.
type preparedFrame struct {
	bitmap        *core.Bitmap
	channels      [3]*core.Bitmap
	channelsReady bool
}

func newPreparedFrame(bitmap *core.Bitmap) *preparedFrame {
	detect.BalanceRGB(bitmap)
	return &preparedFrame{bitmap: bitmap}
}

func (frame *preparedFrame) detectorChannels() [3]*core.Bitmap {
	if !frame.channelsReady {
		frame.channels = detect.BinarizerRGB(frame.bitmap, nil)
		frame.channelsReady = true
	}
	return frame.channels
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
	enlargedAttempts int // enlarged single-scale attempts (cap 1)
	correctionChains int // payload corrections spent (cap 1)
}

const (
	streamRingCap    = 3  // remembered hypotheses kept
	streamPendingCap = 16 // carried hypotheses kept
)

// NewStreamOnly returns an empty stream restricted to variant for internal
// oracle tests. The zero Stream uses every decoder compiled into this build.
func NewStreamOnly(variant wire.Variant) Stream {
	return Stream{capabilities: variant.Mask(), forced: true}
}

// Reset discards every retained hypothesis and evidence group. An
// oracle-restricted stream keeps its selected capability; an ordinary stream
// returns to the same state as its zero value.
func (s *Stream) Reset() {
	_ = s.closeGPU()
	if !s.forced {
		*s = Stream{}
		return
	}
	*s = Stream{capabilities: s.capabilities, forced: true}
}

// Close releases the optional resident GPU workspace. The zero Stream remains
// usable after Close and will reopen a session when a later frame qualifies.
func (s *Stream) Close() error {
	if s == nil {
		return nil
	}
	return s.closeGPU()
}

func (s *Stream) closeGPU() error {
	if s.gpuSession == nil {
		return nil
	}
	err := s.gpuSession.Close()
	s.gpuSession = nil
	s.gpuWidth, s.gpuHeight, s.gpuLevelCount = 0, 0, 0
	return err
}

func (s *Stream) refreshGPU(img image.Image, p *pyramid) *detect.GPUDecodeSession {
	levels := 1
	if p != nil {
		levels = p.count()
	}
	base := core.BitmapFromImage(nrgbaBase(img))
	if s.gpuSession != nil && s.gpuWidth == base.Width && s.gpuHeight == base.Height && s.gpuLevelCount == levels {
		if err := s.gpuSession.ReplaceBase(base); err == nil {
			return s.gpuSession
		}
		_ = s.closeGPU()
	}
	session, err := detect.NewAutomaticGPUDecodeSession(base, levels)
	if err != nil || session == nil {
		return nil
	}
	s.gpuSession = session
	s.gpuWidth, s.gpuHeight, s.gpuLevelCount = base.Width, base.Height, levels
	return session
}

func (s *Stream) capabilitySet() wire.Capabilities {
	compiled := compiledCapabilities()
	if s.forced {
		return s.capabilities & compiled
	}
	return compiled
}

func streamRoutes(capabilities wire.Capabilities) ([4]streamRoute, int) {
	var routes [4]streamRoute
	n := 0
	current, currentCount := currentObservationVariants(capabilities)
	for _, variant := range current[:currentCount] {
		routes[n] = streamRoute{family: detect.FinderFamilyCurrent, variant: variant}
		n++
	}
	historical, historicalCount := historicalObservationVariants(capabilities)
	for _, variant := range historical[:historicalCount] {
		routes[n] = streamRoute{family: detect.FinderFamilyBSI, variant: variant}
		n++
	}
	return routes, n
}

func (s *Stream) orderedRoutes(capabilities wire.Capabilities) ([4]streamRoute, int) {
	routes, n := streamRoutes(capabilities)
	if n < 2 {
		return routes, n
	}
	start := int(s.routeCursor) % n
	var ordered [4]streamRoute
	for i := range n {
		ordered[i] = routes[(start+i)%n]
	}
	return ordered, n
}

func (s *Stream) selectRoute(capabilities wire.Capabilities, selected streamRoute) {
	routes, n := streamRoutes(capabilities)
	for i, route := range routes[:n] {
		if route == selected {
			s.routeCursor = uint8(i)
			s.routeFailures = 0
			return
		}
	}
}

func (s *Stream) failRoute(capabilities wire.Capabilities, attempted streamRoute) {
	// ISO-base observations are the only routes with a defined cross-frame
	// evidence model. Keep that route for one additional correction when its
	// first failed frame established a real group, so the next complementary
	// frame can exercise the two-frame fusion gate. A second failure advances
	// to the next compiled interpretation instead of starving it indefinitely.
	if attempted.variant.UsesISO23634Base() && s.routeFailures == 0 && len(s.group.snaps) > 0 {
		s.routeFailures = 1
		return
	}
	s.routeFailures = 0
	s.advanceRoute(capabilities, attempted)
}

func (s *Stream) advanceRoute(capabilities wire.Capabilities, attempted streamRoute) {
	routes, n := streamRoutes(capabilities)
	if n == 0 {
		s.routeCursor = 0
		return
	}
	for i, route := range routes[:n] {
		if route == attempted {
			s.routeCursor = uint8((i + 1) % n)
			return
		}
	}
	s.routeCursor = 0
}

func streamFinderFamilies(capabilities wire.Capabilities) detect.FinderFamilySet {
	var wanted detect.FinderFamilySet
	routes, n := streamRoutes(capabilities)
	for _, route := range routes[:n] {
		wanted |= route.family.Mask()
	}
	return wanted
}

// Decode reads one frame within the per-frame quota. On success the winning
// hypothesis moves to the ring's front; on failure the ring is kept - a
// single blurred or occluded frame should not throw away a working lock -
// and the frame's unspent hypotheses wait in the queue for the next frame.
func (s *Stream) Decode(img image.Image) ([]byte, error) {
	message, err := s.DecodeMessage(img)
	return messageTransmission(message), err
}

// DecodeMessage reads one frame and returns paired raw data and reader
// transmission from the winning correction.
func (s *Stream) DecodeMessage(img image.Image) (*Message, error) {
	if err := validateImage(img); err != nil {
		return nil, err
	}
	s.gen++
	s.work = streamWork{}
	capabilities := s.capabilitySet()
	wantedFinders := streamFinderFamilies(capabilities)
	if wantedFinders == 0 {
		return nil, errDecodeFailed
	}

	p := newPyramid(img)
	gpuSession := s.refreshGPU(img, p)
	s.work.levelsBuilt = 1
	if p != nil {
		s.work.levelsBuilt = p.count()
	}
	b := img.Bounds()
	src := image.Pt(b.Dx(), b.Dy())

	type canvasKey struct {
		side int
		deg  float64
	}
	tried := map[canvasKey]bool{}

	// canvas builds (or reuses) the prepared balanced bitmap for one
	// hypothesis; scan runs the full locate-and-read on it, banking winning
	// geometry in frame coordinates so the next frame can seed from it
	// directly. Locate-only geometry stays out of the ring: promoting an
	// unproven quad over a working lock misdirects the replay (cross-frame
	// evidence banking is the accumulation step's separate store).
	canvas := func(hyp streamHyp) (*preparedFrame, image.Rectangle) {
		var lvl image.Image
		if hyp.enlarged {
			lvl = detect.UpscaleNRGBA(nrgbaBase(img), enlargeFactor)
		} else {
			lvl = nearestLevelImage(img, p, hyp.side)
		}
		var bm *core.Bitmap
		if hyp.deg != 0 {
			bm = detect.RotateToBitmap(lvl, hyp.deg)
		} else {
			bm = core.BitmapFromImage(lvl)
		}
		return newPreparedFrame(bm), lvl.Bounds()
	}
	scan := func(hyp streamHyp, frame *preparedFrame, lb image.Rectangle) (*Message, bool) {
		key := canvasKey{min(lb.Dx(), lb.Dy()), hyp.deg}
		if tried[key] {
			return nil, false
		}
		tried[key] = true
		var rf finding
		var observed *streamObservation
		gpuBitmap := frame.bitmap
		gpuBounds := lb
		var gpuChannels [3]*core.Bitmap
		gpuUsed := false
		if gpuSession != nil && !hyp.enlarged {
			if level, ok := streamGPULevel(p, hyp.side); ok {
				var detector *detect.PrimaryDetector
				var found detect.FinderFamilySet
				var size image.Point
				var err error
				if hyp.deg == 0 {
					detector, found, err = gpuSession.LocateLevelFamilies(
						level, wantedFinders, detect.IntensiveDetect, nil, nil,
					)
					if p != nil {
						size = p.dim(p.count() - 1 - level)
					} else {
						size = image.Pt(s.gpuWidth, s.gpuHeight)
					}
				} else {
					if p != nil {
						size = p.dim(p.count() - 1 - level)
					} else {
						size = image.Pt(s.gpuWidth, s.gpuHeight)
					}
					crop := image.Rect(0, 0, size.X, size.Y)
					detector, found, size, err = gpuSession.LocateRouteFamilies(
						level, crop, hyp.deg, wantedFinders, detect.IntensiveDetect, nil, nil,
					)
				}
				if err == nil && detector != nil && found != 0 {
					observed = s.observeLocatedDetector(
						detector.BM,
						func() [3]*core.Bitmap {
							if !detector.EnsureChannels() {
								return [3]*core.Bitmap{}
							}
							return detector.Ch
						}, detector, found, &rf, capabilities,
					)
					if observed != nil {
						gpuBitmap = detector.BM
						gpuBounds = image.Rect(0, 0, size.X, size.Y)
						gpuChannels = observed.channels
						gpuUsed = true
					}
				}
			}
		}
		if observed == nil {
			observed = s.observePreparedFrame(frame, &rf, capabilities, wantedFinders)
		}
		if observed == nil {
			return nil, false
		}
		if rf.located {
			rf.toImage(hyp.deg, gpuBitmap.Width, gpuBitmap.Height, gpuBounds.Dx(), gpuBounds.Dy(), image.Point{})
			rf.scale(float64(src.X)/float64(gpuBounds.Dx()), float64(src.Y)/float64(gpuBounds.Dy()))
		}
		data, ok := s.finishStreamObservation(
			gpuBitmap, func() [3]*core.Bitmap {
				if gpuUsed {
					return gpuChannels
				}
				return frame.detectorChannels()
			}, observed, rf, src, capabilities,
		)
		// An enlarged lock is not remembered: the ring replays on native-scale
		// levels, so a quad in enlarged coordinates has no canvas to seed from,
		// and the enlargement itself is the cost a replay would still pay. A
		// small frame simply re-earns the enlarged attempt through the carried
		// hypothesis on a later frame.
		if ok && rf.located && !hyp.enlarged {
			rf.payload = nil
			s.remember(streamPrior{side: key.side, deg: hyp.deg, f: rf, src: src})
		}
		return data, ok
	}
	attempt := func(hyp streamHyp) (*Message, bool) {
		frame, lb := canvas(hyp)
		return scan(hyp, frame, lb)
	}

	// Replay the most recent remembered hypothesis: a located quad seeds the
	// sample directly first (no finder search; the strict validity pre-check
	// is the cheap miss), and on a seed miss the same replay slot spends one
	// re-locating scan on the SAME prepared canvas - the finder search is
	// what survives hand-held drift the stale quad cannot. A miss keeps the
	// entry: geometry decays on repeated disagreement, not on one bad frame.
	if len(s.ring) > 0 {
		r := s.ring[0]
		s.work.replayAttempts++
		hyp := streamHyp{side: r.side, deg: r.deg}
		frame, lb := canvas(hyp)
		if r.f.located && r.src == src {
			if data, ok := s.replayQuad(frame, lb, r, capabilities); ok {
				return data, nil
			}
			if s.work.correctionChains >= 1 {
				return nil, errDecodeFailed
			}
		}
		if data, ok := scan(hyp, frame, lb); ok {
			return data, nil
		}
		if s.work.correctionChains >= 1 {
			return nil, errDecodeFailed
		}
	}

	// One fresh upright scan at the coarsest scale (deduplicated against the
	// replay when that already was the coarse upright).
	s.work.uprightScans++
	if data, ok := attempt(streamHyp{side: coarsestSide(p, min(src.X, src.Y))}); ok {
		return data, nil
	}
	if s.work.correctionChains >= 1 {
		return nil, errDecodeFailed
	}

	// One probe-selected rotated (or finer-level) attempt: carried
	// hypotheses first, then a fresh probe of the coarse level. Whatever the
	// budget cannot try now waits for the following frames.
	if len(s.pending) == 0 {
		s.refillPending(img, p, min(src.X, src.Y))
	}
	if len(s.pending) > 0 {
		hyp := s.pending[0]
		s.pending = s.pending[1:]
		if hyp.enlarged {
			s.work.enlargedAttempts++
			s.enlargedGen = s.gen
		} else {
			s.work.rotatedAttempts++
		}
		if data, ok := attempt(hyp); ok {
			return data, nil
		}
	}

	return nil, errDecodeFailed
}

// observePreparedFrame runs one integrated finder traversal on a prepared,
// balanced canvas. Each located physical family is sampled at most once; the
// shared sample is then offered to applicable wire interpretations in the
// stream's deterministic route order until one is plausible enough to spend
// the frame's correction slot.
func (s *Stream) observePreparedFrame(frame *preparedFrame, f *finding, capabilities wire.Capabilities,
	wantedFinders detect.FinderFamilySet) *streamObservation {
	ch := frame.detectorChannels()
	d := &detect.PrimaryDetector{BM: frame.bitmap, Ch: ch, Mode: detect.IntensiveDetect}
	found := d.LocateFinderFamilies(wantedFinders)
	return s.observeLocatedDetector(frame.bitmap, func() [3]*core.Bitmap { return d.Ch }, d, found, f, capabilities)
}

func (s *Stream) observeLocatedDetector(bitmap *core.Bitmap, channels func() [3]*core.Bitmap,
	d *detect.PrimaryDetector, found detect.FinderFamilySet, f *finding,
	capabilities wire.Capabilities) *streamObservation {
	routes, routeCount := s.orderedRoutes(capabilities)
	var samples [2]streamSample
	var sampled, sampleOK [2]bool
	observed := new(streamObservation)

	for _, route := range routes[:routeCount] {
		if !found.Has(route.family) {
			continue
		}
		familyIndex := int(route.family)
		if familyIndex >= len(samples) {
			continue
		}
		if !sampled[familyIndex] {
			sampled[familyIndex] = true
			if !d.SelectFinderFamily(route.family) {
				continue
			}
			base := core.DecodedSymbol{}
			matrix, stage := sampleLocatedPrimaryTraced(d, route.family, &base, f, nil)
			if stage != readSampled {
				continue
			}
			samples[familyIndex] = streamSample{matrix: matrix, base: base, channels: channels()}
			sampleOK[familyIndex] = true
		}
		if !sampleOK[familyIndex] {
			continue
		}
		if observeStreamRoute(samples[familyIndex], route, capabilities, observed) {
			return observed
		}
	}
	return nil
}

func observeStreamRoute(sample streamSample, route streamRoute, capabilities wire.Capabilities,
	observed *streamObservation) bool {
	if route.family == detect.FinderFamilyCurrent {
		symbols := make([]core.DecodedSymbol, maxSymbolNumber)
		symbols[0] = sample.base
		symbols[0].WireVariant = route.variant
		observation, result := decode.ObservePrimary(sample.matrix, &symbols[0])
		normalizeCurrentVariant(&symbols[0], nil, capabilities, 0)
		if result != core.Success || observation == nil || !observation.AdmitPayloadCorrection() {
			return false
		}
		*observed = streamObservation{
			channels: sample.channels, symbols: symbols, primary: observation,
			route: route,
		}
		return true
	}
	if route.family == detect.FinderFamilyBSI {
		symbols, correction, ok, seedAdmitted := observeHistoricalStreamSampled(sample.matrix, sample.base, route.variant)
		if !ok {
			return false
		}
		return setHistoricalStreamObservation(observed, sample.channels, symbols, correction, route, seedAdmitted)
	}
	return false
}

// replayQuad seeds a read directly from a remembered frame-coordinate quad
// on a prepared, already balanced canvas of the remembered level and angle:
// no finder search, one direct sample, and a strict validity pre-check as
// the cheap miss - a drifted or stale quad is refused before any payload
// correction. There is no alignment-pattern fallback; a miss falls to the
// re-locating scan on the same canvas.
func (s *Stream) replayQuad(frame *preparedFrame, lb image.Rectangle, r streamPrior, capabilities wire.Capabilities) (data *Message, ok bool) {
	bm := frame.bitmap
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
	base := core.DecodedSymbol{
		Index: 0, HostIndex: 0, SideSize: r.f.side,
		ModuleSize: (fps[0].ModuleSize + fps[1].ModuleSize + fps[2].ModuleSize + fps[3].ModuleSize) / 4.0,
	}
	for i := range 4 {
		base.PatternPositions[i] = fps[i].Center
	}
	routes, routeCount := s.orderedRoutes(capabilities)
	if routeCount == 0 || routes[0].family != r.f.family {
		return nil, false
	}
	sample := streamSample{matrix: matrix, base: base}
	var observed streamObservation
	observedOK := false
	for _, route := range routes[:routeCount] {
		if route.family != r.f.family {
			break
		}
		if observeStreamRoute(sample, route, capabilities, &observed) {
			observedOK = true
			break
		}
	}
	if !observedOK {
		return nil, false
	}
	// Validity pre-check, stricter than the general admission gate: a
	// hand-drifted quad still reads clean metadata syndromes (the short
	// metadata codes tolerate misalignment the data modules do not), so
	// without this a stale seed would spend the frame's only correction
	// chain on a doomed sample and starve the re-locating scan behind it.
	// Only a nearly perfect fixed-pattern read may pay for correction here.
	if observed.primary != nil {
		const seedMinFixedAgreement = 0.9
		if agree, checked := observed.primary.FixedPatternAgreement(); checked == 0 || float64(agree) < seedMinFixedAgreement*float64(checked) {
			return nil, false
		}
	} else if !historicalSeedAdmitted(&observed) {
		return nil, false
	}
	// The binarized channels are only needed once a docked secondary has to
	// be detected; the direct sample reads the balanced bitmap.
	return s.finishStreamObservation(
		bm, func() [3]*core.Bitmap { return frame.detectorChannels() }, &observed, r.f, r.src, capabilities,
	)
}

func (s *Stream) finishStreamObservation(bm *core.Bitmap, chFn func() [3]*core.Bitmap, observed *streamObservation,
	f finding, src image.Point, capabilities wire.Capabilities) (data *Message, ok bool) {
	correctionsBefore := s.work.correctionChains
	if observed.primary != nil {
		data, ok = s.finishObservation(
			bm, chFn, observed.symbols, observed.primary, f, src,
		)
	} else {
		data, ok = s.finishHistoricalObservation(bm, chFn, observed)
	}
	if ok {
		s.selectRoute(capabilities, observed.route)
	} else if s.work.correctionChains > correctionsBefore {
		s.failRoute(capabilities, observed.route)
	}
	return data, ok
}

// finishObservation admits at most one immutable observation per input frame,
// reuses a confirmed payload only under exact strong-sign agreement, and
// spends the frame's single correction chain either on materially changed
// aggregate evidence or on the current observation. Aggregate correction is
// primary-only; a docked-symbol result disables it for that content group.
func (s *Stream) finishObservation(bm *core.Bitmap, chFn func() [3]*core.Bitmap, symbols []core.DecodedSymbol,
	obs *decode.PrimaryObservation, f finding, src image.Point) (data *Message, ok bool) {
	if s.work.correctionChains >= 1 {
		return nil, false
	}
	snap := obs.Snapshot()
	grouped, changed := false, false
	if f.located && s.bankedGen != s.gen && len(snapshotFrameEvidence(snap)) > 0 {
		s.bankedGen = s.gen
		grouped, changed = s.group.admit(snap, f, src)
	}
	if grouped {
		frame := snapshotFrameEvidence(snap)
		if s.group.confirmedMatch(frame) {
			return cloneMessage(s.group.confirmedMessage), true
		}
		a := s.group.accumulatedEvidence()
		if changed && s.group.correctionDue(&a) {
			s.work.correctionChains++
			s.group.recordAttempt(&a)
			symbol, ret := snap.CorrectEvidence(a.signed)
			if ret != core.Success || symbol == nil {
				return nil, false
			}
			if symbol.Meta.DockedPosition != 0 {
				s.group.aggregateDisabled = true
				return nil, false
			}
			message, ok := decode.DecodeMessageVariant(symbol.Data, symbol.WireVariant)
			if !ok {
				return nil, false
			}
			data := &message
			s.group.confirm(data, &a)
			return data, true
		}
		if !changed && !s.group.aggregateDisabled {
			return nil, false
		}
		if a.frames > 1 && !s.group.aggregateDisabled {
			return nil, false
		}
	}

	s.work.correctionChains++
	if obs.CorrectPayload() != core.Success {
		return nil, false
	}
	var ch [3]*core.Bitmap
	if symbols[0].Meta.DockedPosition != 0 {
		ch = chFn()
	}
	data, ok = decodeSymbols(bm, ch, symbols, 1)
	if !ok {
		return nil, false
	}
	if symbols[0].Meta.DockedPosition == 0 && f.located && s.group.restart(snap, f, src) {
		a := s.group.accumulatedEvidence()
		s.group.confirm(data, &a)
	} else if symbols[0].Meta.DockedPosition != 0 {
		s.group = evidenceGroup{}
	}
	return data, true
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
func (s *Stream) refillPending(img image.Image, p *pyramid, baseSide int) {
	probeOn := img
	if p != nil {
		probeOn = p.level(0)
	}
	for _, deg := range detect.CoarseOrientationRungs(probeOn) {
		if deg == 0 {
			continue // the upright scan owns the zero angle
		}
		s.enqueue(streamHyp{side: coarsestSide(p, baseSide), deg: deg})
	}
	if p != nil {
		// The finer levels enqueue by dimensions alone; their pixels stay
		// unbuilt until a later frame actually tries the hypothesis.
		for i := 1; i < p.count(); i++ {
			s.enqueue(streamHyp{side: p.side(i)})
		}
	} else if baseSide < detect.SmallestVerifiableFrame() && s.enlargedDue() {
		// A single-scale frame too small to present a maximum primary symbol's
		// modules at the cross-check floor has no finer level to escalate to;
		// the enlarged detection scale is its one cross-frame fallback. It is
		// carried like any other hypothesis, so it only reaches the drain on a
		// frame whose cheaper attempts already found no finder.
		s.enqueue(streamHyp{side: enlargeFactor * baseSide, enlarged: true})
	}
}

// enlargedDue rate-limits the enlarged scale against the rest of the search.
// Being carried is not by itself a bound: a frame whose orientation probe
// yields no rung leaves the enlarged entry alone in the queue, which then
// refills and fires on every single frame. The enlargement costs the square of
// its factor in pixels, so spending it at most once per that many frames holds
// its amortized cost to one native-scale attempt per frame. The cadence counts
// frame generations, so it stays a pure function of the sequence.
func (s *Stream) enlargedDue() bool {
	return s.enlargedGen == 0 || s.gen-s.enlargedGen >= enlargeFactor*enlargeFactor
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
// the wanted scale, or the frame itself when there is no pyramid. Selection
// runs on the level dimensions; only the chosen level materializes.
func nearestLevelImage(img image.Image, p *pyramid, side int) image.Image {
	if p == nil {
		return img
	}
	best := 0
	for i := 1; i < p.count(); i++ {
		if absInt(p.side(i)-side) < absInt(p.side(best)-side) {
			best = i
		}
	}
	return p.level(best)
}

func streamGPULevel(p *pyramid, side int) (int, bool) {
	if p == nil {
		return 0, true
	}
	best := 0
	for i := 1; i < p.count(); i++ {
		if absInt(p.side(i)-side) < absInt(p.side(best)-side) {
			best = i
		}
	}
	return p.count() - 1 - best, true
}

// coarsestSide is the shorter side of the coarsest pyramid level, or of the
// frame itself when there is no pyramid.
func coarsestSide(p *pyramid, baseSide int) int {
	if p == nil {
		return baseSide
	}
	return p.side(0)
}

func shorterSide(img *image.NRGBA) int { return min(img.Rect.Dx(), img.Rect.Dy()) }

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
