package detect

import (
	"fmt"

	"github.com/srlehn/jabcode/internal/core"
)

// FinderFamily identifies one physical primary-finder signature.
type FinderFamily uint8

const (
	// FinderFamilyCurrent is the ISO/current-C finder signature.
	FinderFamilyCurrent FinderFamily = iota
	// FinderFamilyBSI is the primary finder signature defined by BSI
	// TR-03137 and retained by pre-v2.0 releases of the C reference. Its
	// classifier is compiled only when one of those wire variants is enabled.
	FinderFamilyBSI
	finderFamilyCount
)

// FinderFamilySet is the set of physical finder signatures located by one
// integrated detector pass.
type FinderFamilySet uint8

// Mask returns the one-signature set for family.
func (family FinderFamily) Mask() FinderFamilySet {
	if family >= finderFamilyCount {
		return 0
	}
	return 1 << family
}

// Has reports whether set contains family.
func (set FinderFamilySet) Has(family FinderFamily) bool {
	return family < finderFamilyCount && set&(1<<family) != 0
}

// CornerSource says where a completed quad's fourth corner came from. A
// construction is exact while the capture stays affine and a guess once it does
// not; the other values carry progressively stronger image evidence. The coarse
// consistency gates cannot tell those cases apart because interpolation completes
// a parallelogram whose convexity and opposite-edge agreement come from the
// arithmetic itself.
type CornerSource uint8

const (
	CornerFound       CornerSource = iota // every type had its own detection
	CornerConstructed                     // interpolated from the other three
	CornerPooled                          // a strict candidate another direction or pass found
	CornerSought                          // the local seek confirmed the estimate
	CornerContextual                      // a branch-confirmed candidate completed a strong triple
)

// String names the source for diagnostics.
func (c CornerSource) String() string {
	switch c {
	case CornerFound:
		return "found"
	case CornerConstructed:
		return "constructed"
	case CornerPooled:
		return "pooled"
	case CornerSought:
		return "sought"
	case CornerContextual:
		return "contextual"
	}
	return "unknown"
}

// FinderQuadHypothesis is one bounded primary-quad candidate carried from
// detection into sampling. Corner records the evidence behind its weakest
// corner so a construction cannot silently outrank an image-backed candidate.
type FinderQuadHypothesis struct {
	Patterns [4]FinderPattern
	Corner   CornerSource
}

// FinderFamilyScanStats records one scan direction's selection outcome. A pass
// sweeps several directions, so these are the only per-direction numbers; the
// pass-level copies below describe the published one alone.
type FinderFamilyScanStats struct {
	Degrees    float64      // scan direction, 0 for the row walk
	Preprune   [4]int       // group sizes before the 0.5*maxFound prune
	Preselect  [4]int       // FoundCount of each type's best pattern before that prune
	Selected   [4]int       // FoundCount of the selected pattern per type after the prune (0 = absent)
	Missing    int          // types absent after selection
	Status     int          // this direction's findPrimarySymbol status
	Corner     CornerSource // where the fourth corner came from
	Consistent bool         // whether the quad passed ConsistentFinderQuad
	Published  bool         // whether this is the quad the pass published
}

// FinderFamilyPassStats records one physical signature's counters inside a
// shared image pass. They are observation only and never influence detection.
//
// RawHits through CrossSurvivors accumulate over every scan direction the pass
// tried. The selection fields cannot be accumulated that way and would be
// last-writer-wins if they were assigned per direction, so they and Candidates
// mirror the published scan and the per-direction values live in Scans.
type FinderFamilyPassStats struct {
	RawHits        int    // n-1-1-1-m run-length hits (horizontal + conditional vertical scan)
	BranchBlue     int    // green seeds where the blue cross-check fired (-> {FP0,FP3} path)
	BranchRed      int    // green seeds where blue failed and the red cross-check fired (-> {FP1,FP2} path)
	RedColor       int    // red-path candidates passing the inner core-colour check (fp2found)
	RedClassified  int    // red-path candidates matched to fp1/fp2 by core-colour classification
	CrossSurvivors [4]int // candidates passing crossCheckPattern, by finder type
	Scans          []FinderFamilyScanStats
	Preprune       [4]int          // published scan's group sizes before the prune
	Selected       [4]int          // published scan's post-prune selection
	Missing        int             // published scan's absent types
	Status         int             // published scan's status
	Corner         CornerSource    // published scan's fourth-corner source
	Candidates     []FinderPattern // merged finder candidates this pass (pre-prune)
}

// publishScan mirrors one direction's selection up to the pass level, so the
// pass summary describes the quad detection actually used.
func (p *FinderFamilyPassStats) publishScan(i int) {
	if i < 0 || i >= len(p.Scans) {
		return
	}
	s := &p.Scans[i]
	s.Published = true
	p.Preprune, p.Selected = s.Preprune, s.Selected
	p.Missing, p.Status, p.Corner = s.Missing, s.Status, s.Corner
}

// FinderPassStats records one shared finder-detection pass. The embedded
// counters are for the current signature so existing diagnostics retain their
// field names; tagged builds add the optional BSI-era signature's counters to
// the same pass without enlarging the untagged structure.
type FinderPassStats struct {
	optionalFinderPassStats
	Label string // raw, avg-RGB, descreen or print input shared by all signatures
	FinderFamilyPassStats
}

// FinderConsensusStats records work done only by the fallback quad searches.
// These counters make expensive candidate volume visible without influencing
// the deterministic selection order.
type FinderConsensusStats struct {
	GeometryTuples      int
	GeometryScores      int
	InterpolatedTriples int
	InterpolatedSeeks   int
}

// DetectorStats aggregates finder-detection instrumentation across the raw,
// average-RGB, descreen and conditional print passes LocateFinders runs.
type DetectorStats struct {
	Passes    []FinderPassStats // one entry per prepared image pass
	RGBAvg    [3]float32        // retry thresholds from averagePixelValue, between passes
	Consensus FinderConsensusStats
}

// DetectorTrace retains the binarized channels used by each finder pass. Its
// entries align with DetectorStats.Passes. It is populated only when attached
// to a PrimaryDetector by the detailed read trace.
type DetectorTrace struct {
	PassInputs   []*core.Bitmap
	PassChannels [][3]*core.Bitmap
	FinderPasses []FinderPassTrace

	// RejectCounts and Rejections answer "what stops finders here" from the
	// detector itself rather than from a replica. The counts are the funnel;
	// the samples carry the run window, which is the only part that cannot be
	// reconstructed afterwards from a centre and a direction.
	//
	// Scope: the current-family directional scan only. The axis-aligned row walk
	// and the BSI signature do not report here, so a zero count is not evidence
	// that those paths accepted anything.
	RejectCounts [FinderStageCount]int
	Rejections   []FinderRejection

	rejectSeen map[rejectBucket]int
}

// FinderStage names where a candidate left the directional cross-check chain.
// The order is the chain's, so a histogram of RejectCounts reads as a funnel.
type FinderStage uint8

const (
	// StageBranchPattern and StageBranchColor are kept apart because a single
	// "the branch failed" bucket cannot distinguish a candidate whose blue and
	// red pattern walks both failed from one whose pattern walk passed and whose
	// core-colour check then rejected it. Merging them attributes colour
	// rejections to the run window of a walk that succeeded.
	StageBranchPattern FinderStage = iota
	StageBranchColor
	StageBranchModuleSize
	StageClassify
	StageChainBase // the perpendicular passed and moved the centre, the base walk then failed
	StageChainDiagonal
	StageChainModuleSize
	StageChainColor

	// FinderStageCount bounds RejectCounts, whose element count is part of that
	// exported field's type.
	FinderStageCount
)

// String names the stage for diagnostics.
func (s FinderStage) String() string {
	switch s {
	case StageBranchPattern:
		return "branch-pattern"
	case StageBranchColor:
		return "branch-color"
	case StageBranchModuleSize:
		return "branch-module-size"
	case StageClassify:
		return "classify"
	case StageChainBase:
		return "chain-base"
	case StageChainDiagonal:
		return "chain-diagonal"
	case StageChainModuleSize:
		return "chain-module-size"
	case StageChainColor:
		return "chain-color"
	}
	return "unknown"
}

// WalkReject names why a cross-check walk gave up. The run window alone does
// not say: a window that satisfies checkPatternCross is still rejected when the
// module size it implies exceeds the candidate's ceiling, and the two are
// opposite findings - "no finder here" against "a finder, but not one this
// candidate could be".
type WalkReject uint8

const (
	WalkNotWalked  WalkReject = iota // the stage compared sizes or colours instead of walking
	WalkIncomplete                   // the five-run window never completed inside the frame
	WalkTooWide                      // the middle runs outgrew the module ceiling mid-walk
	WalkSignature                    // checkPatternCross rejected the run ratios
	WalkModuleSize                   // the ratios held, the module size they imply did not
)

// String names the reason for diagnostics.
func (r WalkReject) String() string {
	switch r {
	case WalkNotWalked:
		return "not-walked"
	case WalkIncomplete:
		return "incomplete"
	case WalkTooWide:
		return "too-wide"
	case WalkSignature:
		return "signature"
	case WalkModuleSize:
		return "module-size"
	}
	return "unknown"
}

// FinderRejection is one representative rejected candidate. Runs is the five-run
// window of the walk that decided it and Reason is why that walk stopped, both
// empty where the stage compares module sizes or colours rather than walking.
//
// Pass is the index into DetectorStats.Passes. Retries re-binarize the same
// frame, so a rejection means nothing without knowing which binarization
// produced it.
// Centre and WalkDeg describe the walk that failed, not the candidate: the
// cross-checks refine the centre in place and the diagonal turns away from the
// scan direction, so reporting the candidate's own centre and base direction
// would name a position and a line the failing walk never sampled. Confirms is
// the diagonal confirmation count, which distinguishes "no diagonal held" from
// "one held and one was needed".
type FinderRejection struct {
	Stage    FinderStage
	Pass     int
	Typ      int
	Channel  int // channel walked or sampled, -1 where the stage is not per-channel
	BaseDeg  float64
	WalkDeg  float64
	Centre   core.PointF
	Module   float64
	Confirms int
	Runs     [5]int
	Reason   WalkReject
}

// rejectBucket keeps retention representative rather than merely first-come.
// Rejections arrive in scan order, so a plain first-N fills entirely from
// whichever image corner the sweep starts in - which on a noisy capture is
// background clutter, and never the one finder under investigation. Bucketing
// by pass, stage, type, walk direction and a coarse image cell is what makes
// "which stage discarded the candidate at that corner, in which binarization"
// answerable from the samples.
//
// The direction is the walk's own, not the scan's: the two 45-degree turns of a
// scan direction are separate walks failing for separate reasons, and keying on
// the base they share lets whichever ran first spend the bucket for both.
//
// The reason and the diagonal confirmation count are in the key for the same
// reason they are in the record. Common failures arrive first and in bulk, so a
// key blind to them lets two ordinary signature misses spend the bucket and
// leave the rare oversized-module or one-confirmation case at that spot with no
// sample at all - the distinctions the fields were added to preserve.
type rejectBucket struct {
	pass     int
	stage    FinderStage
	typ      int
	channel  int
	walkDeg  float64
	reason   WalkReject
	confirms int
	cellX    int
	cellY    int
}

// Retention bounds. The grid is fine enough that one cell is a neighbourhood
// rather than a quadrant, the per-bucket count is small because the buckets are
// many, and maxRejectionSamples is a hard ceiling so a pathological frame cannot
// turn a diagnostic read into a memory problem.
const (
	rejectGridSide             = 16
	maxRejectionSamplesPerCell = 2
	maxRejectionSamples        = 20000
)

// reject counts one rejection and keeps a bounded, spatially spread sample of
// each kind. img supplies the frame extent the coarse cell is relative to.
func (t *DetectorTrace) reject(img *core.Bitmap, r FinderRejection) {
	if t == nil {
		return
	}
	t.RejectCounts[r.Stage]++
	if len(t.Rejections) >= maxRejectionSamples {
		return
	}
	k := rejectBucket{
		pass: r.Pass, stage: r.Stage, typ: r.Typ, channel: r.Channel,
		walkDeg: r.WalkDeg, reason: r.Reason, confirms: r.Confirms,
		cellX: cellIndex(r.Centre.X, img.Width),
		cellY: cellIndex(r.Centre.Y, img.Height),
	}
	if t.rejectSeen == nil {
		t.rejectSeen = make(map[rejectBucket]int)
	}
	t.rejectSeen[k]++
	if t.rejectSeen[k] <= maxRejectionSamplesPerCell {
		t.Rejections = append(t.Rejections, r)
	}
}

func cellIndex(v float64, extent int) int {
	if extent <= 0 {
		return 0
	}
	i := int(v * rejectGridSide / float64(extent))
	return min(max(i, 0), rejectGridSide-1)
}

// FinderPassTrace retains the requested signatures, each successful
// signature's published quad, and the quad every other scan direction offered.
// It is allocated only for an attached diagnostic trace, so ordinary decoding
// does not retain this rendering state.
type FinderPassTrace struct {
	Families FinderFamilySet
	Finders  [finderFamilyCount][]FinderPattern
	Scans    []FinderScanTrace
}

// FinderScanTrace is one scan direction's selected quad. The sweep reuses its
// working patterns for the next direction, so a quad that was found and not
// published exists nowhere afterwards unless it is copied here - and "the
// direction that would have won was never compared" is a failure mode this
// detector has already had.
//
// Scan indexes the direction's entry in that family's FinderFamilyScanStats,
// which carries its angle, status and published flag; keeping one index rather
// than copies of those keeps the two records from disagreeing.
type FinderScanTrace struct {
	Family FinderFamily
	Scan   int
	Quad   []FinderPattern // the four selected patterns, entries with FoundCount 0 absent
	Pre    []FinderPattern // the same four before the outvoted-type prune
}

// PrimaryDetector orchestrates primary-symbol finder detection over the three
// binarized channels. Its findPrimarySymbol/selectBestPatterns/scanPatternVertical
// methods populate stats, the single source of truth for the diagnostic. The Ch
// field is a by-value [3]*core.Bitmap: the retry's re-binarization (LocateFinders)
// is scoped to this detector and never leaks into secondary decoding.
type PrimaryDetector struct {
	BM         *core.Bitmap
	Ch         [3]*core.Bitmap
	Mode       int
	FPs        []FinderPattern
	Candidates []FinderPattern // last pass's pre-prune candidates, for the geometric quad fallback
	Stats      DetectorStats
	// scanTraces collects the current pass's per-direction quads until the pass
	// records its trace entry. Nil unless a trace is attached.
	scanTraces []FinderScanTrace
	// activeFamily is the signature SelectFinderFamily last published into FPs.
	activeFamily    FinderFamily
	hasActiveFamily bool
	Trace           *DetectorTrace

	familyResults [finderFamilyCount]finderFamilyResult

	// familyPassCandidates unions each physical signature's finder candidates
	// across every binarization pass that ran, deduplicating near-identical
	// hits. The per-pass familyResults candidates only ever see the pass that
	// located, but a finder at the module-scale floor can surface in a
	// different pass than the one that locates; SelectFinderFamily hands this
	// union to the geometric consensus so it can assemble a quad from corners
	// no single pass found together. Working state, kept off Stats so those
	// stay observation-only. Reset per detection in locateInitialFinderFamilies.
	familyPassCandidates [finderFamilyCount][]FinderPattern

	// contextualCandidates retains current-family finder groups that repeated
	// within one scan direction after branch and colour classification, but failed
	// the standalone cross-check chain. A later direction may locate the other
	// three corners, so the qualified groups survive the direction boundary while
	// unsupported individual crossings do not.
	contextualCandidates []FinderPattern

	// Quit, when set, is polled between binarization passes; once it reports
	// true the search abandons its remaining retries and fails. The resolution
	// pyramid cancels levels that can no longer win this way, so an abandoned
	// level stops burning cores within one pass instead of finishing its whole
	// retry ladder in the background.
	Quit func() bool

	// seedModules and bsiFamilySeedModules collect the per-seed module-size
	// estimate of every raw n-1-1-1-m hit for their respective signatures
	// across this detection's passes. Each signature gates its own print retry
	// and derives its own low-pass radius, so enabling another signature cannot
	// perturb the established current-family retry. This is working state kept
	// off Stats so those stay observation-only.
	seedModules          []float64
	bsiFamilySeedModules []float64

	// printPass marks the print-level retry passes, where the finder
	// cross-checks scale their pixel tolerances with the module size:
	// colorant-plane misregistration shifts each channel's pattern and
	// fringes every module boundary by a module fraction, which the fixed
	// 3 px slack of the ported walks cannot absorb. Off elsewhere so the
	// default passes stay byte-compatible. printDetected records that the
	// successful pass was a print-level one, making the per-channel
	// sampling-offset estimate available to the sampler.
	printPass     bool
	printDetected bool
	passFamilies  FinderFamilySet

	// AxisAlignedScan confines the finder search to image rows, suppressing the
	// directional retry. The coarse orientation probe needs it: that probe
	// measures a frame's orientation by pre-rotating it and comparing which
	// rotation the row walk likes best, so a scan that locates the symbol at
	// every orientation would leave it ranking noise. Ordinary reads leave it
	// off and let one prepared frame answer every direction.
	AxisAlignedScan bool

	// materializeBitmap supplies balanced RGBA pixels only when a compact-mask
	// finder pass proves they are needed for missing-finder confirmation,
	// geometry or sampling. CPU detectors already carry pixels and leave it nil.
	materializeBitmap func() error
	materializeErr    error

	// materializeChannels fills the current pass's binarized channel bitmaps
	// from the downloaded packed mask words. A pass whose families all replay
	// device chain outcomes never reads mask pixels, so the expansion runs
	// only for the CPU walk and chain fallbacks, the vertical scans, trace
	// recording and a located success whose downstream geometry and sampling
	// consume the channels. CPU preparers return full bitmaps and leave it
	// nil. Single-shot per pass, like the pass's row hits.
	materializeChannels func() error
	channelExpansions   int
	materializeChanErr  error

	// detachChannels snapshots the current pass's downloaded packed mask
	// words so the deferred expansion survives the GPU route's context
	// lease. Set only by GPU-built detectors; the locate boundary invokes it
	// through detachLocatedChannels on success, while the lease is held.
	detachChannels func() error

	// rowHits carries the device row scan's raw hits for the next
	// findPrimaryFamilies call, which consumes them instead of walking the
	// binarized rows itself; the hits are bit-identical to that walk. Nil or
	// invalid (overflowed) hits keep the CPU walk. Single-shot per pass.
	rowHits *finderPassRowHits
}

type finderFamilyResult struct {
	fps           []FinderPattern
	candidates    []FinderPattern
	alternatives  []FinderQuadHypothesis
	channels      [3]*core.Bitmap
	status        int
	corner        CornerSource
	printDetected bool
	// scan indexes the direction's entry in the family's Scans, so publishing
	// a pick can name which direction produced it. -1 where no scan ran.
	scan int
}

type finderPassPreparer interface {
	averagePixelValue([]FinderPattern) ([3]float32, error)
	estimatePitch() (int, int, error)
	// prepare builds one retry pass's input and binarized channels.
	// scanChannels selects the channels whose finder row scan should run
	// where the masks live; a preparer without a device scan returns nil
	// hits and the detector walks the rows itself. A non-nil materialize
	// result means the channel bitmaps are shape-only until it runs; it is
	// valid until the preparer's next pass.
	prepare(rx, ry int, thresholds []float32, printLevels bool, scanChannels uint32) (*core.Bitmap, [3]*core.Bitmap, *finderPassRowHits, func() error, error)
}

type cpuFinderPassPreparer struct {
	bm *core.Bitmap
	// quit carries the detector's cancellation hook into the full-frame
	// descreen and binarization, which are the longest uninterruptible
	// stretch of a pass. A cancelled prepare reports no channels; the caller
	// polls the same hook and returns before it would read them.
	quit func() bool
}

func (preparer cpuFinderPassPreparer) averagePixelValue(fps []FinderPattern) ([3]float32, error) {
	return averagePixelValue(preparer.bm, fps), nil
}

func (preparer cpuFinderPassPreparer) estimatePitch() (int, int, error) {
	px, py := EstimatePitch(preparer.bm)
	return px, py, nil
}

func (preparer cpuFinderPassPreparer) prepare(
	rx, ry int,
	thresholds []float32,
	printLevels bool,
	scanChannels uint32,
) (*core.Bitmap, [3]*core.Bitmap, *finderPassRowHits, func() error, error) {
	input := preparer.bm
	if rx > 0 || ry > 0 {
		input = descreen(input, rx, ry, preparer.quit)
		if input == nil {
			return nil, [3]*core.Bitmap{}, nil, nil, nil
		}
	}
	ch, ok := binarizeRGB(input, thresholds, printLevels, preparer.quit)
	if !ok {
		return nil, [3]*core.Bitmap{}, nil, nil, nil
	}
	return input, ch, nil, nil, nil
}

// SelectFinderFamily selects one located signature as the detector's active
// finder list for geometry and sampling. It returns false when that signature
// did not form a usable finder quad in the last integrated search.
func (d *PrimaryDetector) SelectFinderFamily(family FinderFamily) bool {
	if family >= finderFamilyCount {
		return false
	}
	result := &d.familyResults[family]
	d.FPs = result.fps
	d.Candidates = d.familyPassCandidates[family]
	d.activeFamily, d.hasActiveFamily = family, true
	if result.status == core.Success {
		d.Ch = result.channels
		d.printDetected = result.printDetected
	}
	return result.status == core.Success
}

// FinderQuadHypotheses returns the located quad and any contextual alternatives
// in sampling order. Image-backed alternatives precede an affine construction;
// otherwise the detector's published quad remains first.
func (d *PrimaryDetector) FinderQuadHypotheses(family FinderFamily) []FinderQuadHypothesis {
	if family >= finderFamilyCount {
		return nil
	}
	result := &d.familyResults[family]
	if result.status != core.Success || len(result.fps) < 4 {
		return nil
	}
	var primary FinderQuadHypothesis
	copy(primary.Patterns[:], result.fps[:4])
	primary.Corner = result.corner

	hypotheses := make([]FinderQuadHypothesis, 0, 1+len(result.alternatives))
	if result.corner == CornerConstructed {
		hypotheses = append(hypotheses, result.alternatives...)
		hypotheses = append(hypotheses, primary)
	} else {
		hypotheses = append(hypotheses, primary)
		hypotheses = append(hypotheses, result.alternatives...)
	}
	return hypotheses
}

// ActiveFinderFamily reports which signature produced the current FPs, and
// whether one was ever selected. A quad means nothing without it: the two
// families have different finder geometry, so attributing one family's quad to
// the other misreads the geometry that produced every downstream number.
func (d *PrimaryDetector) ActiveFinderFamily() (FinderFamily, bool) {
	return d.activeFamily, d.hasActiveFamily
}

// PrintDetected reports whether the successful finder pass was a print-level
// one, which is the gate for the per-channel sampling-offset search.
func (d *PrimaryDetector) PrintDetected() bool { return d.printDetected }

// ccSlack returns the cross-check pixel slack for a candidate of the given
// module size: the ported constant 3 normally, half a module (misregistration
// fringes scale with the module) in the print-level passes.
func (d *PrimaryDetector) ccSlack(moduleSize float64) int {
	if d.printPass {
		return max(3, int(moduleSize/2+0.5))
	}
	return 3
}

// pass returns the current (last-appended) finder pass's stats.
func (d *PrimaryDetector) pass() *FinderPassStats {
	return &d.Stats.Passes[len(d.Stats.Passes)-1]
}

// recordScanQuad retains one scan direction's selection, before and after the
// prune. A failed selection is retained too: a direction that found three true
// corners and was discarded is the case worth seeing, and it is exactly the one
// a success-only record throws away.
func (d *PrimaryDetector) recordScanQuad(family FinderFamily, scan int, fps []FinderPattern, pre [4]FinderPattern) {
	if d.Trace == nil || len(fps) < 4 {
		return
	}
	d.scanTraces = append(d.scanTraces, FinderScanTrace{
		Family: family, Scan: scan,
		Quad: append([]FinderPattern(nil), fps[:4]...),
		Pre:  append([]FinderPattern(nil), pre[:]...),
	})
}

func (d *PrimaryDetector) recordTracePass(input *core.Bitmap) {
	if d.Trace == nil {
		return
	}
	// Diagnostics keep every pass's channels, so a traced pass materializes.
	d.ensureChannels()
	d.Trace.PassInputs = append(d.Trace.PassInputs, input)
	d.Trace.PassChannels = append(d.Trace.PassChannels, d.Ch)
	pass := FinderPassTrace{Families: d.passFamilies, Scans: d.scanTraces}
	d.scanTraces = nil
	for family := FinderFamily(0); family < finderFamilyCount; family++ {
		result := &d.familyResults[family]
		if result.status == core.Success {
			pass.Finders[family] = append([]FinderPattern(nil), result.fps[:4]...)
		}
	}
	d.Trace.FinderPasses = append(d.Trace.FinderPasses, pass)
}

// EnsureChannels fills the located pass's channel bitmaps with mask pixels
// on first need and reports whether channel pixels are available. CPU-built
// detectors always carry full channels; a GPU-located detector expands its
// packed snapshot only when a consumer actually reads mask pixels, such as
// alignment resampling, the docked traversal or a historical wire route.
func (d *PrimaryDetector) EnsureChannels() bool { return d.ensureChannels() }

// ChannelExpansionCount reports how many times deferred mask channels have
// been materialized for this detector. It makes residency tests distinguish a
// bounded Pixel read from a full-mask expansion.
func (d *PrimaryDetector) ChannelExpansionCount() int {
	if d == nil {
		return 0
	}
	return d.channelExpansions
}

// detachLocatedChannels makes a located GPU pass's deferred mask expansion
// outlive its route: the packed words are snapshotted now, while the route
// still holds its device lease, and the expansion stays deferred until a
// downstream consumer reads mask pixels. Materialized passes and CPU
// detectors no-op.
func (d *PrimaryDetector) detachLocatedChannels() error {
	if d.materializeChannels == nil || d.detachChannels == nil {
		return nil
	}
	return d.detachChannels()
}

// ensureChannels fills the current pass's shape-only channel bitmaps with
// mask pixels on first need. It reports false only when materialization
// failed, in which case the pass deterministically fails detection.
func (d *PrimaryDetector) ensureChannels() bool {
	if d == nil || d.Ch[0] == nil {
		return false
	}
	if d.Ch[0].Pix != nil || d.materializeChannels == nil {
		return d.Ch[0].Pix != nil
	}
	materialize := d.materializeChannels
	d.materializeChannels = nil
	d.channelExpansions++
	if err := materialize(); err != nil {
		d.materializeChanErr = err
		return false
	}
	return d.Ch[0].Pix != nil
}

func (d *PrimaryDetector) ensureBitmap() bool {
	if d == nil || d.BM == nil {
		return false
	}
	want := d.BM.Width * d.BM.Height * d.BM.Channels
	if want > 0 && len(d.BM.Pix) >= want {
		return true
	}
	if d.materializeBitmap == nil {
		return false
	}
	materialize := d.materializeBitmap
	d.materializeBitmap = nil
	if err := materialize(); err != nil {
		d.materializeErr = err
		return false
	}
	return len(d.BM.Pix) >= want
}

// Quitting reports whether an installed Quit hook has cancelled this search.
// Consumers poll it at their own stage boundaries so a route that already lost
// stops before work whose result can no longer be used.
func (d *PrimaryDetector) Quitting() bool {
	return cancelled(d.Quit)
}

// cancelled polls an optional quit hook for the full-frame pixel passes, which
// take the hook directly rather than through a detector: they are the longest
// stretch a losing route has no way to be stopped in, and one of them runs
// before the detector that would own the hook even exists.
func cancelled(quit func() bool) bool { return quit != nil && quit() }

// LocateFinders locates the current ISO/current-C finder signature. Optional
// signatures are not enabled by this compatibility wrapper.
func (d *PrimaryDetector) LocateFinders() bool {
	return d.LocateFinderFamilies(FinderFamilyCurrent.Mask()).Has(FinderFamilyCurrent)
}

// LocateInitialFinderFamilies runs only the first balanced-image finder pass.
// It is the compact-mask boundary used by the automatic GPU path: a successful
// pass can materialize pixels for geometry and sampling, while a failed pass
// falls back to the complete CPU retry ladder without changing its behavior.
func (d *PrimaryDetector) LocateInitialFinderFamilies(wanted FinderFamilySet) FinderFamilySet {
	found, _, _, _ := d.locateInitialFinderFamilies(wanted)
	return found
}

// LocateFinderFamilies runs one finder traversal per prepared image pass and
// classifies every requested physical signature inside that traversal. The
// retry re-binarizes d.Ch in place; because the channel array is held by value,
// that swap is scoped to this detector and does not propagate to secondary
// detection. The C reference differs here: its detectMaster overwrites the
// caller's channel array, so it detects docked secondaries on the retry's
// re-binarization while this port detects them on the first-pass channels. The
// two can diverge only for a multi-symbol code whose primary needed the retry;
// the wire format is unaffected.
func (d *PrimaryDetector) LocateFinderFamilies(wanted FinderFamilySet) FinderFamilySet {
	found, _ := d.locateFinderFamilies(wanted, cpuFinderPassPreparer{bm: d.BM, quit: d.Quit})
	return found
}

func (d *PrimaryDetector) locateFinderFamilies(
	wanted FinderFamilySet,
	preparer finderPassPreparer,
) (FinderFamilySet, error) {
	// Ports the retry orchestration of detectMaster in detector.c.
	found, wantCurrent, wantBSI, stop := d.locateInitialFinderFamilies(wanted)
	if stop {
		return found, nil
	}
	maxSurvivors := d.familySurvivors(wantCurrent, wantBSI)

	scanChannels := finderScanChannelMask(wantCurrent, wantBSI)

	// Retry 1: re-binarize using adaptive thresholds from around the found patterns.
	rgbAvg, err := preparer.averagePixelValue(d.retrySeedFinders(wantCurrent, wantBSI))
	if err != nil {
		return 0, err
	}
	d.Stats.RGBAvg = rgbAvg
	input, ch2, hits, materialize, err := preparer.prepare(0, 0, rgbAvg[:], false, scanChannels)
	if err != nil {
		return 0, err
	}
	// A pass cancelled inside its own preprocessing returns no channels, so
	// every prepare is followed by the same poll before anything reads them.
	if d.Quitting() {
		return 0, nil
	}
	d.Ch[0], d.Ch[1], d.Ch[2] = ch2[0], ch2[1], ch2[2]
	d.rowHits = hits
	d.materializeChannels = materialize
	d.materializeChanErr = nil
	found = d.findPrimaryFamilies(wantCurrent, wantBSI)
	d.pass().Label = "avg-RGB retry"
	d.recordTracePass(input)
	if found != 0 {
		d.selectLocatedFinderFamily(found)
		return found, nil
	}
	mergeFamilySurvivors(&maxSurvivors, d.familySurvivors(wantCurrent, wantBSI))

	// Retry 2 (descreen): screen captures inject the display's subpixel/diode lattice
	// and moiré, which can leave the raw and avg-RGB passes without enough surviving
	// finders. Estimate the lattice pitch per image and low-pass ≈ one grid cell (then
	// a coarser pass) before binarizing - the kernel is derived, not a fixed radius.
	// bm is left untouched so colour sampling still reads the original pixels; the
	// d.Ch swap stays primary-scoped.
	// The pitch estimate is a full-frame autocorrelation, and it sits between
	// the avg-RGB pass and the first descreen poll, so a route cancelled during
	// that pass would otherwise pay for it before noticing. Nothing downstream
	// of a cancelled locate reads the pitch, unlike the geometry a cancelled
	// route still publishes, so this poll can precede the work rather than
	// follow it.
	if d.Quitting() {
		return 0, nil
	}
	px, py, err := preparer.estimatePitch()
	if err != nil {
		return 0, err
	}
	for _, r := range descreenSchedule(px, py) {
		if d.Quitting() {
			return 0, nil
		}
		filtered, chN, hitsN, materializeN, err := preparer.prepare(r[0], r[1], nil, false, scanChannels)
		if err != nil {
			return 0, err
		}
		if d.Quitting() {
			return 0, nil
		}
		d.Ch[0], d.Ch[1], d.Ch[2] = chN[0], chN[1], chN[2]
		d.rowHits = hitsN
		d.materializeChannels = materializeN
		d.materializeChanErr = nil
		found = d.findPrimaryFamilies(wantCurrent, wantBSI)
		d.pass().Label = fmt.Sprintf("descreen %dx%d", r[0], r[1])
		d.recordTracePass(filtered)
		if found != 0 {
			d.selectLocatedFinderFamily(found)
			return found, nil
		}
		mergeFamilySurvivors(&maxSurvivors, d.familySurvivors(wantCurrent, wantBSI))
	}

	// Retry 3 (print levels): subtractive print colours are dark - a printed
	// blue's own channel can sit below the block mean, so the default black
	// gate swallows whole colour modules as black. When the failed passes
	// show the print signature - raw run-length seeds by the hundred with
	// cross-check survivors near zero - re-binarize with the black gate on
	// the block-floor anchor, then once more on a copy low-passed at a
	// quarter of the seeds' own module-size estimate, which fuses halftone
	// cells, dither grain and colorant-plane fringes.
	currentPrint := wantCurrent && len(d.seedModules) >= printRetryMinSeeds &&
		maxSurvivors[FinderFamilyCurrent] <= printRetryMaxSurvivors
	bsiPrint := wantBSI && len(d.bsiFamilySeedModules) >= printRetryMinSeeds &&
		maxSurvivors[FinderFamilyBSI] <= printRetryMaxSurvivors
	if currentPrint || bsiPrint {
		// Two binarizations, and the first success wins, so order matters:
		// on coarse grain the sharp pass can succeed with a wrong finder
		// quad and poison the downstream side estimate - the low-passed one
		// lands the true geometry and goes first. On small modules the
		// integer blur radius collapses to a large module fraction and
		// shifts the finder centres instead, so there the sharp pass leads.
		// The radius itself separates the regimes: quantization dominates
		// it below printBlurLeadRadius.
		printSeeds := d.seedModules
		printCurrent := wantCurrent
		if !currentPrint {
			printSeeds = d.bsiFamilySeedModules
			printCurrent = false
		}
		r := max(1, int(seedModuleScale(printSeeds)/4+0.5))
		passes := [2]struct {
			label  string
			rx, ry int
		}{
			{label: fmt.Sprintf("print blurred r=%d", r), rx: r, ry: r},
			{label: "print sharp"},
		}
		if r < printBlurLeadRadius {
			passes[0], passes[1] = passes[1], passes[0]
		}
		d.printPass = true
		defer func() { d.printPass = false }()
		for _, p := range passes {
			if d.Quitting() {
				return 0, nil
			}
			input, chP, hitsP, materializeP, err := preparer.prepare(
				p.rx, p.ry, nil, true, finderScanChannelMask(printCurrent, wantBSI),
			)
			if err != nil {
				return 0, err
			}
			if d.Quitting() {
				return 0, nil
			}
			d.Ch[0], d.Ch[1], d.Ch[2] = chP[0], chP[1], chP[2]
			d.rowHits = hitsP
			d.materializeChannels = materializeP
			d.materializeChanErr = nil
			found = d.findPrimaryFamilies(printCurrent, wantBSI)
			d.pass().Label = p.label
			d.recordTracePass(input)
			if found != 0 {
				d.selectLocatedFinderFamily(found)
				return found, nil
			}
		}
	}
	return 0, nil
}

func (d *PrimaryDetector) locateInitialFinderFamilies(
	wanted FinderFamilySet,
) (found FinderFamilySet, wantCurrent, wantBSI, stop bool) {
	wantCurrent = wanted.Has(FinderFamilyCurrent)
	wantBSI = wanted.Has(FinderFamilyBSI) && bsiFamilyFinderEnabled
	d.seedModules = d.seedModules[:0]
	d.bsiFamilySeedModules = d.bsiFamilySeedModules[:0]
	for i := range d.familyPassCandidates {
		d.familyPassCandidates[i] = d.familyPassCandidates[i][:0]
	}
	d.contextualCandidates = d.contextualCandidates[:0]
	d.printDetected = false
	clear(d.familyResults[:])
	if d.Trace != nil {
		d.Trace.PassInputs = d.Trace.PassInputs[:0]
		d.Trace.PassChannels = d.Trace.PassChannels[:0]
		d.Trace.FinderPasses = d.Trace.FinderPasses[:0]
		d.Trace.Rejections = d.Trace.Rejections[:0]
		clear(d.Trace.RejectCounts[:])
		clear(d.Trace.rejectSeen)
	}
	if d.Quitting() {
		return 0, wantCurrent, wantBSI, true
	}
	found = d.findPrimaryFamilies(wantCurrent, wantBSI)
	d.pass().Label = "raw"
	d.recordTracePass(d.BM)
	if wantCurrent && d.familyResults[FinderFamilyCurrent].status == core.FatalError {
		return 0, wantCurrent, wantBSI, true
	}
	if found != 0 {
		d.selectLocatedFinderFamily(found)
		return found, wantCurrent, wantBSI, true
	}
	if d.Quitting() {
		return 0, wantCurrent, wantBSI, true
	}
	return 0, wantCurrent, wantBSI, false
}

func (d *PrimaryDetector) selectLocatedFinderFamily(found FinderFamilySet) {
	// Downstream geometry, version detection and sampling read the balanced
	// pixels, not the mask channels, so a located GPU pass keeps its masks
	// packed here; the locate boundary snapshots them and the few consumers
	// that do read mask pixels expand them on first need.
	if found.Has(FinderFamilyCurrent) {
		d.SelectFinderFamily(FinderFamilyCurrent)
		return
	}
	d.SelectFinderFamily(FinderFamilyBSI)
}

func (d *PrimaryDetector) familySurvivors(wantCurrent, wantBSI bool) [finderFamilyCount]int {
	var n [finderFamilyCount]int
	if wantCurrent {
		n[FinderFamilyCurrent] = len(d.familyResults[FinderFamilyCurrent].candidates)
	}
	if wantBSI {
		n[FinderFamilyBSI] = len(d.familyResults[FinderFamilyBSI].candidates)
	}
	return n
}

func mergeFamilySurvivors(dst *[finderFamilyCount]int, src [finderFamilyCount]int) {
	for family := range finderFamilyCount {
		dst[family] = max(dst[family], src[family])
	}
}

func (d *PrimaryDetector) retrySeedFinders(wantCurrent, wantBSI bool) []FinderPattern {
	current := &d.familyResults[FinderFamilyCurrent]
	if wantCurrent {
		return current.fps
	}
	if wantBSI {
		return d.familyResults[FinderFamilyBSI].fps
	}
	return nil
}
