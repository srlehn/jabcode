//go:build !js

package detect

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/phaseprobe"
)

// The device form of sweepDirection's seek and packed-mask chain stages. The
// fused window kernel produces every along-line candidate; once the shared
// chain kernels are ready, a second dispatch performs the branch walks,
// classification and final cross-check over the other mask channels. Source
// RGB signal validation remains on the host.
//
// The kernel is dispatched with FLAG_SKIP_CROSS_CHECK, so no off-line walk runs
// and every window that passes along the line is reported. The walks exist and
// work, but they confirm on the seek channel while the host chain confirms on
// the other two, and measurement on real captures shows them rejecting
// candidates the host chain keeps. Filtering on the device is where this is
// going; it is not what ships first.

// finderRunsGeometry is the sweep a directional kernel is asked to walk: the
// step along a line, the perpendicular basis the lines are indexed on, and how
// many lines and samples that gives. It mirrors the Params block the shader
// reads, so one struct describes the sweep on both sides.
type finderRunsGeometry struct {
	dx, dy     float32
	nx, ny     float32
	qLo, qStep float32
	lineCount  int
	lineLength int
}

// gpuFinderDirectionalCapacity is how many records one direction may produce.
// A 12 MP capture measures around twenty thousand per angle, so this is an
// order of magnitude of headroom; a sweep that exceeds it reports the overflow
// and the caller walks that direction on the CPU rather than acting on a
// truncated list.
const gpuFinderDirectionalCapacity = 1 << 18

// gpuFinderDirectionalCompactCapacity bounds the candidates one direction may
// hand back after the device has compacted them, and is set to everything the
// host could consume: a scan retains at most maxFinderPatterns survivors and
// maxContextualFinderSeeds weak candidates, so a longer list could not be acted
// on in full however large this buffer were.
//
// Sizing it by what the host publishes instead was measured wrong. Survivors
// are indeed few, but contextual seeds are not - a full-resolution level of a
// 12 MP display capture compacts over eleven thousand candidates in a single
// direction - and an overflow does not degrade that direction gracefully. It
// abandons the device for the whole level, which then walks every direction on
// the host and pulls the entire balanced frame back to answer the finder colour
// signal. The overflow was costing a 30 MB download and tens of thousands of
// host cross-checks, not a few candidates.
const gpuFinderDirectionalCompactCapacity = maxFinderPatterns + maxContextualFinderSeeds

// gpuFinderDirectionalPrefixCapacity is how much of that list rides back in the
// sweep's own submission. It covers the directions a capture actually produces;
// a crowded one costs a second, exactly sized download rather than making every
// direction carry the full bound.
const gpuFinderDirectionalPrefixCapacity = 4096

// The summary block: the compacted count, the raw hit count, four branch
// counters, the raw record count the dispatch-argument kernel publishes, and
// the module-size histogram.
const (
	gpuFinderDirectionalSummaryCompacted = 0
	gpuFinderDirectionalSummaryRequired  = 6
	gpuFinderDirectionalSummaryHeader    = 7
	gpuFinderDirectionalSummaryBuckets   = moduleSeedsBuckets
	gpuFinderDirectionalSummaryWords     = gpuFinderDirectionalSummaryHeader + gpuFinderDirectionalSummaryBuckets
	gpuFinderDirectionalSummaryBytes     = gpuFinderDirectionalSummaryWords * 4
)

const (
	gpuFinderDirectionalRecordBytes  = gpuFinderDirectionalCapacity * finderWindowRecordWords * 4
	gpuFinderDirectionalOutcomeBytes = gpuFinderDirectionalCompactCapacity * gpuFinderChainOutcomeWords * 4
	gpuFinderDirectionalPrefixBytes  = gpuFinderDirectionalPrefixCapacity * gpuFinderChainOutcomeWords * 4
	// Three workgroup counts for the indirect dispatch plus the invocation
	// bound the chain kernel reads back out of them.
	gpuFinderDirectionalArgsBytes = 4 * 4
	// The common block, six sweep-basis scalars and four directions of three
	// scalars each, all native f32.
	gpuFinderDirectionalChainParamsBytes = gpuFinderChainParamsSize + 6*4 + 4*3*4
	gpuFinderDirectionalRetainedBytes    = gpuFinderDirectionalRecordBytes +
		finderWindowCounterCount*4 + finderScanParamsBytes +
		gpuFinderDirectionalOutcomeBytes + gpuFinderDirectionalChainParamsBytes +
		gpuFinderDirectionalSummaryBytes + gpuFinderDirectionalArgsBytes
)

// scanDirectionHits sweeps one direction over the still resident packed masks
// and returns its raw signatures plus device-chain outcomes when available. A
// nil result is not an error: it means this pass has no directional kernel and
// the caller should sweep on the CPU.
func (b *gpuBinarizer) scanDirectionHits(
	width, height int,
	dir scanDirection,
	step, channel int,
) (finderDirSweep, error) {
	var sweep finderDirSweep
	if b == nil || b.device == nil || b.packedMasks == nil {
		return sweep, nil
	}
	if err := b.kernels.directionalFinderChainError(); err != nil {
		return sweep, err
	}
	kernel, err := b.kernels.finderWindows(finderScanInterleaved)
	if err != nil {
		return sweep, err
	}
	if err := b.ensureDirectionalBuffers(kernel); err != nil {
		return sweep, err
	}
	geom := directionalSweepGeometry(width, height, dir, step)
	if geom.lineCount <= 0 {
		return sweep, nil
	}
	chained := !b.scanOnly && b.kernels.directionalFinderChainReady()
	if chained {
		if err := b.ensureDirectionalChainBuffers(); err != nil {
			return sweep, err
		}
	}

	recorder, err := b.device.NewRecorder()
	if err != nil {
		return sweep, fmt.Errorf("jabcode: create GPU directional scan recorder: %w", err)
	}
	defer recorder.Abort()
	params := directionalScanParams(width, height, uint32(1)<<uint(channel), geom)
	if err := recorder.Update(b.dirParams, 0, params); err != nil {
		return sweep, fmt.Errorf("jabcode: update GPU directional scan parameters: %w", err)
	}
	var zero [finderWindowCounterCount * 4]byte
	if err := recorder.Update(b.dirCounters, 0, zero[:]); err != nil {
		return sweep, fmt.Errorf("jabcode: clear GPU directional scan counters: %w", err)
	}
	if err := recorder.Barrier(b.packedMasks); err != nil {
		return sweep, fmt.Errorf("jabcode: synchronize GPU packed masks for the directional scan: %w", err)
	}
	if err := recorder.Dispatch(kernel, b.dirBindings, vulki.Workgroups{
		X: uint32(geom.lineCount), Y: 3, Z: 1,
	}); err != nil {
		return sweep, fmt.Errorf("jabcode: dispatch GPU directional scan: %w", err)
	}
	if !chained {
		return b.downloadDirectionalRecords(recorder, geom, dir)
	}
	return b.chainDirectionalSweep(recorder, width, height, geom, dir, channel)
}

// downloadDirectionalRecords is the sweep tail for a direction the device
// cannot chain: the BSI-era family, which has no directional chain kernel, and
// any pass before the chain finished compiling. The host classifies every hit
// itself, so the raw records have to come back, and their size is only known
// once the counter does. That second round trip is the cost of a host-side
// consumer and is the reason the chained tail does not have one.
func (b *gpuBinarizer) downloadDirectionalRecords(
	recorder *vulki.Recorder,
	geom finderRunsGeometry,
	dir scanDirection,
) (finderDirSweep, error) {
	var sweep finderDirSweep
	counts := make([]byte, finderWindowCounterCount*4)
	phaseprobe.Count("download.directional_counters", len(counts))
	if err := recorder.Download(b.dirCounters, 0, counts); err != nil {
		return sweep, fmt.Errorf("jabcode: record GPU directional scan counter download: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return sweep, fmt.Errorf("jabcode: run GPU directional scan: %w", err)
	}
	// counters[0] is the required count and is never clamped, so an overflow is
	// visible here rather than as a short list. A truncated sweep is worse than
	// no sweep: it looks like a direction that found little.
	required := int(binary.LittleEndian.Uint32(counts))
	if required == 0 || required > gpuFinderDirectionalCapacity {
		return sweep, nil
	}
	download, err := b.device.NewRecorder()
	if err != nil {
		return sweep, fmt.Errorf("jabcode: create GPU directional record downloader: %w", err)
	}
	defer download.Abort()
	raw := make([]byte, required*finderWindowRecordWords*4)
	phaseprobe.Count("download.directional_records", len(raw))
	if err := download.Download(b.dirRecords, 0, raw); err != nil {
		return sweep, fmt.Errorf("jabcode: record GPU directional record download: %w", err)
	}
	if err := download.SubmitAndWait(); err != nil {
		return sweep, fmt.Errorf("jabcode: download GPU directional scan records: %w", err)
	}
	sweep.hits = parseFinderDirectionalRecords(raw, geom, dir)
	return sweep, nil
}

// chainDirectionalSweep finishes the sweep recording with the chain, in the
// same submission as the scan that feeds it. The hit count stays on the device
// throughout: a one-lane kernel turns the scan's counter into the chain's
// workgroup counts and into the chain's own invocation bound, so one direction
// costs one submission rather than a stall to learn a number and a second
// submission to act on it.
func (b *gpuBinarizer) chainDirectionalSweep(
	recorder *vulki.Recorder,
	width, height int,
	geom finderRunsGeometry,
	dir scanDirection,
	channel int,
) (finderDirSweep, error) {
	var sweep finderDirSweep
	chainKernel, chainBindings := b.directionalChainFor(channel)
	if chainBindings == nil {
		return sweep, fmt.Errorf("jabcode: no GPU directional chain for seek channel %d", channel)
	}
	chainParams := directionalChainParams(
		width, height, gpuFinderDirectionalCapacity,
		b.directionalPrintLevels, b.colorSource != nil, geom, dir,
	)
	if err := recorder.Update(b.dirChainParams, 0, chainParams[:]); err != nil {
		return sweep, fmt.Errorf("jabcode: update GPU directional chain parameters: %w", err)
	}
	var summaryZero [gpuFinderDirectionalSummaryBytes]byte
	if err := recorder.Update(b.dirSummary, 0, summaryZero[:]); err != nil {
		return sweep, fmt.Errorf("jabcode: clear GPU directional summary: %w", err)
	}
	if err := recorder.Barrier(b.dirRecords, b.dirCounters); err != nil {
		return sweep, fmt.Errorf("jabcode: synchronize GPU directional records for the chain: %w", err)
	}
	argsKernel, err := b.kernels.finderDispatchArgs()
	if err != nil {
		return sweep, err
	}
	if err := recorder.Dispatch(argsKernel, b.dirArgsBindings, vulki.Workgroups{X: 1, Y: 1, Z: 1}); err != nil {
		return sweep, fmt.Errorf("jabcode: dispatch GPU directional chain arguments: %w", err)
	}
	if err := recorder.Barrier(b.dirArgs, b.dirSummary); err != nil {
		return sweep, fmt.Errorf("jabcode: synchronize GPU directional chain arguments: %w", err)
	}
	if err := recorder.DispatchIndirect(chainKernel, chainBindings, b.dirArgs, 0); err != nil {
		return sweep, fmt.Errorf("jabcode: dispatch GPU directional chain: %w", err)
	}
	if err := recorder.Barrier(b.dirChainOutcomes, b.dirSummary); err != nil {
		return sweep, fmt.Errorf("jabcode: synchronize GPU directional chain outcomes: %w", err)
	}
	// The compacted length is a device fact, so asking for it first and then for
	// the list would stall the pipeline for a number. Instead a prefix that
	// covers almost every direction rides along in this submission, and only a
	// direction that genuinely produced more comes back for its tail. That keeps
	// the common sweep at one submission without sizing every sweep's transfer
	// for the rare crowded one.
	summaryBytes := make([]byte, gpuFinderDirectionalSummaryBytes)
	compact := make([]byte, gpuFinderDirectionalPrefixBytes)
	phaseprobe.Count("download.directional_summary", len(summaryBytes))
	if err := recorder.Download(b.dirSummary, 0, summaryBytes); err != nil {
		return sweep, fmt.Errorf("jabcode: record GPU directional summary download: %w", err)
	}
	phaseprobe.Count("download.directional_outcomes", len(compact))
	if err := recorder.Download(b.dirChainOutcomes, 0, compact); err != nil {
		return sweep, fmt.Errorf("jabcode: record GPU directional chain download: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return sweep, fmt.Errorf("jabcode: run GPU directional sweep: %w", err)
	}
	compacted := int(binary.LittleEndian.Uint32(
		summaryBytes[gpuFinderDirectionalSummaryCompacted*4:]))
	if compacted > gpuFinderDirectionalPrefixCapacity &&
		compacted <= gpuFinderDirectionalCompactCapacity {
		full, err := b.downloadDirectionalTail(compacted)
		if err != nil {
			return sweep, err
		}
		compact = full
	}
	return parseDirectionalSweep(summaryBytes, compact), nil
}

// downloadDirectionalTail re-reads one crowded direction's compacted list at
// its real length. It is the rare path: the alternative is either paying the
// full bound on every direction or abandoning the device for the level, and the
// second was measured to cost a whole-frame download.
func (b *gpuBinarizer) downloadDirectionalTail(compacted int) ([]byte, error) {
	recorder, err := b.device.NewRecorder()
	if err != nil {
		return nil, fmt.Errorf("jabcode: create GPU directional tail downloader: %w", err)
	}
	defer recorder.Abort()
	full := make([]byte, compacted*gpuFinderChainOutcomeWords*4)
	phaseprobe.Count("download.directional_outcomes_tail", len(full))
	if err := recorder.Download(b.dirChainOutcomes, 0, full); err != nil {
		return nil, fmt.Errorf("jabcode: record GPU directional tail download: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return nil, fmt.Errorf("jabcode: download GPU directional chain tail: %w", err)
	}
	return full, nil
}

// parseDirectionalSweep restores one direction's compacted candidates and the
// counters behind them. Either overflow reports no hits, which sends that
// direction to the host walk rather than to a truncated candidate list: a scan
// that outgrew its record buffer never ran the chain at all, and a compaction
// that outgrew the outcome buffer kept only a prefix.
func parseDirectionalSweep(summaryBytes, compact []byte) finderDirSweep {
	var sweep finderDirSweep
	word := func(index int) int {
		return int(binary.LittleEndian.Uint32(summaryBytes[index*4:]))
	}
	required := word(gpuFinderDirectionalSummaryRequired)
	if required == 0 || required > gpuFinderDirectionalCapacity {
		return sweep
	}
	compacted := word(gpuFinderDirectionalSummaryCompacted)
	if compacted > len(compact)/(gpuFinderChainOutcomeWords*4) {
		return sweep
	}
	sweep.summarized = true
	sweep.summary = finderDirSummary{
		rawHits:       word(1),
		branchBlue:    word(2),
		branchRed:     word(3),
		redColor:      word(4),
		redClassified: word(5),
	}
	buckets := make([]uint32, gpuFinderDirectionalSummaryBuckets)
	for bucket := range buckets {
		buckets[bucket] = binary.LittleEndian.Uint32(
			summaryBytes[(gpuFinderDirectionalSummaryHeader+bucket)*4:])
	}
	sweep.summary.moduleBuckets = buckets
	sweep.hits = make([]finderDirHit, compacted)
	for index := range sweep.hits {
		outcome := parseFinderChainOutcome(compact[index*gpuFinderChainOutcomeWords*4:])
		sweep.hits[index] = finderDirHit{
			centre:     core.PointF{X: outcome.centerX, Y: outcome.centerY},
			module:     outcome.moduleSize,
			outcome:    outcome,
			chained:    true,
			summarized: true,
		}
	}
	return sweep
}

func (b *gpuBinarizer) ensureDirectionalBuffers(kernel *vulki.Kernel) error {
	if b.dirBindings != nil {
		return nil
	}
	records, err := b.device.NewBuffer(gpuFinderDirectionalRecordBytes)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU directional scan records: %w", err)
	}
	counters, err := b.device.NewBuffer(finderWindowCounterCount * 4)
	if err != nil {
		_ = records.Close()
		return fmt.Errorf("jabcode: allocate GPU directional scan counters: %w", err)
	}
	params, err := b.device.NewBuffer(finderScanParamsBytes)
	if err != nil {
		_ = records.Close()
		_ = counters.Close()
		return fmt.Errorf("jabcode: allocate GPU directional scan parameters: %w", err)
	}
	bindings, err := kernel.NewBindings(
		vulki.BindBuffer(0, b.packedMasks),
		vulki.BindBuffer(1, records),
		vulki.BindBuffer(2, params),
		vulki.BindBuffer(3, counters),
	)
	if err != nil {
		_ = records.Close()
		_ = counters.Close()
		_ = params.Close()
		return fmt.Errorf("jabcode: bind GPU directional scan: %w", err)
	}
	b.dirRecords, b.dirCounters, b.dirParams, b.dirBindings = records, counters, params, bindings
	if b.onRetainedAllocation != nil {
		b.onRetainedAllocation(uint64(gpuFinderDirectionalRecordBytes +
			finderWindowCounterCount*4 + finderScanParamsBytes))
	}
	return nil
}

// ensureDirectionalChainBuffers allocates the chain's device state on first
// use and binds every compiled family's chain over it. The families sweep one
// after another inside a locate and each sweep clears the summary it is about
// to write, so one set of buffers serves them all; only the binding sets and
// the kernels differ.
func (b *gpuBinarizer) ensureDirectionalChainBuffers() error {
	if b.dirChainBindings != nil {
		return nil
	}
	kernel, err := b.kernels.finderChainDirectional()
	if err != nil {
		return err
	}
	outcomes, err := b.device.NewBuffer(gpuFinderDirectionalOutcomeBytes)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU directional chain outcomes: %w", err)
	}
	params, err := b.device.NewBuffer(gpuFinderDirectionalChainParamsBytes)
	if err != nil {
		_ = outcomes.Close()
		return fmt.Errorf("jabcode: allocate GPU directional chain parameters: %w", err)
	}
	summary, err := b.device.NewBuffer(gpuFinderDirectionalSummaryBytes)
	if err != nil {
		_ = outcomes.Close()
		_ = params.Close()
		return fmt.Errorf("jabcode: allocate GPU directional summary: %w", err)
	}
	args, err := b.device.NewBuffer(gpuFinderDirectionalArgsBytes)
	if err != nil {
		_ = outcomes.Close()
		_ = params.Close()
		_ = summary.Close()
		return fmt.Errorf("jabcode: allocate GPU directional dispatch arguments: %w", err)
	}
	closeAll := func() {
		_ = outcomes.Close()
		_ = params.Close()
		_ = summary.Close()
		_ = args.Close()
	}
	// A binarizer without a balanced image binds the packed masks in the
	// colour slot: the kernel never reads it, because the parameter flag that
	// enables the colour stage stays clear, and Vulkan still needs every
	// declared binding filled.
	colorSource := b.colorSource
	if colorSource == nil {
		colorSource = b.packedMasks
	}
	bindings, err := kernel.NewBindings(
		vulki.BindBuffer(0, b.packedMasks),
		vulki.BindBuffer(1, b.dirRecords),
		vulki.BindBuffer(2, outcomes),
		vulki.BindBuffer(3, params),
		vulki.BindBuffer(4, colorSource),
		vulki.BindBuffer(5, summary),
		vulki.BindBuffer(6, args),
	)
	if err != nil {
		closeAll()
		return fmt.Errorf("jabcode: bind GPU directional chain: %w", err)
	}
	argsKernel, err := b.kernels.finderDispatchArgs()
	if err != nil {
		_ = bindings.Close()
		closeAll()
		return err
	}
	argsBindings, err := argsKernel.NewBindings(
		vulki.BindBuffer(0, b.dirCounters),
		vulki.BindBuffer(1, args),
		vulki.BindBuffer(2, summary),
		vulki.BindBuffer(3, params),
	)
	if err != nil {
		_ = bindings.Close()
		closeAll()
		return fmt.Errorf("jabcode: bind GPU directional chain arguments: %w", err)
	}
	bsiBindings, err := b.bindDirectionalBSIChain(
		outcomes, params, summary, args, colorSource,
	)
	if err != nil {
		_ = argsBindings.Close()
		_ = bindings.Close()
		closeAll()
		return err
	}
	b.dirChainOutcomes, b.dirChainParams, b.dirSummary, b.dirArgs = outcomes, params, summary, args
	b.dirChainBindings, b.dirArgsBindings = bindings, argsBindings
	b.dirChainBSIBindings = bsiBindings
	if b.onRetainedAllocation != nil {
		b.onRetainedAllocation(gpuFinderDirectionalOutcomeBytes +
			gpuFinderDirectionalChainParamsBytes + gpuFinderDirectionalSummaryBytes +
			gpuFinderDirectionalArgsBytes)
	}
	return nil
}

func (b *gpuBinarizer) closeDirectional() error {
	if b.dirBindings == nil {
		return nil
	}
	var closeErrors []error
	for _, set := range []**vulki.BindingSet{
		&b.dirArgsBindings, &b.dirChainBSIBindings, &b.dirChainBindings,
	} {
		if *set != nil {
			closeErrors = append(closeErrors, (*set).Close())
			*set = nil
		}
	}
	closeErrors = append(closeErrors, b.dirBindings.Close())
	b.dirBindings = nil
	for _, buf := range []**vulki.Buffer{
		&b.dirChainOutcomes, &b.dirChainParams, &b.dirSummary, &b.dirArgs,
		&b.dirRecords, &b.dirCounters, &b.dirParams,
	} {
		if *buf != nil {
			closeErrors = append(closeErrors, (*buf).Close())
			*buf = nil
		}
	}
	return errors.Join(closeErrors...)
}

// directionalSweepGeometry derives the line set from the frame corners exactly
// as sweepDirection does, so the device covers the same lines the CPU walk
// would have.
func directionalSweepGeometry(width, height int, dir scanDirection, step int) finderRunsGeometry {
	perp := dir.perpendicular()
	nx, ny := perp.dx/perp.pxPerSample, perp.dy/perp.pxPerSample
	qLo, qHi := math.Inf(1), math.Inf(-1)
	for _, c := range [4][2]float64{
		{0, 0}, {float64(width - 1), 0}, {0, float64(height - 1)}, {float64(width - 1), float64(height - 1)},
	} {
		q := c[0]*nx + c[1]*ny
		qLo, qHi = math.Min(qLo, q), math.Max(qHi, q)
	}
	return finderRunsGeometry{
		dx: float32(dir.dx), dy: float32(dir.dy),
		nx: float32(nx), ny: float32(ny),
		qLo: float32(qLo), qStep: float32(step),
		lineCount: int((qHi-qLo)/float64(step)) + 1,
		// A straight walk cannot exceed the frame's own extent in either axis,
		// and the kernel clips per line, so this bounds the span rather than
		// setting it.
		lineLength: width + height,
	}
}

func directionalScanParams(width, height int, channelMask uint32, geom finderRunsGeometry) []byte {
	params := make([]byte, finderScanParamsBytes)
	put := func(word int, v uint32) { binary.LittleEndian.PutUint32(params[word*4:], v) }
	put(0, uint32(width))
	put(1, uint32(height))
	put(2, channelMask)
	put(3, uint32(geom.lineLength))
	put(4, math.Float32bits(geom.dx))
	put(5, math.Float32bits(geom.dy))
	put(6, math.Float32bits(geom.nx))
	put(7, math.Float32bits(geom.ny))
	put(8, math.Float32bits(geom.qLo))
	put(9, math.Float32bits(geom.qStep))
	put(10, uint32(geom.lineCount))
	// run_capacity and plane_words belong to the boundary prototypes and the
	// bitplane layout; neither applies here.
	put(13, finderScanSkipCrossCheck)
	return params
}

func directionalChainParams(
	width, height, count int,
	printLevels bool,
	colorSource bool,
	geom finderRunsGeometry,
	base scanDirection,
) [gpuFinderDirectionalChainParamsBytes]byte {
	var params [gpuFinderDirectionalChainParamsBytes]byte
	common := gpuFinderChainParams(width, height, count, printLevels)
	copy(params[:], common[:])
	if colorSource {
		flags := binary.LittleEndian.Uint32(params[12:]) | gpuFinderChainFlagColorSource
		binary.LittleEndian.PutUint32(params[12:], flags)
	}
	// The common block's last word is the compaction bound in this layout.
	binary.LittleEndian.PutUint32(params[28:], gpuFinderDirectionalCompactCapacity)
	put := func(offset int, value float64) {
		binary.LittleEndian.PutUint32(params[offset:], math.Float32bits(float32(value)))
	}
	for index, value := range []float64{
		float64(geom.dx), float64(geom.dy),
		float64(geom.nx), float64(geom.ny),
		float64(geom.qLo), float64(geom.qStep),
	} {
		put(gpuFinderChainParamsSize+index*4, value)
	}
	offset := gpuFinderChainParamsSize + 24
	for _, direction := range []scanDirection{
		base,
		base.perpendicular(),
		base.turn(45),
		base.turn(-45),
	} {
		put(offset, direction.dx)
		put(offset+4, direction.dy)
		put(offset+8, direction.pxPerSample)
		offset += 12
	}
	return params
}

// parseFinderDirectionalRecords resolves each record to the pair the host chain
// consumes. This is the survivor contract's own arithmetic: the centre is the
// midpoint of the middle run projected back through the sweep basis, and the
// module size is the mean inner run in pixels.
func parseFinderDirectionalRecords(raw []byte, geom finderRunsGeometry, dir scanDirection) []finderDirHit {
	count := len(raw) / (finderWindowRecordWords * 4)
	hits := make([]finderDirHit, 0, count)
	for i := range count {
		at := i * finderWindowRecordWords * 4
		key := binary.LittleEndian.Uint32(raw[at:]) & (1<<finderEvidenceBits - 1)
		b := func(k int) float64 {
			return float64(binary.LittleEndian.Uint32(raw[at+(k+1)*4:]))
		}
		q := float64(geom.qLo) + float64(key/3)*float64(geom.qStep)
		along := (b(2) + b(3)) / 2
		hits = append(hits, finderDirHit{
			centre: core.PointF{
				X: q*float64(geom.nx) + along*float64(geom.dx),
				Y: q*float64(geom.ny) + along*float64(geom.dy),
			},
			// (b[4] - b[1]) / 3 is the mean of the three inner runs in samples;
			// pxPerSample converts to pixels, which is the unit every module
			// comparison downstream is in.
			module: (b(4) - b(1)) / 3 * dir.pxPerSample,
		})
	}
	return hits
}

func parseFinderDirectionalOutcomes(hits []finderDirHit, raw []byte) {
	for index := range hits {
		offset := index * gpuFinderChainOutcomeWords * 4
		hits[index].outcome = parseFinderChainOutcome(raw[offset:])
		hits[index].chained = true
	}
}
