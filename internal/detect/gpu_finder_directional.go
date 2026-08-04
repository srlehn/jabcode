//go:build !js

package detect

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
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

const (
	gpuFinderDirectionalRecordBytes  = gpuFinderDirectionalCapacity * finderWindowRecordWords * 4
	gpuFinderDirectionalOutcomeBytes = gpuFinderDirectionalCapacity * gpuFinderChainOutcomeWords * 4
	// The common block, six sweep-basis scalars and four directions of three
	// scalars each, all native f32.
	gpuFinderDirectionalChainParamsBytes = gpuFinderChainParamsSize + 6*4 + 4*3*4
	gpuFinderDirectionalRetainedBytes    = gpuFinderDirectionalRecordBytes +
		finderWindowCounterCount*4 + finderScanParamsBytes +
		gpuFinderDirectionalOutcomeBytes + gpuFinderDirectionalChainParamsBytes
)

// scanDirectionHits sweeps one direction over the still resident packed masks
// and returns its raw signatures plus device-chain outcomes when available. A
// nil result is not an error: it means this pass has no directional kernel and
// the caller should sweep on the CPU.
func (b *gpuBinarizer) scanDirectionHits(
	width, height int,
	dir scanDirection,
	step, channel int,
) ([]finderDirHit, error) {
	if b == nil || b.device == nil || b.packedMasks == nil {
		return nil, nil
	}
	if channel == currentFamilySeekChannel {
		if err := b.kernels.directionalFinderChainError(); err != nil {
			return nil, err
		}
	}
	kernel, err := b.kernels.finderWindows(finderScanInterleaved)
	if err != nil {
		return nil, err
	}
	if err := b.ensureDirectionalBuffers(kernel); err != nil {
		return nil, err
	}
	geom := directionalSweepGeometry(width, height, dir, step)
	if geom.lineCount <= 0 {
		return nil, nil
	}

	recorder, err := b.device.NewRecorder()
	if err != nil {
		return nil, fmt.Errorf("jabcode: create GPU directional scan recorder: %w", err)
	}
	defer recorder.Abort()
	params := directionalScanParams(width, height, uint32(1)<<uint(channel), geom)
	if err := recorder.Update(b.dirParams, 0, params); err != nil {
		return nil, fmt.Errorf("jabcode: update GPU directional scan parameters: %w", err)
	}
	var zero [finderWindowCounterCount * 4]byte
	if err := recorder.Update(b.dirCounters, 0, zero[:]); err != nil {
		return nil, fmt.Errorf("jabcode: clear GPU directional scan counters: %w", err)
	}
	if err := recorder.Barrier(b.packedMasks); err != nil {
		return nil, fmt.Errorf("jabcode: synchronize GPU packed masks for the directional scan: %w", err)
	}
	if err := recorder.Dispatch(kernel, b.dirBindings, vulki.Workgroups{
		X: uint32(geom.lineCount), Y: 3, Z: 1,
	}); err != nil {
		return nil, fmt.Errorf("jabcode: dispatch GPU directional scan: %w", err)
	}
	counts := make([]byte, finderWindowCounterCount*4)
	// Keep the fixed-size counter readback in the dispatch submission. A
	// standalone Buffer.Download creates another transient command pool and
	// fence for sixteen bytes on every direction.
	if err := recorder.Download(b.dirCounters, 0, counts); err != nil {
		return nil, fmt.Errorf("jabcode: record GPU directional scan counter download: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return nil, fmt.Errorf("jabcode: run GPU directional scan: %w", err)
	}
	// counters[0] is the required count and is never clamped, so an overflow is
	// visible here rather than as a short list. A truncated sweep is worse than
	// no sweep: it looks like a direction that found little.
	required := int(binary.LittleEndian.Uint32(counts))
	if required > gpuFinderDirectionalCapacity {
		return nil, nil
	}
	if required == 0 {
		return nil, nil
	}
	raw := make([]byte, required*finderWindowRecordWords*4)
	chained := channel == currentFamilySeekChannel &&
		!b.scanOnly && b.kernels.directionalFinderChainReady()
	var chainRaw []byte
	if chained {
		if err := b.ensureDirectionalChainBuffers(); err != nil {
			return nil, err
		}
		chainRaw = make([]byte, required*gpuFinderChainOutcomeWords*4)
	}
	download, err := b.device.NewRecorder()
	if err != nil {
		return nil, fmt.Errorf("jabcode: create GPU directional record downloader: %w", err)
	}
	defer download.Abort()
	if chained {
		params := directionalChainParams(
			width, height, required, b.directionalPrintLevels, geom, dir,
		)
		if err := download.Update(b.dirChainParams, 0, params[:]); err != nil {
			return nil, fmt.Errorf("jabcode: update GPU directional chain parameters: %w", err)
		}
		if err := download.Barrier(b.dirRecords); err != nil {
			return nil, fmt.Errorf("jabcode: synchronize GPU directional records for the chain: %w", err)
		}
		kernel, err := b.kernels.finderChainDirectional()
		if err != nil {
			return nil, err
		}
		groups := vulki.Workgroups{
			X: uint32((required + gpuFinderChainWorkgroupSize - 1) / gpuFinderChainWorkgroupSize),
			Y: 1,
			Z: 1,
		}
		if err := download.Dispatch(kernel, b.dirChainBindings, groups); err != nil {
			return nil, fmt.Errorf("jabcode: dispatch GPU directional chain: %w", err)
		}
		if err := download.Barrier(b.dirChainOutcomes); err != nil {
			return nil, fmt.Errorf("jabcode: synchronize GPU directional chain outcomes: %w", err)
		}
	}
	if err := download.Download(b.dirRecords, 0, raw); err != nil {
		return nil, fmt.Errorf("jabcode: record GPU directional record download: %w", err)
	}
	if chained {
		if err := download.Download(b.dirChainOutcomes, 0, chainRaw); err != nil {
			return nil, fmt.Errorf("jabcode: record GPU directional chain download: %w", err)
		}
	}
	if err := download.SubmitAndWait(); err != nil {
		return nil, fmt.Errorf("jabcode: download GPU directional scan records: %w", err)
	}
	hits := parseFinderDirectionalRecords(raw, geom, dir)
	if chained {
		parseFinderDirectionalOutcomes(hits, chainRaw)
	}
	return hits, nil
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
	bindings, err := kernel.NewBindings(
		vulki.BindBuffer(0, b.packedMasks),
		vulki.BindBuffer(1, b.dirRecords),
		vulki.BindBuffer(2, outcomes),
		vulki.BindBuffer(3, params),
	)
	if err != nil {
		_ = outcomes.Close()
		_ = params.Close()
		return fmt.Errorf("jabcode: bind GPU directional chain: %w", err)
	}
	b.dirChainOutcomes, b.dirChainParams, b.dirChainBindings = outcomes, params, bindings
	if b.onRetainedAllocation != nil {
		b.onRetainedAllocation(gpuFinderDirectionalOutcomeBytes + gpuFinderDirectionalChainParamsBytes)
	}
	return nil
}

func (b *gpuBinarizer) closeDirectional() error {
	if b.dirBindings == nil {
		return nil
	}
	var closeErrors []error
	if b.dirChainBindings != nil {
		closeErrors = append(closeErrors, b.dirChainBindings.Close())
		b.dirChainBindings = nil
	}
	closeErrors = append(closeErrors, b.dirBindings.Close())
	b.dirBindings = nil
	for _, buf := range []**vulki.Buffer{
		&b.dirChainOutcomes, &b.dirChainParams,
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
	geom finderRunsGeometry,
	base scanDirection,
) [gpuFinderDirectionalChainParamsBytes]byte {
	var params [gpuFinderDirectionalChainParamsBytes]byte
	common := gpuFinderChainParams(width, height, count, printLevels)
	copy(params[:], common[:])
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
