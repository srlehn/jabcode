//go:build !js

package detect

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
)

// The device form of sweepDirection's seek stage. It replaces seekPatternAlong
// and nothing else: the per-hit chain, which confirms on the two channels this
// scan does not read, stays on the host.
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

// scanDirectionHits sweeps one direction over the still resident packed masks
// and returns the raw signatures found on channel. A nil result is not an
// error: it means this pass has no directional kernel and the caller should
// sweep on the CPU.
func (b *gpuBinarizer) scanDirectionHits(
	width, height int,
	dir scanDirection,
	step, channel int,
) ([]finderDirHit, error) {
	if b == nil || b.device == nil || b.packedMasks == nil {
		return nil, nil
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
	download, err := b.device.NewRecorder()
	if err != nil {
		return nil, fmt.Errorf("jabcode: create GPU directional record downloader: %w", err)
	}
	defer download.Abort()
	if err := download.Download(b.dirRecords, 0, raw); err != nil {
		return nil, fmt.Errorf("jabcode: record GPU directional record download: %w", err)
	}
	if err := download.SubmitAndWait(); err != nil {
		return nil, fmt.Errorf("jabcode: download GPU directional scan records: %w", err)
	}
	return parseFinderDirectionalRecords(raw, geom, dir), nil
}

func (b *gpuBinarizer) ensureDirectionalBuffers(kernel *vulki.Kernel) error {
	if b.dirBindings != nil {
		return nil
	}
	records, err := b.device.NewBuffer(gpuFinderDirectionalCapacity * finderWindowRecordWords * 4)
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
	return nil
}

func (b *gpuBinarizer) closeDirectional() error {
	if b.dirBindings == nil {
		return nil
	}
	err := b.dirBindings.Close()
	b.dirBindings = nil
	for _, buf := range []**vulki.Buffer{&b.dirRecords, &b.dirCounters, &b.dirParams} {
		if *buf != nil {
			_ = (*buf).Close()
			*buf = nil
		}
	}
	return err
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
