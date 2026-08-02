//go:build !js

package detect

import (
	"encoding/binary"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"testing"

	"github.com/srlehn/jabcode/internal/core"

	"github.com/srlehn/vulki"
)

// benchScanImageEnv names a capture to sweep. Captures stay outside the
// repository, so the path is supplied rather than referenced; without it the
// benchmark builds a frame of the same size and comparable transition density
// so it still runs from a clean checkout.
const benchScanImageEnv = "JABCODE_SCAN_IMAGE"

// benchScanAngleEnv overrides the swept angle in degrees. The default is 45,
// where a directional sweep does the most work per line and where the CPU
// walk's cost was measured.
const benchScanAngleEnv = "JABCODE_SCAN_ANGLE"

// scanPrimitiveCase is one prototype paired with one storage layout. The
// prototypes answer different questions - two produce a boundary list for a
// later pass, the third produces surviving windows directly - so what is
// compared is the cost of reaching each one's own output, not a shared
// intermediate.
type scanPrimitiveCase struct {
	name   string
	layout finderScanLayout
	fused  bool
	kernel func(*gpuDecodeKernels, finderScanLayout) (*vulki.Kernel, error)
}

// scanPrimitiveCases lists the cases this device can run. A portable-only
// adapter still benchmarks its two available designs rather than aborting: the
// comparison is between what is actually selectable there.
func scanPrimitiveCases(b *testing.B, kernels *gpuDecodeKernels) []scanPrimitiveCase {
	b.Helper()
	subgroups, err := kernels.subgroupKernelsUsable()
	if err != nil {
		b.Fatalf("device advertises ballot support but the ballot kernel did not build: %v", err)
	}
	var out []scanPrimitiveCase
	for _, layout := range []finderScanLayout{finderScanInterleaved, finderScanBitplane} {
		out = append(out,
			scanPrimitiveCase{"hillis/" + layout.name(), layout, false, (*gpuDecodeKernels).finderRunsHillis},
			scanPrimitiveCase{"fused-scan/" + layout.name(), layout, true, (*gpuDecodeKernels).finderWindowsScan},
		)
		if subgroups {
			out = append(out,
				scanPrimitiveCase{"subgroup/" + layout.name(), layout, false, (*gpuDecodeKernels).finderRunsSubgroup},
				scanPrimitiveCase{"fused-ballot/" + layout.name(), layout, true, (*gpuDecodeKernels).finderWindowsBallot},
			)
		}
	}
	return out
}

// scanPrimitiveResult is what the comparison is decided on. Device time and
// allocated bytes are the two that pick the data contract: a primitive that
// wins on time but needs an output buffer that cannot be sized for a real frame
// has not won anything.
type scanPrimitiveResult struct {
	deviceSeconds float64
	outputBytes   uint64
	maskBytes     uint64
	emitted       uint64
	strict        uint64
	overflowed    int
}

// The primitive comparison the directional device route waits on. Three
// designs, each against both mask layouts, over a real capture at its real
// sweep geometry.
//
// It is a benchmark rather than a test because the answer is a measurement, and
// it deliberately reports allocated bytes alongside time: the boundary
// prototypes need a slot per sample per line per channel, which at a 12 MP
// frame is hundreds of megabytes for one angle, and no amount of speed makes
// that a shape a second stage may inherit.
func BenchmarkGPUScanPrimitives(b *testing.B) {
	device, err := vulki.Open()
	if err != nil {
		b.Skipf("Vulkan unavailable: %v", err)
	}
	kernels := newGPUDecodeKernels(device)
	b.Cleanup(func() {
		_ = kernels.Close()
		_ = device.Close()
	})
	b.Logf("Vulkan adapter: %s", device.Info().AdapterName)

	masks, width, height := benchScanMasks(b)
	angle := 45.0
	if raw := os.Getenv(benchScanAngleEnv); raw != "" {
		if _, err := fmt.Sscanf(raw, "%f", &angle); err != nil {
			b.Fatalf("parse %s: %v", benchScanAngleEnv, err)
		}
	}
	// The sweep spacing the detector itself uses, so the line count is the one
	// a real read pays for rather than a chosen number.
	step := max(height/(2*maxSymbolRows*maxModules), 1)
	geom := sweepGeometry(width, height, angle, float64(step))
	transitions := benchScanTransitions(masks, width, height, geom)
	b.Logf("frame %dx%d, %.0f degrees, step %d: %d lines of up to %d samples, %d transitions on channel 1",
		width, height, angle, step, geom.lineCount, geom.lineLength, transitions)

	// Worst case is one boundary per sample, which is what a per-line slot
	// layout has to reserve. Reporting it next to the capacity actually used is
	// the point: the gap is the reason the layout cannot ship.
	worstCase := uint64(geom.lineCount) * 3 * uint64(geom.lineLength) * 4
	b.Logf("per-line slot layout at full capacity would need %.1f MB for this one angle", float64(worstCase)/(1<<20))

	// Enough for the densest line seen, so the boundary prototypes are measured
	// writing complete lists rather than dropping writes past a short capacity.
	capacity := geom.lineLength + 8

	for _, tc := range scanPrimitiveCases(b, kernels) {
		b.Run(tc.name, func(b *testing.B) {
			packed, planeWords := packBenchScanMasks(tc.layout, masks, width, height)
			result := runScanPrimitive(b, device, kernels, tc, width, height, capacity, packed, planeWords, geom)
			b.ReportMetric(result.deviceSeconds*1000, "device-ms")
			b.ReportMetric(float64(result.outputBytes)/(1<<20), "output-MB")
			b.ReportMetric(float64(result.maskBytes)/(1<<20), "mask-MB")
			b.ReportMetric(float64(result.emitted), "emitted")
			if tc.fused {
				b.ReportMetric(float64(result.strict), "strict")
			} else if result.overflowed > 0 {
				b.ReportMetric(float64(result.overflowed), "overflowed-lines")
			}
		})
	}
}

// The endpoint every device prototype has to beat: the CPU directional sweep
// itself, over the same masks at the same geometry. Without it the device table
// only ranks device designs against each other, which says nothing about
// whether any of them is worth wiring.
//
// This is sweepDirection's loop with the per-hit chain left out, so it is the
// raw seek cost - the profile's single largest line in a rotated read - and not
// the whole detector. That makes it a floor for what the CPU route costs, which
// is the conservative direction for a comparison the device is trying to win.
func BenchmarkCPUDirectionalSweep(b *testing.B) {
	masks, width, height := benchScanMasks(b)
	angle := 45.0
	if raw := os.Getenv(benchScanAngleEnv); raw != "" {
		if _, err := fmt.Sscanf(raw, "%f", &angle); err != nil {
			b.Fatalf("parse %s: %v", benchScanAngleEnv, err)
		}
	}
	step := max(height/(2*maxSymbolRows*maxModules), 1)
	dir := newScanDirection(angle)
	perp := dir.perpendicular()
	nx, ny := perp.dx/perp.pxPerSample, perp.dy/perp.pxPerSample
	qLo, qHi := math.Inf(1), math.Inf(-1)
	for _, c := range [4][2]float64{
		{0, 0}, {float64(width - 1), 0}, {0, float64(height - 1)}, {float64(width - 1), float64(height - 1)},
	} {
		q := c[0]*nx + c[1]*ny
		qLo, qHi = math.Min(qLo, q), math.Max(qHi, q)
	}
	seek := masks[1]

	var hits int
	for b.Loop() {
		hits = 0
		for q := qLo; q <= qHi; q += float64(step) {
			p0 := core.PointF{X: q * nx, Y: q * ny}
			start, count, ok := clipScanLine(width, height, p0, dir)
			if !ok {
				continue
			}
			for count > 0 {
				_, _, next, hit := seekPatternAlong(seek, dir, p0.X, p0.Y, start, count)
				if !hit {
					break
				}
				count -= next - start
				start = next
				hits++
			}
		}
	}
	b.ReportMetric(float64(hits), "emitted")
}

// The mask-production cost neither the CPU sweep nor the device dispatch
// includes, measured so it is not left as an unquantified caveat on the
// comparison. The device kernels read packed or bitplane masks; the CPU sweep
// reads the binarizer's byte-per-pixel bitmaps directly. Today the packing is
// host work done outside the timed loop, so a route-level figure has to account
// for it. The intended fix is for the resident binarizer to write the layout
// the kernels want, at which point this cost disappears rather than moving.
func BenchmarkScanMaskPacking(b *testing.B) {
	masks, width, height := benchScanMasks(b)
	for _, layout := range []finderScanLayout{finderScanInterleaved, finderScanBitplane} {
		b.Run(layout.name(), func(b *testing.B) {
			for b.Loop() {
				packBenchScanMasks(layout, masks, width, height)
			}
		})
	}
}

// runScanPrimitive times one case. Buffers and bindings are built once and the
// timed loop submits only the dispatch, because allocating a 100 MB output
// buffer per iteration would measure the allocator.
func runScanPrimitive(
	b *testing.B,
	device *vulki.Device,
	kernels *gpuDecodeKernels,
	tc scanPrimitiveCase,
	width, height, capacity int,
	packed []byte,
	planeWords int,
	geom finderRunsGeometry,
) scanPrimitiveResult {
	b.Helper()
	kernel, err := tc.kernel(kernels, tc.layout)
	if err != nil {
		b.Fatalf("compile %s: %v", tc.name, err)
	}
	var result scanPrimitiveResult
	result.maskBytes = uint64(len(packed))

	// The fused prototype's output scales with accepted windows, not with the
	// frame, so it is given a generous cap and reports the true count; the
	// boundary prototypes need the full per-line slot layout.
	outputBytes := uint64(geom.lineCount) * 3 * uint64(capacity) * 4
	countBytes := uint64(geom.lineCount) * 3 * 4
	if tc.fused {
		outputBytes = uint64(1<<20) * finderWindowRecord * 4
		countBytes = 8
	}
	result.outputBytes = outputBytes

	masks, err := device.NewBuffer(uint64(len(packed)))
	if err != nil {
		b.Fatalf("allocate masks: %v", err)
	}
	defer func() { _ = masks.Close() }()
	out, err := device.NewBuffer(outputBytes)
	if err != nil {
		b.Fatalf("allocate output (%.1f MB): %v", float64(outputBytes)/(1<<20), err)
	}
	defer func() { _ = out.Close() }()
	counts, err := device.NewBuffer(countBytes)
	if err != nil {
		b.Fatalf("allocate counts: %v", err)
	}
	defer func() { _ = counts.Close() }()
	paramBuf, err := device.NewBuffer(finderScanParamsBytes)
	if err != nil {
		b.Fatalf("allocate params: %v", err)
	}
	defer func() { _ = paramBuf.Close() }()

	bindings, err := kernel.NewBindings(
		vulki.BindBuffer(0, masks),
		vulki.BindBuffer(1, out),
		vulki.BindBuffer(2, paramBuf),
		vulki.BindBuffer(3, counts),
	)
	if err != nil {
		b.Fatalf("bind %s: %v", tc.name, err)
	}
	defer func() { _ = bindings.Close() }()

	upload, err := device.NewRecorder()
	if err != nil {
		b.Fatalf("new recorder: %v", err)
	}
	defer upload.Abort()
	if err := upload.Upload(masks, 0, packed); err != nil {
		b.Fatalf("upload masks: %v", err)
	}
	if err := upload.Update(paramBuf, 0, finderScanParams(width, height, 1<<1, capacity, planeWords, geom)); err != nil {
		b.Fatalf("upload params: %v", err)
	}
	if err := upload.SubmitAndWait(); err != nil {
		b.Fatalf("prepare %s: %v", tc.name, err)
	}

	groups := vulki.Workgroups{X: uint32(geom.lineCount), Y: 3, Z: 1}
	dispatch := func(timed bool) float64 {
		recorder, err := device.NewRecorder()
		if err != nil {
			b.Fatalf("new recorder: %v", err)
		}
		defer recorder.Abort()
		// Counters accumulate, so the fused prototype needs them cleared per
		// pass or its reported total multiplies by the iteration count. A
		// device-side fill keeps that off the staging path, which matters here
		// because the clear is inside the timed loop.
		if err := recorder.Fill(counts, 0, countBytes, 0); err != nil {
			b.Fatalf("clear counts: %v", err)
		}
		if err := recorder.Barrier(counts); err != nil {
			b.Fatalf("barrier: %v", err)
		}
		if timed {
			if err := recorder.TimestampBegin("scan"); err != nil {
				b.Fatalf("timestamp begin: %v", err)
			}
		}
		if err := recorder.Dispatch(kernel, bindings, groups); err != nil {
			b.Fatalf("dispatch %s: %v", tc.name, err)
		}
		if timed {
			if err := recorder.TimestampEnd("scan"); err != nil {
				b.Fatalf("timestamp end: %v", err)
			}
		}
		if err := recorder.SubmitAndWait(); err != nil {
			b.Fatalf("run %s: %v", tc.name, err)
		}
		if !timed {
			return 0
		}
		spans, err := recorder.Timestamps()
		if err != nil || len(spans) == 0 {
			return 0
		}
		return spans[0].Duration.Seconds()
	}

	// One untimed pass so the pipeline is resident and the first iteration is
	// not paying for it.
	dispatch(false)

	var deviceTotal float64
	for b.Loop() {
		deviceTotal += dispatch(true)
	}
	if n := b.N; n > 0 {
		result.deviceSeconds = deviceTotal / float64(n)
	}

	raw := make([]byte, countBytes)
	if err := counts.Download(raw); err != nil {
		b.Fatalf("download counts: %v", err)
	}
	if tc.fused {
		result.emitted = uint64(binary.LittleEndian.Uint32(raw[0:]))
		result.strict = uint64(binary.LittleEndian.Uint32(raw[4:]))
		return result
	}
	for i := 0; i+4 <= len(raw); i += 4 {
		n := binary.LittleEndian.Uint32(raw[i:])
		result.emitted += uint64(n)
		if int(n) > capacity {
			result.overflowed++
		}
	}
	return result
}

// benchScanMasks returns the three binary channel masks of the frame under
// test, produced by the same binarizer a read uses so the transition density is
// the real one.
func benchScanMasks(b *testing.B) (masks [3]*core.Bitmap, width, height int) {
	b.Helper()
	img := benchScanImage(b)
	bounds := img.Bounds()
	bm := core.BitmapFromImage(img)
	if bm == nil {
		b.Fatal("convert frame to bitmap")
	}
	return BinarizerRGB(bm, nil), bounds.Dx(), bounds.Dy()
}

func benchScanImage(b *testing.B) image.Image {
	b.Helper()
	path := os.Getenv(benchScanImageEnv)
	if path == "" {
		return syntheticScanFrame(3024, 4032)
	}
	f, err := os.Open(path)
	if err != nil {
		b.Fatalf("open %s: %v", benchScanImageEnv, err)
	}
	defer func() { _ = f.Close() }()
	img, _, err := image.Decode(f)
	if err != nil {
		b.Fatalf("decode %s: %v", path, err)
	}
	return img
}

// syntheticScanFrame stands in for a capture when none is supplied: a
// module-sized block pattern over part of the frame and a coarse gradient
// elsewhere, so the sweep meets both dense and sparse lines. It is a fallback,
// not a model of a photograph, and results from it should be labelled as such.
func syntheticScanFrame(width, height int) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	symbol := image.Rect(width/4, height/4, 3*width/4, 3*height/4)
	for y := range height {
		for x := range width {
			var r, g, bl uint8 = 200, 200, 200
			if image.Pt(x, y).In(symbol) {
				cell := (x/8 + y/8) % 2
				if cell == 0 {
					r, g, bl = 20, 20, 20
				}
				if (x/8+y/16)%3 == 0 {
					g = 220
				}
			} else if (x/64+y/64)%2 == 0 {
				r, g, bl = 160, 170, 180
			}
			at := img.PixOffset(x, y)
			img.Pix[at], img.Pix[at+1], img.Pix[at+2], img.Pix[at+3] = r, g, bl, 255
		}
	}
	return img
}

func packBenchScanMasks(layout finderScanLayout, masks [3]*core.Bitmap, width, height int) ([]byte, int) {
	return packFinderRunsMasks(layout, width, height, func(x, y, channel int) bool {
		m := masks[channel]
		if m == nil {
			return false
		}
		return m.Pix[y*m.Width+x] > 0
	})
}

// benchScanTransitions counts the run boundaries the sweep will find on the
// seek channel, walking the same sample positions the kernels do. It is the
// number that actually drives every prototype's cost, so it is reported rather
// than inferred from the frame size.
func benchScanTransitions(masks [3]*core.Bitmap, width, height int, geom finderRunsGeometry) int {
	m := masks[1]
	if m == nil {
		return 0
	}
	total := 0
	for line := range geom.lineCount {
		q := geom.qLo + float32(line)*geom.qStep
		ox, oy := q*geom.nx, q*geom.ny
		prev, started := byte(0), false
		for i := range geom.lineLength {
			x := int(math.Floor(float64(ox + float32(i)*geom.dx)))
			y := int(math.Floor(float64(oy + float32(i)*geom.dy)))
			if x < 0 || x >= width || y < 0 || y >= height {
				continue
			}
			v := m.Pix[y*m.Width+x]
			if !started {
				prev, started = v, true
				total++
				continue
			}
			if v != prev {
				total++
				prev = v
			}
		}
		if started {
			total++
		}
	}
	return total
}
