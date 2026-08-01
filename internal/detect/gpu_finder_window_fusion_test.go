//go:build !js

package detect

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	"github.com/srlehn/vulki"
)

const finderWindowRecord = 8

// finderWindow is one accepted five-run signature: the line and channel it was
// found on, and the six boundaries that define it.
type finderWindow struct {
	key      uint32
	boundary [6]uint32
}

// runFinderWindows dispatches the fused prototype and returns the survivors it
// wrote, the true accepted count and the subset whose inner runs are at least
// three samples.
// finderWindowVariant is one fused kernel: the ballot form where the device
// supports it, and the portable workgroup-scan form everywhere. Both are
// checked, because the fallback is only a fallback if it is known to agree.
type finderWindowVariant struct {
	name     string
	layout   finderScanLayout
	subgroup bool
	kernel   func(*gpuDecodeKernels, finderScanLayout) (*vulki.Kernel, error)
}

func finderWindowVariants() []finderWindowVariant {
	var out []finderWindowVariant
	for _, layout := range []finderScanLayout{finderScanInterleaved, finderScanBitplane} {
		out = append(out,
			finderWindowVariant{"ballot " + layout.name(), layout, true, (*gpuDecodeKernels).finderWindowsBallot},
			finderWindowVariant{"scan " + layout.name(), layout, false, (*gpuDecodeKernels).finderWindowsScan},
		)
	}
	return out
}

func runFinderWindows(
	t *testing.T,
	device *vulki.Device,
	kernels *gpuDecodeKernels,
	variant finderWindowVariant,
	width, height int,
	channelMask uint32,
	capacity int,
	packed []byte,
	planeWords int,
	geom finderRunsGeometry,
) (found []finderWindow, accepted, strict uint32) {
	t.Helper()
	kernel, err := variant.kernel(kernels, variant.layout)
	if err != nil {
		t.Fatalf("compile fused windows %s: %v", variant.name, err)
	}
	masks, err := device.NewBuffer(uint64(len(packed)))
	if err != nil {
		t.Fatalf("allocate masks: %v", err)
	}
	defer func() { _ = masks.Close() }()
	out, err := device.NewBuffer(uint64(capacity * finderWindowRecord * 4))
	if err != nil {
		t.Fatalf("allocate survivors: %v", err)
	}
	defer func() { _ = out.Close() }()
	countBuf, err := device.NewBuffer(8)
	if err != nil {
		t.Fatalf("allocate counters: %v", err)
	}
	defer func() { _ = countBuf.Close() }()
	paramBuf, err := device.NewBuffer(uint64(finderRunsParamsWords * 4))
	if err != nil {
		t.Fatalf("allocate params: %v", err)
	}
	defer func() { _ = paramBuf.Close() }()

	bindings, err := kernel.NewBindings(
		vulki.BindBuffer(0, masks),
		vulki.BindBuffer(1, out),
		vulki.BindBuffer(2, paramBuf),
		vulki.BindBuffer(3, countBuf),
	)
	if err != nil {
		t.Fatalf("bind fused windows: %v", err)
	}
	defer func() { _ = bindings.Close() }()

	recorder, err := device.NewRecorder()
	if err != nil {
		t.Fatalf("new recorder: %v", err)
	}
	defer recorder.Abort()
	if err := recorder.Upload(masks, 0, packed); err != nil {
		t.Fatalf("upload masks: %v", err)
	}
	if err := recorder.Update(paramBuf, 0, finderScanParams(width, height, channelMask, capacity, planeWords, geom)); err != nil {
		t.Fatalf("upload params: %v", err)
	}
	if err := recorder.Update(countBuf, 0, make([]byte, 8)); err != nil {
		t.Fatalf("clear counters: %v", err)
	}
	if err := recorder.Dispatch(kernel, bindings, vulki.Workgroups{X: uint32(geom.lineCount), Y: 3, Z: 1}); err != nil {
		t.Fatalf("dispatch fused windows: %v", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		t.Fatalf("run fused windows: %v", err)
	}

	rawCounts := make([]byte, 8)
	if err := countBuf.Download(rawCounts); err != nil {
		t.Fatalf("download counters: %v", err)
	}
	accepted = binary.LittleEndian.Uint32(rawCounts[0:])
	strict = binary.LittleEndian.Uint32(rawCounts[4:])
	stored := min(int(accepted), capacity)
	raw := make([]byte, stored*finderWindowRecord*4)
	if stored > 0 {
		if err := out.DownloadAt(0, raw); err != nil {
			t.Fatalf("download survivors: %v", err)
		}
	}
	found = make([]finderWindow, stored)
	for i := range found {
		at := i * finderWindowRecord * 4
		found[i].key = binary.LittleEndian.Uint32(raw[at:])
		for k := range found[i].boundary {
			found[i].boundary[k] = binary.LittleEndian.Uint32(raw[at+(k+1)*4:])
		}
	}
	return found, accepted, strict
}

// windowsFromBoundaries is the oracle for the fused prototype: every six
// consecutive boundaries of every line, classified by the same acceptance test.
// It reads the boundary lists prototypes 1 and 2 produce, so agreement between
// the two designs is checked against shared ground truth rather than against
// one of them being declared correct.
//
// The classification is three-way, not two. Acceptance is a ratio comparison
// evaluated in f32, and a run pattern such as 5-5-10-5-5 sits exactly on the
// tolerance: the device compiler is free to contract the division into a
// reciprocal multiply, which moves the last bit and flips the answer. Demanding
// that the host reproduce those bits would be pinning down float behaviour this
// project has deliberately left free. So a window near the tie is reported as
// undecided and the test permits either answer, while everything clear of the
// tie must match exactly - which is what actually exercises the seam carry.
func windowsFromBoundaries(boundaries [][]uint32) (accepted, undecided []finderWindow) {
	for key, list := range boundaries {
		for t := 0; t+5 < len(list); t++ {
			w := finderWindow{key: uint32(key)}
			copy(w.boundary[:], list[t:t+6])
			switch classifyFinderWindow(
				list[t+1]-list[t], list[t+2]-list[t+1], list[t+3]-list[t+2],
				list[t+4]-list[t+3], list[t+5]-list[t+4],
			) {
			case windowAccepted:
				accepted = append(accepted, w)
			case windowUndecided:
				undecided = append(undecided, w)
			}
		}
	}
	return accepted, undecided
}

type windowVerdict int

const (
	windowRejected windowVerdict = iota
	windowAccepted
	windowUndecided
)

// classifyFinderWindow evaluates checkPatternCross in float64 and reports
// undecided when any comparison is closer to its threshold than f32 can
// resolve.
func classifyFinderWindow(s0, s1, s2, s3, s4 uint32) windowVerdict {
	if s1 == 0 || s2 == 0 || s3 == 0 {
		return windowRejected
	}
	layer := float64(s1+s2+s3) / 3
	tol := layer / 2
	// A few f32 ulps at the magnitude the comparison is made, which covers a
	// reciprocal multiply substituted for the division on either side.
	eps := tol * 4 * 1.1920929e-7
	rejected, undecided := false, false
	// less records that value < limit is required to accept.
	less := func(value, limit float64) {
		switch {
		case math.Abs(value-limit) <= eps:
			undecided = true
		case value >= limit:
			rejected = true
		}
	}
	less(math.Abs(layer-float64(s1)), tol)
	less(math.Abs(layer-float64(s2)), tol)
	less(math.Abs(layer-float64(s3)), tol)
	less(0.5*tol, float64(s0))
	less(0.5*tol, float64(s4))
	less(math.Abs(float64(s1)-float64(s3)), tol)
	switch {
	case rejected:
		return windowRejected
	case undecided:
		return windowUndecided
	default:
		return windowAccepted
	}
}

// The fused prototype trades the boundary buffer for workgroup state carried
// across block seams, which is exactly where a windowing kernel breaks: a
// window spanning a seam needs the five boundaries that preceded the block's
// own first one. Carrying four instead of five drops one window per seam, and
// nothing about the kernel's output shape would reveal it. So its survivors are
// compared against the windows enumerated from the boundary lists, over angled
// sweeps whose lines cross many seams.
func TestGPUFinderWindowsMatchBoundaryWindows(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	kernels := newGPUDecodeKernels(device)
	t.Cleanup(func() {
		_ = kernels.Close()
		_ = device.Close()
	})
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)

	// A module-like mask so accepted windows are plentiful rather than
	// incidental: agreement on zero survivors would prove nothing.
	mask := func(x, y int) bool { return (x/5+y/5)%2 == 0 }
	// Ballot support and a full-subgroup guarantee are separate capabilities and
	// both are needed, so the gate is the same one production selection uses
	// rather than a partitioning probe that a ballot-less device would pass.
	subgroups, err := kernels.subgroupKernelsUsable()
	if err != nil {
		t.Fatalf("device advertises ballot support but the ballot kernel did not build: %v", err)
	}

	for _, variant := range finderWindowVariants() {
		for _, deg := range []float64{0, 15, 45, 75} {
			t.Run(fmt.Sprintf("%s/%.0f degrees", variant.name, deg), func(t *testing.T) {
				if variant.subgroup && !subgroups {
					t.Skip("this adapter cannot build the ballot kernels; the portable twin is its route")
				}
				const width, height = 400, 400
				layout := variant.layout
				geom := sweepGeometry(width, height, deg, 11)
				packed, planeWords := packFinderRunsMasks(layout, width, height, func(x, y, channel int) bool {
					return channel == 1 && mask(x, y)
				})
				capacity := geom.lineLength + 8
				boundaries, _ := runFinderRuns(t, device, kernels,
					finderRunsVariant{"hillis", layout, (*gpuDecodeKernels).finderRunsHillis},
					width, height, 1<<1, capacity, packed, planeWords, geom)
				want, undecided := windowsFromBoundaries(boundaries)
				if len(want) == 0 {
					t.Fatal("the case accepted no windows, so it is not testing agreement")
				}

				got, accepted, strict := runFinderWindows(t, device, kernels, variant,
					width, height, 1<<1, len(want)+len(undecided)+64, packed, planeWords, geom)
				assertWindowsCover(t, got, want, undecided)
				if int(accepted) != len(got) {
					t.Fatalf("fused kernel reported %d accepted windows but wrote %d", accepted, len(got))
				}
				if strict > accepted {
					t.Fatalf("strict count %d exceeds accepted %d", strict, accepted)
				}
			})
		}
	}
}

// assertWindowsCover checks the fused kernel emitted every clearly accepted
// window and nothing outside the accepted-or-undecided set. Comparison is by
// set because the kernel appends in whatever order its blocks reserve slots.
func assertWindowsCover(t *testing.T, got, want, undecided []finderWindow) {
	t.Helper()
	index := func(list []finderWindow) map[finderWindow]bool {
		out := make(map[finderWindow]bool, len(list))
		for _, w := range list {
			out[w] = true
		}
		return out
	}
	gotIndex, wantIndex, tieIndex := index(got), index(want), index(undecided)
	if len(got) != len(gotIndex) {
		t.Fatalf("fused kernel emitted %d records for %d distinct windows, so it duplicated some", len(got), len(gotIndex))
	}
	for w := range wantIndex {
		if !gotIndex[w] {
			t.Fatalf("window %v is clearly accepted but the fused kernel did not emit it", w)
		}
	}
	for w := range gotIndex {
		if !wantIndex[w] && !tieIndex[w] {
			t.Fatalf("fused kernel emitted window %v, which the boundary oracle rejects outright", w)
		}
	}
}

// sweepGeometry builds the same line family sweepDirection walks: lines at the
// given angle, spaced step apart perpendicular to it, indexed so that the
// frame's corners bound the perpendicular offset.
func sweepGeometry(width, height int, deg float64, step float64) finderRunsGeometry {
	dir := newScanDirection(deg)
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
		lineCount:  int((qHi-qLo)/step) + 1,
		lineLength: width + height,
	}
}

// finderScanParams serializes the parameter block every directional prototype
// reads.
func finderScanParams(
	width, height int,
	channelMask uint32,
	capacity, planeWords int,
	geom finderRunsGeometry,
) []byte {
	params := make([]byte, finderRunsParamsWords*4)
	binary.LittleEndian.PutUint32(params[0:], uint32(width))
	binary.LittleEndian.PutUint32(params[4:], uint32(height))
	binary.LittleEndian.PutUint32(params[8:], channelMask)
	binary.LittleEndian.PutUint32(params[12:], uint32(geom.lineLength))
	binary.LittleEndian.PutUint32(params[16:], math.Float32bits(geom.dx))
	binary.LittleEndian.PutUint32(params[20:], math.Float32bits(geom.dy))
	binary.LittleEndian.PutUint32(params[24:], math.Float32bits(geom.nx))
	binary.LittleEndian.PutUint32(params[28:], math.Float32bits(geom.ny))
	binary.LittleEndian.PutUint32(params[32:], math.Float32bits(geom.qLo))
	binary.LittleEndian.PutUint32(params[36:], math.Float32bits(geom.qStep))
	binary.LittleEndian.PutUint32(params[40:], uint32(geom.lineCount))
	binary.LittleEndian.PutUint32(params[44:], uint32(capacity))
	binary.LittleEndian.PutUint32(params[48:], uint32(planeWords))
	return params
}
