//go:build !js

package detect

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/srlehn/vulki"
)

const (
	finderRunsWorkgroup   = 256
	finderRunsParamsWords = 12
)

// packFinderRunsMasks builds the packed-mask buffer the run kernel reads: three
// channel bits per pixel, eight pixels per u32, which is the resident
// binarizer's own layout.
func packFinderRunsMasks(width, height int, bit func(x, y, channel int) bool) []byte {
	words := (width*height + 7) / 8
	packed := make([]byte, words*4)
	for y := range height {
		for x := range width {
			pixel := y*width + x
			var word uint32
			for channel := range 3 {
				if bit(x, y, channel) {
					word = 1 << uint((pixel%8)*3+channel)
				} else {
					continue
				}
				offset := (pixel / 8) * 4
				existing := binary.LittleEndian.Uint32(packed[offset:])
				binary.LittleEndian.PutUint32(packed[offset:], existing|word)
			}
		}
	}
	return packed
}

// runFinderRuns dispatches the run kernel over one horizontal direction, where
// line i is image row i, and returns the per-line boundary lists and the counts
// the kernel reported. A horizontal sweep is used because the expected
// boundaries are then exactly the row's own transitions, so the oracle is the
// image rather than a second implementation of the thing under test.
func runFinderRuns(
	t *testing.T,
	device *vulki.Device,
	kernels *gpuDecodeKernels,
	width, height int,
	channelMask uint32,
	capacity int,
	packed []byte,
) (boundaries [][]uint32, counts []uint32) {
	t.Helper()
	kernel, err := kernels.finderRuns()
	if err != nil {
		t.Fatalf("compile finder runs: %v", err)
	}
	masks, err := device.NewBuffer(uint64(len(packed)))
	if err != nil {
		t.Fatalf("allocate masks: %v", err)
	}
	defer func() { _ = masks.Close() }()
	slots := height * 3 * capacity
	out, err := device.NewBuffer(uint64(slots * 4))
	if err != nil {
		t.Fatalf("allocate boundaries: %v", err)
	}
	defer func() { _ = out.Close() }()
	countBuf, err := device.NewBuffer(uint64(height * 3 * 4))
	if err != nil {
		t.Fatalf("allocate counts: %v", err)
	}
	defer func() { _ = countBuf.Close() }()
	paramBuf, err := device.NewBuffer(uint64(finderRunsParamsWords * 4))
	if err != nil {
		t.Fatalf("allocate params: %v", err)
	}
	defer func() { _ = paramBuf.Close() }()

	var params [finderRunsParamsWords * 4]byte
	binary.LittleEndian.PutUint32(params[0:], uint32(width))
	binary.LittleEndian.PutUint32(params[4:], uint32(height))
	binary.LittleEndian.PutUint32(params[8:], channelMask)
	binary.LittleEndian.PutUint32(params[12:], uint32(width))
	binary.LittleEndian.PutUint32(params[16:], math.Float32bits(1)) // dx
	binary.LittleEndian.PutUint32(params[20:], math.Float32bits(0)) // dy
	binary.LittleEndian.PutUint32(params[24:], math.Float32bits(0)) // nx
	binary.LittleEndian.PutUint32(params[28:], math.Float32bits(1)) // ny
	binary.LittleEndian.PutUint32(params[32:], math.Float32bits(0)) // q_lo
	binary.LittleEndian.PutUint32(params[36:], math.Float32bits(1)) // q_step
	binary.LittleEndian.PutUint32(params[40:], uint32(height))      // line_count
	binary.LittleEndian.PutUint32(params[44:], uint32(capacity))    // run_capacity

	bindings, err := kernel.NewBindings(
		vulki.BindBuffer(0, masks),
		vulki.BindBuffer(1, out),
		vulki.BindBuffer(2, paramBuf),
		vulki.BindBuffer(3, countBuf),
	)
	if err != nil {
		t.Fatalf("bind finder runs: %v", err)
	}
	defer func() { _ = bindings.Close() }()

	recorder, err := device.NewRecorder()
	if err != nil {
		t.Fatalf("new recorder: %v", err)
	}
	defer recorder.Abort()
	if err := recorder.Update(masks, 0, packed); err != nil {
		t.Fatalf("upload masks: %v", err)
	}
	if err := recorder.Update(paramBuf, 0, params[:]); err != nil {
		t.Fatalf("upload params: %v", err)
	}
	zero := make([]byte, height*3*4)
	if err := recorder.Update(countBuf, 0, zero); err != nil {
		t.Fatalf("clear counts: %v", err)
	}
	if err := recorder.Dispatch(kernel, bindings, vulki.Workgroups{X: uint32(height), Y: 3, Z: 1}); err != nil {
		t.Fatalf("dispatch finder runs: %v", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		t.Fatalf("run finder runs: %v", err)
	}

	rawCounts := make([]byte, height*3*4)
	if err := countBuf.Download(rawCounts); err != nil {
		t.Fatalf("download counts: %v", err)
	}
	rawOut := make([]byte, slots*4)
	if err := out.Download(rawOut); err != nil {
		t.Fatalf("download boundaries: %v", err)
	}
	counts = make([]uint32, height*3)
	for i := range counts {
		counts[i] = binary.LittleEndian.Uint32(rawCounts[i*4:])
	}
	boundaries = make([][]uint32, height*3)
	for i := range boundaries {
		n := min(int(counts[i]), capacity)
		list := make([]uint32, n)
		for j := range n {
			list[j] = binary.LittleEndian.Uint32(rawOut[(i*capacity+j)*4:])
		}
		boundaries[i] = list
	}
	return boundaries, counts
}

// expectedBoundaries is the oracle: sample 0 always opens a run, and every
// later sample opens one when it differs from its predecessor.
func expectedBoundaries(width int, row func(x int) bool) []uint32 {
	out := []uint32{0}
	for x := 1; x < width; x++ {
		if row(x) != row(x-1) {
			out = append(out, uint32(x))
		}
	}
	return out
}

// The run kernel is the first stage of the GPU-native finder pipeline, and it
// replaced a draft whose cross-block carry could not work. Compiling proved
// nothing about that, so these cases exercise the places where a block-strided
// parallel scan actually breaks: the seams between 256-sample blocks, a partial
// final block, lines with no transitions at all, lines that transition on every
// sample, disabled channels, and capacity overflow.
func TestGPUFinderRunsBoundaries(t *testing.T) {
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

	tests := []struct {
		name  string
		width int
		row   func(x int) bool
	}{
		{"transition at the first block seam", 600, func(x int) bool { return x >= finderRunsWorkgroup }},
		{"transition just before the seam", 600, func(x int) bool { return x >= finderRunsWorkgroup-1 }},
		{"transition just after the seam", 600, func(x int) bool { return x >= finderRunsWorkgroup+1 }},
		{"transition at the second seam", 900, func(x int) bool { return x >= 2*finderRunsWorkgroup }},
		{"constant line has one run", 700, func(int) bool { return false }},
		{"alternating line transitions everywhere", 300, func(x int) bool { return x%2 == 0 }},
		{"partial final block", 260, func(x int) bool { return x >= 200 }},
		{"single sample line", 1, func(int) bool { return true }},
		{"exact block multiple", finderRunsWorkgroup * 2, func(x int) bool { return x >= 300 }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			const height = 2
			packed := packFinderRunsMasks(tc.width, height, func(x, _, channel int) bool {
				return channel == 1 && tc.row(x)
			})
			capacity := tc.width + 8
			got, counts := runFinderRuns(t, device, kernels, tc.width, height, 1<<1, capacity, packed)
			want := expectedBoundaries(tc.width, tc.row)
			for line := range height {
				idx := line*3 + 1
				if int(counts[idx]) != len(want) {
					t.Fatalf("line %d: count = %d, want %d", line, counts[idx], len(want))
				}
				if len(got[idx]) != len(want) {
					t.Fatalf("line %d: %d boundaries, want %d", line, len(got[idx]), len(want))
				}
				for i := range want {
					if got[idx][i] != want[i] {
						t.Fatalf("line %d boundary %d = %d, want %d", line, i, got[idx][i], want[i])
					}
				}
			}
		})
	}
}

// A channel the mask does not request must be left completely alone, including
// its count slot, or a later stage reads another channel's runs as its own.
func TestGPUFinderRunsSkipsDisabledChannels(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	kernels := newGPUDecodeKernels(device)
	t.Cleanup(func() {
		_ = kernels.Close()
		_ = device.Close()
	})
	const width, height = 400, 2
	packed := packFinderRunsMasks(width, height, func(x, _, _ int) bool { return x >= 100 })
	_, counts := runFinderRuns(t, device, kernels, width, height, 1<<1, width+8, packed)
	for line := range height {
		if got := counts[line*3+1]; got != 2 {
			t.Fatalf("line %d requested channel count = %d, want 2", line, got)
		}
		for _, channel := range []int{0, 2} {
			if got := counts[line*3+channel]; got != 0 {
				t.Fatalf("line %d disabled channel %d count = %d, want 0", line, channel, got)
			}
		}
	}
}

// Overflow must stay visible. The kernel drops writes past capacity but reports
// the true count, so the host can tell a truncated list from a complete one and
// grow or reroute; clamping the count instead would hand a later stage a short
// list that looks whole.
func TestGPUFinderRunsReportsOverflow(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	kernels := newGPUDecodeKernels(device)
	t.Cleanup(func() {
		_ = kernels.Close()
		_ = device.Close()
	})
	const width, height = 512, 1
	const capacity = 16
	packed := packFinderRunsMasks(width, height, func(x, _, _ int) bool { return x%2 == 0 })
	got, counts := runFinderRuns(t, device, kernels, width, height, 1<<1, capacity, packed)
	if int(counts[1]) != width {
		t.Fatalf("overflow count = %d, want the true %d", counts[1], width)
	}
	if len(got[1]) != capacity {
		t.Fatalf("stored %d boundaries, want the capacity %d", len(got[1]), capacity)
	}
	for i := range capacity {
		if want := uint32(i); got[1][i] != want {
			t.Fatalf("boundary %d = %d, want %d", i, got[1][i], want)
		}
	}
}
