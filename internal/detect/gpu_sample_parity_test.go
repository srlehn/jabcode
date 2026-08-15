//go:build !js

package detect

import (
	"bytes"
	"image"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
)

// gpuSampleTolerance is how far a device module value may sit from the host's.
// The two agree on which source pixels a module covers and on how they are
// weighted; they differ only in that the host accumulates in f64 and the device
// in f32, against warped positions carried through f32 coefficients. That moves
// a sample to a neighbouring source pixel when a warped coordinate lands within
// about a thousandth of a pixel of an integer, which a tent weight near the
// footprint edge then almost cancels.
//
// One count is therefore expected on a minority of modules and is invisible to
// palette classification, which decides between colours hundreds of counts
// apart. Anything wider would mean the two samplers disagree about geometry
// rather than about arithmetic, so the tolerance is deliberately too tight to
// absorb that.
const gpuSampleTolerance = 1

// gpuSampleTestQuad returns the four finder centres of a symbol of the given
// side and module size, centred in the frame and slightly skewed so the
// transform is projective rather than affine. The centres sit 3.5 modules
// inside the symbol, which is what PerspectiveTransform expects and what makes
// the whole grid land on the image instead of extrapolating off it.
func gpuSampleTestQuad(width, height int, side image.Point, module float64) [4]core.PointF {
	spanX := float64(side.X) * module
	spanY := float64(side.Y) * module
	originX := (float64(width) - spanX) / 2
	originY := (float64(height) - spanY) / 2
	inset := 3.5 * module
	left := originX + inset
	right := originX + spanX - inset
	top := originY + inset
	bottom := originY + spanY - inset
	skew := module / 3
	return [4]core.PointF{
		core.Pt(left, top), core.Pt(right, top+skew),
		core.Pt(right-skew, bottom), core.Pt(left+skew/2, bottom-skew/2),
	}
}

// TestGPUSampleSymbolMatchesHost holds the device sampler to the host sampler
// on the same resident balanced image, across both sampling regimes and with a
// per-channel offset. The regimes are selected by module extent, so the cases
// pick side sizes that put the same image on either side of the threshold.
func TestGPUSampleSymbolMatchesHost(t *testing.T) {
	const width = 257
	const height = 193
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	input, err := device.NewBuffer(width * height * 4)
	if err != nil {
		_ = device.Close()
		t.Fatalf("allocate GPU sampler test input: %v", err)
	}
	resident, err := newGPUResidentBinarizerWithDevice(device, width, height)
	if err != nil {
		_ = input.Close()
		_ = device.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		if err := resident.Close(); err != nil {
			t.Errorf("close resident GPU binarizer: %v", err)
		}
		if err := input.Close(); err != nil {
			t.Errorf("close GPU sampler test input: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close GPU sampler test device: %v", err)
		}
	})

	bm := gpuTestBitmap(width, height)
	if err := input.Upload(bm.Pix); err != nil {
		t.Fatalf("upload GPU sampler test input: %v", err)
	}
	if _, _, _, err := resident.Binarize(input, width, height, nil, false, 0); err != nil {
		t.Fatalf("resident GPU Binarize: %v", err)
	}
	balanced, err := resident.DownloadBalanced(width, height)
	if err != nil {
		t.Fatalf("download resident GPU balanced image: %v", err)
	}

	tests := []struct {
		name   string
		side   image.Point
		module float64
		delta  [3]core.PointF
	}{
		{name: "footprint", side: image.Pt(17, 13), module: 12},
		{name: "centre kernel", side: image.Pt(31, 23), module: 6},
		{
			name:   "channel offsets",
			side:   image.Pt(17, 13),
			module: 12,
			delta:  [3]core.PointF{{X: 1.5, Y: -0.5}, {}, {X: -1.25, Y: 0.75}},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			quad := gpuSampleTestQuad(width, height, test.side, test.module)
			pt := core.PerspectiveTransform(quad[0], quad[1], quad[2], quad[3], test.side)
			modW, modH := moduleExtent(pt, test.side)
			footprint := min(modW, modH) >= legacySampleBelowPx
			if footprint != (test.name != "centre kernel") {
				t.Fatalf("case selects the wrong regime: module extent %.2fx%.2f", modW, modH)
			}
			want := SampleSymbolOffset(balanced, pt, test.side, test.delta)
			if want == nil {
				t.Fatal("host sampler rejected the test geometry")
			}
			got, err := resident.SampleSymbol(width, height, pt, test.side, test.delta)
			if err != nil {
				t.Fatalf("GPU SampleSymbol: %v", err)
			}
			if got == nil {
				t.Fatal("GPU sampler rejected geometry the host sampler accepted")
			}
			// A device sample keeps its modules resident, and comparing them is
			// the one thing that needs them here.
			if !resident.MaterializeGrid(got) {
				t.Fatal("could not materialize the sampled grid")
			}
			if got.Width != want.Width || got.Height != want.Height ||
				got.Channels != want.Channels || len(got.Pix) != len(want.Pix) {
				t.Fatalf(
					"GPU grid %dx%dx%d (%d bytes), want %dx%dx%d (%d bytes)",
					got.Width, got.Height, got.Channels, len(got.Pix),
					want.Width, want.Height, want.Channels, len(want.Pix),
				)
			}
			var differing, worst int
			for i := range want.Pix {
				diff := int(got.Pix[i]) - int(want.Pix[i])
				if diff < 0 {
					diff = -diff
				}
				if diff == 0 {
					continue
				}
				differing++
				if diff > worst {
					worst = diff
				}
			}
			if worst > gpuSampleTolerance {
				t.Errorf(
					"GPU sampler differs from host by up to %d counts over %d of %d values",
					worst, differing, len(want.Pix),
				)
			}
			t.Logf(
				"%s: %d of %d values differ, worst %d counts",
				test.name, differing, len(want.Pix), worst,
			)
		})
	}
}

// blockAt carves one alignment-style block out of a whole-symbol transform: the
// block's own module coordinates warp to the same image points the symbol
// coordinates at its origin do, which is what the alignment path builds when
// every pattern in a rectangle was found where it was predicted.
func blockAt(pt core.Perspective, origin, size image.Point) AlignmentBlock {
	src := [4]core.PointF{
		core.Pt(0.5, 0.5),
		core.Pt(float64(size.X)-0.5, 0.5),
		core.Pt(float64(size.X)-0.5, float64(size.Y)-0.5),
		core.Pt(0.5, float64(size.Y)-0.5),
	}
	var dst [4]core.PointF
	for i, s := range src {
		dst[i] = pt.Warp(core.Pt(s.X+float64(origin.X), s.Y+float64(origin.Y)))
	}
	return AlignmentBlock{
		Transform: core.QuadToQuad(src, dst),
		Size:      size,
		Origin:    origin,
	}
}

// TestGPUSampleBlocksMatchesHost holds the resident block scatter to the host
// assembly it replaces. The cases matter beyond value parity: blocks overlap
// and the later one has to win on both sides, a block may hang past the grid,
// and modules no block covers have to read zero rather than whatever the buffer
// held for the previous symbol.
func TestGPUSampleBlocksMatchesHost(t *testing.T) {
	const width = 257
	const height = 193
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	input, err := device.NewBuffer(width * height * 4)
	if err != nil {
		_ = device.Close()
		t.Fatalf("allocate GPU sampler test input: %v", err)
	}
	resident, err := newGPUResidentBinarizerWithDevice(device, width, height)
	if err != nil {
		_ = input.Close()
		_ = device.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		_ = resident.Close()
		_ = input.Close()
		_ = device.Close()
	})
	bm := gpuTestBitmap(width, height)
	if err := input.Upload(bm.Pix); err != nil {
		t.Fatalf("upload GPU sampler test input: %v", err)
	}
	if _, _, _, err := resident.Binarize(input, width, height, nil, false, 0); err != nil {
		t.Fatalf("resident GPU Binarize: %v", err)
	}
	balanced, err := resident.DownloadBalanced(width, height)
	if err != nil {
		t.Fatalf("download resident GPU balanced image: %v", err)
	}

	side := image.Pt(25, 21)
	quad := gpuSampleTestQuad(width, height, side, 7)
	pt := core.PerspectiveTransform(quad[0], quad[1], quad[2], quad[3], side)

	// The last block samples one module further into the symbol than it claims
	// to occupy, so what it writes differs from what the wide block already put
	// there. Blocks built honestly overlap with identical values, and then the
	// order they land in would prove nothing. It also hangs two modules past
	// the grid, which the scatter has to drop rather than wrap.
	shifted := blockAt(pt, image.Pt(14, 6), image.Pt(13, 9))
	shifted.Origin = image.Pt(13, 5)

	// Widest first, as the rectangle selection sorts them, so the later blocks
	// overwrite modules the first one already wrote. The bottom rows are left
	// uncovered on purpose.
	blocks := []AlignmentBlock{
		blockAt(pt, image.Pt(0, 0), image.Pt(25, 14)),
		blockAt(pt, image.Pt(0, 6), image.Pt(13, 8)),
		shifted,
	}

	want := SampleAlignmentBlocks(balanced, side, blocks)
	if want == nil {
		t.Fatal("host assembler rejected the test geometry")
	}
	got, err := resident.SampleBlocks(width, height, side, blocks)
	if err != nil {
		t.Fatalf("GPU SampleBlocks: %v", err)
	}
	if got == nil {
		t.Fatal("GPU assembler rejected geometry the host assembler accepted")
	}
	if !resident.MaterializeGrid(got) {
		t.Fatal("could not materialize the assembled grid")
	}
	if got.Width != want.Width || got.Height != want.Height ||
		got.Channels != want.Channels || len(got.Pix) != len(want.Pix) {
		t.Fatalf(
			"GPU grid %dx%dx%d (%d bytes), want %dx%dx%d (%d bytes)",
			got.Width, got.Height, got.Channels, len(got.Pix),
			want.Width, want.Height, want.Channels, len(want.Pix),
		)
	}
	var differing, worst int
	for i := range want.Pix {
		diff := int(got.Pix[i]) - int(want.Pix[i])
		if diff < 0 {
			diff = -diff
		}
		if diff == 0 {
			continue
		}
		differing++
		if diff > worst {
			worst = diff
		}
	}
	if worst > gpuSampleTolerance {
		t.Errorf(
			"GPU assembler differs from host by up to %d counts over %d of %d values",
			worst, differing, len(want.Pix),
		)
	}
	t.Logf("%d of %d values differ, worst %d counts", differing, len(want.Pix), worst)

	// A module the last block covers must carry that block's sample, not the
	// first block's, or the two sides agreed only because nothing overlapped.
	single := []AlignmentBlock{blocks[0]}
	coarse, err := resident.SampleBlocks(width, height, side, single)
	if err != nil {
		t.Fatalf("GPU SampleBlocks for the coarse block alone: %v", err)
	}
	// This second sample took the resident buffer over, which is why the first
	// grid had to be materialized above and not here.
	if !resident.MaterializeGrid(coarse) {
		t.Fatal("could not materialize the coarse grid")
	}
	if resident.MaterializeGrid(&core.Bitmap{Width: side.X, Height: side.Y, Channels: 4}) {
		t.Error("a grid the sampler does not hold was materialized from the resident buffer")
	}
	overlap := (8*side.X + 20) * got.Channels
	if string(coarse.Pix[overlap:overlap+3]) == string(got.Pix[overlap:overlap+3]) {
		t.Error("the offset block left no trace, so block order proves nothing here")
	}

	// Modules under no block read zero, and the row after the last covered one
	// is the cheapest place to see the buffer carrying a previous grid.
	for y := 14; y < side.Y; y++ {
		for x := range side.X {
			offset := (y*side.X + x) * got.Channels
			for c := range got.Channels {
				if got.Pix[offset+c] != 0 {
					t.Fatalf("uncovered module %d,%d channel %d is %d, want 0",
						x, y, c, got.Pix[offset+c])
				}
			}
		}
	}
}

// TestGPUSampleSymbolRejectsOffImageGeometry pins the device's replacement for
// the host sampler's early return: a lane cannot abandon a grid its neighbours
// are filling, so it raises a shared flag instead, and the host has to read
// that as the same refusal.
func TestGPUSampleSymbolRejectsOffImageGeometry(t *testing.T) {
	const width = 129
	const height = 97
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	input, err := device.NewBuffer(width * height * 4)
	if err != nil {
		_ = device.Close()
		t.Fatalf("allocate GPU sampler test input: %v", err)
	}
	resident, err := newGPUResidentBinarizerWithDevice(device, width, height)
	if err != nil {
		_ = input.Close()
		_ = device.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		_ = resident.Close()
		_ = input.Close()
		_ = device.Close()
	})
	bm := gpuTestBitmap(width, height)
	if err := input.Upload(bm.Pix); err != nil {
		t.Fatalf("upload GPU sampler test input: %v", err)
	}
	if _, _, _, err := resident.Binarize(input, width, height, nil, false, 0); err != nil {
		t.Fatalf("resident GPU Binarize: %v", err)
	}
	balanced, err := resident.DownloadBalanced(width, height)
	if err != nil {
		t.Fatalf("download resident GPU balanced image: %v", err)
	}

	side := image.Pt(17, 13)
	// A module size the frame cannot hold puts whole rows of modules outside it.
	outside := gpuSampleTestQuad(width, height, side, 40)
	pt := core.PerspectiveTransform(outside[0], outside[1], outside[2], outside[3], side)
	if want := SampleSymbolOffset(balanced, pt, side, [3]core.PointF{}); want != nil {
		t.Fatal("host sampler accepted geometry the case needs it to reject")
	}
	got, err := resident.SampleSymbol(width, height, pt, side, [3]core.PointF{})
	if err != nil {
		t.Fatalf("GPU SampleSymbol: %v", err)
	}
	if got != nil {
		t.Error("GPU sampler accepted geometry the host sampler rejected")
	}

	// The flag must not survive into the next symbol's grid.
	inside := gpuSampleTestQuad(width, height, side, 6)
	pt = core.PerspectiveTransform(inside[0], inside[1], inside[2], inside[3], side)
	got, err = resident.SampleSymbol(width, height, pt, side, [3]core.PointF{})
	if err != nil {
		t.Fatalf("GPU SampleSymbol after a rejection: %v", err)
	}
	if got == nil {
		t.Error("the reject flag carried over into the next symbol's grid")
	}
}

// TestGPUSampleGridSurvivesLaterSample pins the residency the shared
// current-family sample depends on. One physical sample is decoded once per
// compiled wire variant, and a variant that rejects the declared shape
// resamples at the metadata version, which takes the working grid over. Every
// grid the sampler handed out has to keep answering with its own modules
// afterwards, whether or not it had crossed to the host first, and a grid the
// device never produced has to be refused rather than filled from whatever
// sample the buffer holds now.
func TestGPUSampleGridSurvivesLaterSample(t *testing.T) {
	const width = 257
	const height = 193
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	input, err := device.NewBuffer(width * height * 4)
	if err != nil {
		_ = device.Close()
		t.Fatalf("allocate GPU sampler test input: %v", err)
	}
	resident, err := newGPUResidentBinarizerWithDevice(device, width, height)
	if err != nil {
		_ = input.Close()
		_ = device.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		_ = resident.Close()
		_ = input.Close()
		_ = device.Close()
	})
	bm := gpuTestBitmap(width, height)
	if err := input.Upload(bm.Pix); err != nil {
		t.Fatalf("upload GPU sampler test input: %v", err)
	}
	if _, _, _, err := resident.Binarize(input, width, height, nil, false, 0); err != nil {
		t.Fatalf("resident GPU Binarize: %v", err)
	}

	side := image.Pt(17, 13)
	sampleAt := func(module float64) *core.Bitmap {
		t.Helper()
		quad := gpuSampleTestQuad(width, height, side, module)
		pt := core.PerspectiveTransform(quad[0], quad[1], quad[2], quad[3], side)
		grid, err := resident.SampleSymbol(width, height, pt, side, [3]core.PointF{})
		if err != nil {
			t.Fatalf("GPU SampleSymbol at module %v: %v", module, err)
		}
		if grid == nil {
			t.Fatalf("GPU sampler rejected the module %v geometry", module)
		}
		return grid
	}

	// Two geometries that disagree, so a grid answering with the wrong sample is
	// visible rather than absorbed. They are read here through the current grid,
	// which is the one path that never depended on residency.
	modules := [2]float64{12, 9}
	var want [2][]byte
	for at, module := range modules {
		grid := sampleAt(module)
		if !resident.MaterializeGrid(grid) {
			t.Fatalf("could not materialize the module %v reference sample", module)
		}
		want[at] = append([]byte(nil), grid.Pix...)
	}
	if bytes.Equal(want[0], want[1]) {
		t.Fatal("the two geometries sample alike, so they prove nothing here")
	}

	// More samples than the device can hold, none of them read until every one
	// has been taken. The last few stay readable, which is the capacity the ring
	// is sized for; the ones past it are refused rather than answered from
	// whatever took their slot, and no transfer happens to keep them.
	const live = gpuSampleRetainSlots + 1
	const samples = live + 2
	grids := make([]*core.Bitmap, samples)
	for at := range grids {
		grids[at] = sampleAt(modules[at%len(modules)])
		if grids[at].HasPixels() {
			t.Fatalf("sample %d crossed to the host without being asked", at)
		}
	}
	if got := resident.sampleDisplaced; got != samples-live {
		t.Fatalf("the ring displaced %d grids over %d samples, want %d",
			got, samples, samples-live)
	}
	for at, grid := range grids {
		got := resident.MaterializeGrid(grid)
		if want := at >= samples-live; got != want {
			t.Fatalf("sample %d readable = %t after %d later samples, want %t",
				at, got, samples-1-at, want)
		}
		if got && !bytes.Equal(grid.Pix, want[at%len(modules)]) {
			t.Fatalf("sample %d answered with another sample's modules", at)
		}
	}
	stale := &core.Bitmap{Width: side.X, Height: side.Y, Channels: 4}
	if resident.MaterializeGrid(stale) {
		t.Error("a grid the sampler never produced was filled from a resident buffer")
	}

	// A grid that has crossed frees its slot: it answers from its own pixels and
	// needs no device copy. Without that the ring would displace a live grid
	// while holding slots for ones nobody can lose.
	displaced := resident.sampleDisplaced
	for range 2 * gpuSampleRetainSlots {
		grid := sampleAt(modules[0])
		if !resident.MaterializeGrid(grid) {
			t.Fatal("a fresh sample was not readable")
		}
	}
	if got := resident.sampleDisplaced; got != displaced {
		t.Fatalf("the ring displaced %d more grids while every retained one had crossed",
			got-displaced)
	}
}

// TestGPUSampleRetentionPublishesNothingMidCopy pins the window the retention
// opens. While a slot's copy is recorded but not yet submitted, the slot must
// belong to no grid at all: leaving the previous identity in place would answer
// a materialization with the incoming sample's modules under the old grid's
// name, which is the cross-sample answer the identity check exists to refuse.
func TestGPUSampleRetentionPublishesNothingMidCopy(t *testing.T) {
	const width = 257
	const height = 193
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	input, err := device.NewBuffer(width * height * 4)
	if err != nil {
		_ = device.Close()
		t.Fatalf("allocate GPU sampler test input: %v", err)
	}
	resident, err := newGPUResidentBinarizerWithDevice(device, width, height)
	if err != nil {
		_ = input.Close()
		_ = device.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		_ = resident.Close()
		_ = input.Close()
		_ = device.Close()
	})
	bm := gpuTestBitmap(width, height)
	if err := input.Upload(bm.Pix); err != nil {
		t.Fatalf("upload GPU sampler test input: %v", err)
	}
	if _, _, _, err := resident.Binarize(input, width, height, nil, false, 0); err != nil {
		t.Fatalf("resident GPU Binarize: %v", err)
	}
	side := image.Pt(17, 13)
	quad := gpuSampleTestQuad(width, height, side, 12)
	pt := core.PerspectiveTransform(quad[0], quad[1], quad[2], quad[3], side)
	first, err := resident.SampleSymbol(width, height, pt, side, [3]core.PointF{})
	if err != nil || first == nil {
		t.Fatalf("first GPU SampleSymbol: %v", err)
	}

	// Take the retention as a sampler would, and stop before submitting.
	recorder, err := resident.device.NewRecorder()
	if err != nil {
		t.Fatalf("create retention recorder: %v", err)
	}
	defer recorder.Abort()
	retained, err := resident.retainSampledGrid(recorder)
	if err != nil {
		t.Fatalf("retain the sampled grid: %v", err)
	}
	if retained.current != first {
		t.Fatal("the retention did not take the grid the sampler last produced")
	}
	for slot, held := range resident.sampleRetained {
		if held != nil {
			t.Fatalf("slot %d still names a grid while its copy is unsubmitted", slot)
		}
	}
	if resident.MaterializeGrid(first) {
		t.Error("a grid whose retention has not landed was answered from the device")
	}
	resident.commitRetainedSample(retained)
	if resident.sampleRetained[retained.slot] != first {
		t.Fatal("the committed retention did not publish its grid")
	}
}
