//go:build !js

package detect

import (
	"image"
	"testing"

	"github.com/srlehn/vulki"
)

// rowFoldPass is one summarized row pass: the resident binarizer that produced
// it, the frame it covers, and how many candidates the current family's channel
// compacted on the device.
//
// It scans the current family's channel alone on purpose. A pass that also
// scans the BSI channel is only summarized when that channel's chain ran too,
// which the untagged build does not compile, so a two-channel pass comes back
// as raw records and never reaches the fold this exercises.
type rowFoldPass struct {
	resident *gpuResidentBinarizer
	frame    image.Point
	step     int
	count    int
}

func rowFoldSession(t *testing.T, verify func(t *testing.T, pass rowFoldPass)) {
	t.Helper()
	const width, height = 360, 300
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	bm := chainParityBitmap(width, height, 21, true)
	input, err := device.NewBuffer(width * height * 4)
	if err != nil {
		_ = device.Close()
		t.Fatalf("allocate row fold input: %v", err)
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
			t.Errorf("close row fold input: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close row fold device: %v", err)
		}
	})
	// Both chains compile in the background, and a pass that beats the compiler
	// is summarized by neither. Compiling here makes every outcome below the
	// stage's rather than the compiler's.
	if err := resident.kernels.compileFinderChains(); err != nil {
		t.Fatalf("compile finder chains: %v", err)
	}
	if err := resident.kernels.compileDirectionalFinderChain(); err != nil {
		t.Fatalf("compile directional finder chain: %v", err)
	}
	if err := input.Upload(bm.Pix); err != nil {
		t.Fatalf("upload row fold input: %v", err)
	}
	_, hits, materialize, err := resident.Binarize(
		input, width, height, nil, false, 1<<currentFamilySeekChannel,
	)
	if err != nil {
		t.Fatalf("binarize with the device chain: %v", err)
	}
	if err := materialize(); err != nil {
		t.Fatalf("materialize device masks: %v", err)
	}
	if hits == nil || !hits.valid || !hits.chained(currentFamilySeekChannel) {
		t.Fatal("the pass did not run the current-family chain on the device")
	}
	count := hits.compactedCount(currentFamilySeekChannel)
	if count == 0 {
		t.Fatal("the pass compacted no row candidates, so it summarized nothing")
	}
	verify(t, rowFoldPass{
		resident: resident,
		frame:    image.Pt(width, height),
		step:     finderRowStride(height, normalDetect),
		count:    count,
	})
}

// TestGPUFinderVerticalSweepIsResident holds the fact the vertical rescan rests
// on: 90 degrees is an ordinary sweep direction. The production set covers
// [0,90) because of where the four finders sit relative to each other, not
// because the sweep machinery stops there, so a column sweep needs no kernel of
// its own - but that is a structural argument, and this is the measurement.
//
// It asserts the sweep comes back resident without materializing either its
// candidate count or counters. The following fold test is the behavioral gate
// for the device-held source itself.
func TestGPUFinderVerticalSweepIsResident(t *testing.T) {
	rowFoldSession(t, func(t *testing.T, pass rowFoldPass) {
		sweeps, err := pass.resident.ScanDirectionBatch(
			pass.frame.X, pass.frame.Y,
			[]scanDirection{newScanDirection(90)}, pass.step, currentFamilySeekChannel,
		)
		if err != nil {
			t.Fatalf("sweep 90 degrees: %v", err)
		}
		if len(sweeps) != 1 {
			t.Fatalf("sweeping one direction returned %d sweeps", len(sweeps))
		}
		if !sweeps[0].resident {
			t.Fatal("the column sweep came back with nothing resident to fold")
		}
		if sweeps[0].outcomes != 0 || sweeps[0].summarized {
			t.Fatalf("the column sweep materialized count=%d summarized=%v",
				sweeps[0].outcomes, sweeps[0].summarized)
		}
	})
}

// TestGPUFinderRowFoldCarriesCandidatesOnlyForATrace pins both halves of what
// the folded list is for. A tracing read gets it, because the finder overlay
// draws the population the selection chose from and a quad over an empty frame
// reads as a detector that found four patterns out of nothing. An ordinary
// decode does not, because the route reads the four selected patterns and the
// list is close to a megabyte that would cross for nobody.
func TestGPUFinderRowFoldCarriesCandidatesOnlyForATrace(t *testing.T) {
	rowFoldSession(t, func(t *testing.T, pass rowFoldPass) {
		fold := func(trace bool) *finderDirQuad {
			t.Helper()
			if err := pass.resident.ResetFinderPools(); err != nil {
				t.Fatalf("reset finder pools: %v", err)
			}
			quad, err := pass.resident.FoldRow(
				pass.frame, currentFamilySeekChannel, pass.count, false, trace)
			if err != nil {
				t.Fatalf("fold the row pass (trace=%v): %v", trace, err)
			}
			if quad == nil {
				t.Fatalf("the device declined a pass it chained itself (trace=%v)", trace)
			}
			return quad
		}
		if plain := fold(false); len(plain.Candidates) != 0 {
			t.Errorf("an ordinary fold carried %d candidates back, want none",
				len(plain.Candidates))
		}
		traced := fold(true)
		if len(traced.Candidates) == 0 {
			t.Fatal("a traced fold carried no candidates, so the overlay has nothing to draw")
		}
		// Nothing here asserts the list is longer than the selection. On a clean
		// fixture the fold can produce exactly the four patterns the selection
		// then keeps, so a length comparison would be a property of the fixture
		// rather than of the list.
	})
}

// TestGPUFinderRowVerticalFoldKeepsEveryType holds the union fold to the one
// property adding a source cannot break: a type the row pass found must still
// be found once the column sweep's candidates fold in with it.
//
// The counts themselves may move - more candidates merge differently, and the
// device is not required to reproduce the host's vertical walk - but a type
// disappearing would mean the rescan removed evidence rather than adding it.
func TestGPUFinderRowVerticalFoldKeepsEveryType(t *testing.T) {
	rowFoldSession(t, func(t *testing.T, pass rowFoldPass) {
		if err := pass.resident.ResetFinderPools(); err != nil {
			t.Fatalf("reset finder pools: %v", err)
		}
		rows, err := pass.resident.FoldRow(
			pass.frame, currentFamilySeekChannel, pass.count, false, true)
		if err != nil {
			t.Fatalf("fold the row pass: %v", err)
		}
		if rows == nil {
			t.Fatal("the device declined a pass it chained and compacted itself")
		}
		if err := pass.resident.ResetFinderPools(); err != nil {
			t.Fatalf("reset finder pools: %v", err)
		}
		union, err := pass.resident.FoldRowVertical(
			pass.frame, currentFamilySeekChannel, pass.count, pass.step, false, true)
		if err != nil {
			t.Fatalf("fold the row pass with its column sweep: %v", err)
		}
		if union == nil {
			t.Fatal("the device answered the row pass but declined it with the rescan")
		}
		for typ, found := range rows.TypeCount {
			if found > 0 && union.TypeCount[typ] == 0 {
				t.Errorf("type %d had %d row candidates and none after the rescan folded in: %v against %v",
					typ, found, union.TypeCount, rows.TypeCount)
			}
		}
	})
}
