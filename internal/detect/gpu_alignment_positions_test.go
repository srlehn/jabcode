//go:build !js

package detect

import (
	"image"
	"math"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/encode"
)

// TestGPUAlignmentPositionsFindPatterns holds the explicit-position search to
// the encoder's own placement, which is the only oracle that settles it: the
// host walk is the arm being replaced, so agreeing with it would say which arm
// the other disagrees with rather than which one is right.
//
// Each candidate is seeded a full module away from the truth, the way the
// version walk's predictions arrive. That doubles as the vacuity guard: a search
// that returned its prediction unchanged, or that accepted the first thing in
// the window, would land a module out and fail the bound.
func TestGPUAlignmentPositionsFindPatterns(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Cleanup(func() {
		if err := device.Close(); err != nil {
			t.Errorf("close GPU alignment position device: %v", err)
		}
	})

	rendered, err := encode.Render(encode.Config{
		Colors: 8, ModuleSize: 1, SymbolNumber: 1,
		SymbolVersions: []image.Point{{X: 12, Y: 12}},
	}, []byte("device alignment pattern search over explicit positions"))
	if err != nil {
		t.Fatalf("encode the alignment fixture: %v", err)
	}
	const modulePx = 12
	img, fwd := renderRotatedRGBA(rendered, modulePx, 0)
	bm := core.BitmapFromImage(img)

	input, err := device.NewBuffer(uint64(bm.Width * bm.Height * 4))
	if err != nil {
		t.Fatalf("allocate GPU alignment position input: %v", err)
	}
	resident, err := newGPUResidentBinarizerWithDevice(device, bm.Width, bm.Height)
	if err != nil {
		_ = input.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		if err := resident.Close(); err != nil {
			t.Errorf("close resident GPU binarizer: %v", err)
		}
		if err := input.Close(); err != nil {
			t.Errorf("close GPU alignment position input: %v", err)
		}
	})
	if err := input.Upload(bm.Pix); err != nil {
		t.Fatalf("upload GPU alignment position input: %v", err)
	}
	if _, _, _, err := resident.Binarize(input, bm.Width, bm.Height, nil, false, 1<<1); err != nil {
		t.Fatalf("binarize for the alignment position search: %v", err)
	}

	truth := apGroundTruth(rendered, fwd)
	if len(truth) == 0 {
		t.Fatal("fixture has no interior alignment patterns")
	}
	if len(truth) > gpuAlignMaxPositions {
		truth = truth[:gpuAlignMaxPositions]
	}
	candidates := make([]alignmentCandidate, len(truth))
	for at, want := range truth {
		candidates[at] = alignmentCandidate{
			Center:     core.Pt(want.X+modulePx, want.Y+modulePx),
			ModuleSize: modulePx,
			ModuleMax:  modulePx * 2,
			U:          core.Pt(1, 0),
			V:          core.Pt(0, 1),
		}
	}

	found, err := resident.SearchAlignmentPositions(bm.Width, bm.Height, apx, candidates)
	if err != nil {
		t.Fatalf("device alignment position search: %v", err)
	}
	if len(found) != len(candidates) {
		t.Fatalf("device answered %d candidates, want %d", len(found), len(candidates))
	}
	for at, ap := range found {
		if ap.FoundCount == 0 {
			t.Fatalf("candidate %d at %v found nothing", at, candidates[at].Center)
		}
		if ap.Typ != apx {
			t.Fatalf("candidate %d came back as type %d, want %d", at, ap.Typ, apx)
		}
		// One pixel, against a seed a whole module out. The run midpoints are
		// measured on whole pixels while the truth is a module centre, so half a
		// pixel on each axis is the sampling convention rather than error; the
		// bound is tight enough that a search reporting the offset it tried
		// rather than what it measured fails by an order of magnitude.
		if off := math.Hypot(
			ap.Center.X-truth[at].X, ap.Center.Y-truth[at].Y,
		); off > 1 {
			t.Fatalf("candidate %d landed %v from the truth %v, at %v",
				at, off, truth[at], ap.Center)
		}
		if ap.ModuleSize <= 0 {
			t.Fatalf("candidate %d reports module size %v", at, ap.ModuleSize)
		}
	}
}

// TestGPUAlignmentPositionsRejectEmptyRegion is the other half of the bound: the
// search has to be able to say no. The version walk asks about several candidate
// versions and only one of them can be right, so a search that accepted
// everywhere would confirm whichever version it was asked about first.
func TestGPUAlignmentPositionsRejectEmptyRegion(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Cleanup(func() {
		if err := device.Close(); err != nil {
			t.Errorf("close GPU alignment position device: %v", err)
		}
	})

	const width, height = 320, 320
	bm := core.NewBitmap(width, height, 4)
	for i := range bm.Pix {
		bm.Pix[i] = 255
	}
	input, err := device.NewBuffer(width * height * 4)
	if err != nil {
		t.Fatalf("allocate GPU alignment position input: %v", err)
	}
	resident, err := newGPUResidentBinarizerWithDevice(device, width, height)
	if err != nil {
		_ = input.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		if err := resident.Close(); err != nil {
			t.Errorf("close resident GPU binarizer: %v", err)
		}
		if err := input.Close(); err != nil {
			t.Errorf("close GPU alignment position input: %v", err)
		}
	})
	if err := input.Upload(bm.Pix); err != nil {
		t.Fatalf("upload GPU alignment position input: %v", err)
	}
	if _, _, _, err := resident.Binarize(input, width, height, nil, false, 1<<1); err != nil {
		t.Fatalf("binarize for the alignment position search: %v", err)
	}

	candidates := []alignmentCandidate{{
		Center:     core.Pt(width/2, height/2),
		ModuleSize: 12,
		ModuleMax:  24,
		U:          core.Pt(1, 0),
		V:          core.Pt(0, 1),
	}}
	found, err := resident.SearchAlignmentPositions(width, height, apx, candidates)
	if err != nil {
		t.Fatalf("device alignment position search: %v", err)
	}
	if len(found) != 1 {
		t.Fatalf("device answered %d candidates, want 1", len(found))
	}
	if found[0].FoundCount != 0 {
		t.Fatalf("blank field accepted a pattern at %v", found[0].Center)
	}
}
