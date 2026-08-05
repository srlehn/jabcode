//go:build !js

package detect

import (
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/tables"
)

// alignmentGridFor builds the search request the detector would issue for a
// symbol of the given side version over a located finder quad.
func alignmentGridFor(sideVersion int, corners [4]FinderPattern, sideX, sideY int) alignmentGrid {
	index := sideVersion - 1
	n := tables.APNum[index]
	pos := tables.APPos[index][:n]
	return alignmentGrid{
		nApX: n, nApY: n,
		sideX: sideX, sideY: sideY,
		apType:  fp0,
		corners: corners,
		posX:    pos,
		posY:    pos,
	}
}

// TestGPUAlignmentSearchRunsOnDevice pins the mechanics of the device search:
// the whole grid resolves in one submission, the seeded quad corners come back
// as measurements rather than being searched over, and every interior cell
// carries a usable prediction whether or not it accepted a candidate. What the
// search accepts is covered against the host oracle separately; this is the
// gate that a grid never comes back structurally malformed.
func TestGPUAlignmentSearchRunsOnDevice(t *testing.T) {
	const width, height = 360, 300
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	input, err := device.NewBuffer(width * height * 4)
	if err != nil {
		_ = device.Close()
		t.Fatalf("allocate GPU alignment input: %v", err)
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
			t.Errorf("close GPU alignment input: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close GPU alignment device: %v", err)
		}
	})

	bm := chainParityBitmap(width, height, 31, true)
	if err := input.Upload(bm.Pix); err != nil {
		t.Fatalf("upload GPU alignment input: %v", err)
	}
	if _, _, _, err := resident.Binarize(input, width, height, nil, false, 1<<1); err != nil {
		t.Fatalf("binarize for the alignment search: %v", err)
	}

	// A quad spanning most of the frame, with the corners a located pass would
	// have measured.
	const side = 21
	corners := [4]FinderPattern{
		{Typ: fp0, Center: core.PointF{X: 40, Y: 40}, ModuleSize: 12, FoundCount: 1},
		{Typ: fp1, Center: core.PointF{X: 320, Y: 40}, ModuleSize: 12, FoundCount: 1},
		{Typ: fp2, Center: core.PointF{X: 320, Y: 260}, ModuleSize: 12, FoundCount: 1},
		{Typ: fp3, Center: core.PointF{X: 40, Y: 260}, ModuleSize: 12, FoundCount: 1},
	}
	grid := alignmentGridFor(6, corners, side, side)
	if grid.nApX < 3 {
		t.Fatalf("side version 6 should have an interior cell, got a %dx%d grid", grid.nApX, grid.nApY)
	}

	aps, err := resident.SearchAlignment(width, height, grid)
	if err != nil {
		t.Fatalf("device alignment search: %v", err)
	}
	if len(aps) != grid.nApX*grid.nApY {
		t.Fatalf("device returned %d cells, want %d", len(aps), grid.nApX*grid.nApY)
	}

	cornerIndex := [4]int{
		0,
		grid.nApX - 1,
		(grid.nApY-1)*grid.nApX + grid.nApX - 1,
		(grid.nApY - 1) * grid.nApX,
	}
	for at, index := range cornerIndex {
		got := aps[index]
		want := corners[at]
		if got.FoundCount == 0 {
			t.Fatalf("corner %d came back unfound", at)
		}
		if got.Center.X != want.Center.X || got.Center.Y != want.Center.Y {
			t.Fatalf("corner %d moved from %v to %v; seeded corners must not be searched",
				at, want.Center, got.Center)
		}
	}
	interior := 0
	for index, ap := range aps {
		i, j := index/grid.nApX, index%grid.nApX
		isCorner := (i == 0 || i == grid.nApY-1) && (j == 0 || j == grid.nApX-1)
		if isCorner {
			continue
		}
		interior++
		// Found or not, a cell must carry a finite prediction and a positive
		// module size: the next diagonal extrapolates along both, so a zero
		// here would poison the rest of the grid rather than fail locally.
		if ap.ModuleSize <= 0 {
			t.Fatalf("cell (%d,%d) has module size %v", i, j, ap.ModuleSize)
		}
		if ap.Center.X != ap.Center.X || ap.Center.Y != ap.Center.Y {
			t.Fatalf("cell (%d,%d) centre is not finite: %v", i, j, ap.Center)
		}
	}
	if interior == 0 {
		t.Fatal("grid had no interior cells, so the search was never exercised")
	}
	if testing.Verbose() {
		t.Logf("%dx%d grid, %d interior cells searched", grid.nApX, grid.nApY, interior)
	}
}
