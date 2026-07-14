package tables

import (
	"slices"
	"testing"

	"github.com/srlehn/jabcode/internal/wire"
)

func TestISOFourColorPlacement(t *testing.T) {
	wantPrimary := [4][4]int{
		{0, 1, 2, 3},
		{0, 3, 2, 1},
		{3, 0, 2, 1},
		{1, 0, 2, 3},
	}
	for copyIndex, want := range wantPrimary {
		got := make([]int, len(want))
		for colorIndex := range got {
			got[colorIndex] = PrimaryPalettePlacementIndexVariant(copyIndex, colorIndex, 4, wire.ISO23634)
		}
		if !slices.Equal(got, want[:]) {
			t.Errorf("primary copy %d = %v, want %v", copyIndex, got, want)
		}
	}
	wantSecondary := []int{1, 3, 2, 0}
	gotSecondary := make([]int, len(wantSecondary))
	for colorIndex := range gotSecondary {
		gotSecondary[colorIndex] = SecondaryPalettePlacementIndexVariant(colorIndex, 4, wire.ISO23634)
	}
	if !slices.Equal(gotSecondary, wantSecondary) {
		t.Errorf("secondary placement = %v, want %v", gotSecondary, wantSecondary)
	}

	wantFinder := []int{0, 0, 3, 1}
	gotFinder := make([]int, len(wantFinder))
	for fp := range gotFinder {
		gotFinder[fp] = FPCoreColorIndex(fp, 1, wire.ISO23634)
	}
	if !slices.Equal(gotFinder, wantFinder) {
		t.Errorf("finder core indices = %v, want %v", gotFinder, wantFinder)
	}
	if got := APNCoreColorIndex(1, wire.ISO23634); got != 1 {
		t.Errorf("U/L alignment core index = %d, want 1", got)
	}
	if got := APXCoreColorIndex(1, wire.ISO23634); got != 3 {
		t.Errorf("X alignment core index = %d, want 3", got)
	}
}

// TestNcMetadataColorIndexParity checks that for the 4- and 8-color modes the Nc
// metadata color index is the legacy value%colorNumber, so those symbols stay
// byte-identical to the reference after the higher-mode remap was added.
func TestNcMetadataColorIndexParity(t *testing.T) {
	for _, tc := range []struct{ nc, colors int }{{1, 4}, {2, 8}} {
		for _, v := range []int{0, 3, 6} {
			if got, want := NcMetadataColorIndexVariant(v, tc.nc, wire.CurrentC), v%tc.colors; got != want {
				t.Errorf("NcMetadataColorIndex(%d, nc=%d) = %d, want %d (parity)", v, tc.nc, got, want)
			}
		}
	}
}

// TestNcMetadataColorIndexHigh checks that above 8 colors the marker maps black,
// cyan and yellow to the per-mode indices carrying those colors (the finder core
// columns), which is what makes Part I readable before the palette in those modes.
func TestNcMetadataColorIndexHigh(t *testing.T) {
	for nc := 1; nc <= 7; nc++ {
		if got := NcMetadataColorIndexVariant(0, nc, wire.CurrentC); got != 0 {
			t.Errorf("black nc=%d: got %d, want 0", nc, got)
		}
		if got, want := NcMetadataColorIndexVariant(3, nc, wire.CurrentC), FPCoreColor[3][nc]; got != want {
			t.Errorf("cyan nc=%d: got %d, want %d", nc, got, want)
		}
		if got, want := NcMetadataColorIndexVariant(6, nc, wire.CurrentC), FPCoreColor[2][nc]; got != want {
			t.Errorf("yellow nc=%d: got %d, want %d", nc, got, want)
		}
	}
}

// TestPalettePlacementIndex checks the placement helpers return the reference
// table for the first eight slots and the identity beyond, so encoder and decoder
// agree on higher-color palettes.
func TestPalettePlacementIndex(t *testing.T) {
	for c := range PrimaryPalettePlacement {
		for i := 0; i < 8; i++ {
			if got, want := PrimaryPalettePlacementIndex(c, i), PrimaryPalettePlacement[c][i]; got != want {
				t.Errorf("PrimaryPalettePlacementIndex(%d, %d) = %d, want %d", c, i, got, want)
			}
		}
		for i := 8; i < 64; i++ {
			if got := PrimaryPalettePlacementIndex(c, i); got != i {
				t.Errorf("PrimaryPalettePlacementIndex(%d, %d) = %d, want identity %d", c, i, got, i)
			}
		}
	}
	for i := 0; i < 8; i++ {
		if got, want := SecondaryPalettePlacementIndex(i), SecondaryPalettePlacement[i]; got != want {
			t.Errorf("SecondaryPalettePlacementIndex(%d) = %d, want %d", i, got, want)
		}
	}
	for i := 8; i < 64; i++ {
		if got := SecondaryPalettePlacementIndex(i); got != i {
			t.Errorf("SecondaryPalettePlacementIndex(%d) = %d, want identity %d", i, got, i)
		}
	}
}
