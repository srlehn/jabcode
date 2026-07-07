package spec

import "testing"

// TestPaletteCopies pins the embedded-palette copy count: four for the reference
// 4/8-color layout, two for the higher (ISO Annex G) modes.
func TestPaletteCopies(t *testing.T) {
	cases := map[int]int{4: 4, 8: 4, 16: 2, 32: 2, 64: 2, 128: 2, 256: 2}
	for colorNumber, want := range cases {
		if got := PaletteCopies(colorNumber); got != want {
			t.Errorf("PaletteCopies(%d) = %d, want %d", colorNumber, got, want)
		}
	}
}

// TestPaletteFinderColors pins how many palette colors ride in the finder/alignment
// cores: two for 4/8-color, none for the higher modes (which embed every color).
func TestPaletteFinderColors(t *testing.T) {
	cases := map[int]int{4: 2, 8: 2, 16: 0, 32: 0, 64: 0, 128: 0, 256: 0}
	for colorNumber, want := range cases {
		if got := PaletteFinderColors(colorNumber); got != want {
			t.Errorf("PaletteFinderColors(%d) = %d, want %d", colorNumber, got, want)
		}
	}
}
