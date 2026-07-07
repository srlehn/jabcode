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
