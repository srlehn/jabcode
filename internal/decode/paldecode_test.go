package decode

import "testing"

// TestDecodeUnsupportedColorCount checks that a symbol whose metadata declares an
// unsupported (reserved) colour mode fails to decode rather than panicking on the
// palette-placement table.
func TestDecodeUnsupportedColorCount(t *testing.T) {
	// Nc = 3 corresponds to 16 colours (a reserved mode); readColorPaletteIn*
	// must reject it instead of indexing primaryPalettePlacement[*][8..15].
	sym := &decodedSymbol{}
	sym.meta.Nc = 3
	matrix := newBitmap(21, 21, 4)
	if got := readColorPaletteInPrimary(matrix, sym, make([]byte, 21*21), new(int), new(int), new(int)); got >= 0 {
		t.Errorf("readColorPaletteInPrimary accepted 16-color symbol: got %d, want < 0", got)
	}
	if got := readColorPaletteInSecondary(matrix, sym, make([]byte, 21*21)); got >= 0 {
		t.Errorf("readColorPaletteInSecondary accepted 16-color symbol: got %d, want < 0", got)
	}
}
