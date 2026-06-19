package jabcode

import (
	"image"
	"testing"
)

// TestEncodeInvalidOptions checks that malformed options return an error instead
// of panicking via table indexing.
func TestEncodeInvalidOptions(t *testing.T) {
	data := []byte("x")
	cases := []struct {
		name string
		enc  *Encoder
	}{
		{"ecc level too high", NewEncoder(WithECCLevel(11))},
		{"ecc level negative", NewEncoder(WithECCLevel(-1))},
		{"unsupported colors", NewEncoder(WithColors(16))},
		{"symbol position out of range", NewEncoder(WithSymbols(
			[]int{0, 61}, []image.Point{{X: 4, Y: 4}, {X: 4, Y: 4}}, []int{0, 0}))},
		{"secondary ecc too high", NewEncoder(WithSymbols(
			[]int{0, 2}, []image.Point{{X: 4, Y: 4}, {X: 4, Y: 4}}, []int{0, 11}))},
		{"mismatched WithSymbols lengths", NewEncoder(WithSymbols(
			[]int{0, 2}, []image.Point{{X: 4, Y: 4}}, []int{0, 0}))},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := tc.enc.Encode(data); err == nil {
				t.Errorf("expected an error, got nil")
			}
		})
	}
}

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
