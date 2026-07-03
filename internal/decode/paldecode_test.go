package decode

import (
	"testing"

	"github.com/srlehn/jabcode/internal/spec"
)

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

// TestDecodeModuleHDFourColorGray checks that a grey module in a 4-colour
// symbol classifies without panicking. The palette read from a real capture
// has a non-zero black, so a grey module can win the nearest-normalized-colour
// search at index 0 and reach the black/white tie-break - which indexes
// palette entry 7, past a 4-colour corner palette's four entries.
func TestDecodeModuleHDFourColorGray(t *testing.T) {
	const colorNumber = 4
	// Four identical corner palettes: capture-like non-zero black, then
	// magenta, yellow, cyan.
	base := []byte{40, 40, 40, 255, 0, 255, 255, 255, 0, 0, 255, 255}
	sym := &decodedSymbol{}
	for range spec.ColorPaletteNumber {
		sym.palette = append(sym.palette, base...)
	}
	normPalette := make([]float64, colorNumber*4*spec.ColorPaletteNumber)
	normalizeColorPalette(sym, normPalette, colorNumber)
	palThs := make([]float64, 3*spec.ColorPaletteNumber)
	for i := range spec.ColorPaletteNumber {
		th := getPaletteThreshold(sym.palette[colorNumber*3*i:], colorNumber)
		palThs[i*3+0], palThs[i*3+1], palThs[i*3+2] = th[0], th[1], th[2]
	}
	matrix := newBitmap(21, 21, 4)
	// A grey module nearest the bottom-left corner palette, whose entry 7
	// would sit past the palette slice's end: grey normalizes to (1,1,1),
	// exactly the normalized non-zero black.
	x, y := 2, 18
	o := matrix.offset(x, y)
	matrix.pix[o], matrix.pix[o+1], matrix.pix[o+2], matrix.pix[o+3] = 200, 200, 200, 255
	if got := decodeModuleHD(matrix, sym.palette, colorNumber, normPalette, palThs, x, y); int(got) >= colorNumber {
		t.Errorf("decodeModuleHD returned index %d, want < %d", got, colorNumber)
	}
}
