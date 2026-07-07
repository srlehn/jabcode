package decode

import (
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/spec"
)

// TestPaletteReadUndersizedMatrix checks that reading a palette for a color mode
// whose metadata does not fit the given matrix fails gracefully rather than
// panicking on the metadata walk or the palette-placement table. A 16-color
// symbol needs a larger version than 21x21; both readers must bail, not index out
// of range.
func TestPaletteReadUndersizedMatrix(t *testing.T) {
	sym := &core.DecodedSymbol{}
	sym.Meta.NC = 3 // 16 colors
	matrix := core.NewBitmap(21, 21, 4)
	if got := ReadColorPaletteInPrimary(matrix, sym, make([]byte, 21*21), new(int), new(int), new(int)); got >= 0 {
		t.Errorf("ReadColorPaletteInPrimary accepted an undersized 16-color matrix: got %d, want < 0", got)
	}
	// The secondary reader must not panic on the same input.
	sym2 := &core.DecodedSymbol{}
	sym2.Meta.NC = 3
	readColorPaletteInSecondary(matrix, sym2, make([]byte, 21*21))
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
	sym := &core.DecodedSymbol{}
	for range spec.ColorPaletteNumber {
		sym.Palette = append(sym.Palette, base...)
	}
	normPalette := make([]float64, colorNumber*4*spec.ColorPaletteNumber)
	NormalizeColorPalette(sym, normPalette, colorNumber)
	palThs := make([]float64, 3*spec.ColorPaletteNumber)
	for i := range spec.ColorPaletteNumber {
		th := PaletteThreshold(sym.Palette[colorNumber*3*i:], colorNumber)
		palThs[i*3+0], palThs[i*3+1], palThs[i*3+2] = th[0], th[1], th[2]
	}
	matrix := core.NewBitmap(21, 21, 4)
	// A grey module nearest the bottom-left corner palette, whose entry 7
	// would sit past the palette slice's end: grey normalizes to (1,1,1),
	// exactly the normalized non-zero black.
	x, y := 2, 18
	o := matrix.Offset(x, y)
	matrix.Pix[o], matrix.Pix[o+1], matrix.Pix[o+2], matrix.Pix[o+3] = 200, 200, 200, 255
	if got := DecodeModuleHD(matrix, sym.Palette, colorNumber, normPalette, palThs, x, y); int(got) >= colorNumber {
		t.Errorf("DecodeModuleHD returned index %d, want < %d", got, colorNumber)
	}
}
