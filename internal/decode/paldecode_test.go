package decode

import (
	"bytes"
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

// TestDecodeModuleAbs checks the absolute-RGB classifier the higher-color modes
// use: a module color must resolve to the nearest embedded palette entry, and the
// corner index must fold onto an embedded copy (the higher modes embed two, not
// the four spatial corners nearestPalette ranks among).
func TestDecodeModuleAbs(t *testing.T) {
	const colorNumber = 16
	copies := spec.PaletteCopies(colorNumber) // 2
	palette := make([]byte, colorNumber*3*copies)
	// Both copies: color i is a gray ramp on R, so nearest by absolute RGB is
	// unambiguous and brightness (not hue) separates the entries.
	for c := 0; c < copies; c++ {
		base := colorNumber * 3 * c
		for i := 0; i < colorNumber; i++ {
			palette[base+i*3] = byte(i * 16) // R = 0, 16, ... 240
		}
	}
	// A color at R=158 is nearest entry 10 (R=160). Every corner index must land
	// on it, since both embedded copies hold the same ramp.
	for pIndex := 0; pIndex < 4; pIndex++ {
		if got := decodeModuleAbs([3]byte{158, 1, 2}, palette, colorNumber, pIndex); got != 10 {
			t.Errorf("decodeModuleAbs pIndex=%d = %d, want 10", pIndex, got)
		}
	}
}

// TestInterpolateBlock checks the four gap blocks a 128-/256-color palette copy
// interpolates from its four carried blocks, at the reference weights.
func TestInterpolateBlock(t *testing.T) {
	const span = 3
	p := make([]byte, 8*span)
	set := func(block int, v byte) {
		for j := 0; j < span; j++ {
			p[block*span+j] = v
		}
	}
	// Carried blocks 0, 2, 5, 7 (block1, block3, block6, block8).
	set(0, 10)
	set(2, 40)
	set(5, 130)
	set(7, 190)
	interpolateBlock(p, 0, span)
	want := map[int]byte{
		1: (10 + 40) / 2,    // block2 = mean(block1, block3) = 25
		3: (40*2 + 130) / 3, // block4 = (2*block3 + block6)/3 = 70
		4: (40 + 130*2) / 3, // block5 = (block3 + 2*block6)/3 = 100
		6: (130 + 190) / 2,  // block7 = mean(block6, block8) = 160
	}
	for block, wv := range want {
		for j := 0; j < span; j++ {
			if got := p[block*span+j]; got != wv {
				t.Errorf("interpolateBlock: block %d byte %d = %d, want %d", block, j, got, wv)
			}
		}
	}
}

// TestCopyAndInterpolateSubblock checks the 16-to-32 sub-block expansion: the four
// carried quarters are copied to their slots and the four gaps interpolated.
func TestCopyAndInterpolateSubblock(t *testing.T) {
	p := make([]byte, 192) // src at [0:48], dst at [96:192]
	const src, dst = 0, 96
	// Carried quarters of the source sub-block.
	quarter := func(off, q int, v byte) {
		for j := 0; j < 12; j++ {
			p[off+q*12+j] = v
		}
	}
	quarter(src, 0, 0)
	quarter(src, 1, 30)
	quarter(src, 2, 90)
	quarter(src, 3, 180)
	copyAndInterpolateSubblockFrom16To32(p, dst, src)
	want := map[int]byte{
		0:  0,               // copied src quarter 0
		12: (0 + 30) / 2,    // 15
		24: 30,              // copied src quarter 1
		36: (30*2 + 90) / 3, // 50
		48: (0 + 90*2) / 3,  // 60
		60: 90,              // copied src quarter 2
		72: (90 + 180) / 2,  // 135
		84: 180,             // copied src quarter 3
	}
	for off, wv := range want {
		for j := 0; j < 12; j++ {
			if got := p[dst+off+j]; got != wv {
				t.Errorf("copyAndInterpolateSubblock: dst+%d byte %d = %d, want %d", off, j, got, wv)
			}
		}
	}
}

// TestInterpolatePaletteFillsAndRuns checks the top-level interpolation runs over
// exactly the embedded copy count (two) for 128 and 256 colors without touching
// out-of-range memory, and is a no-op for the counts that embed every color.
func TestInterpolatePaletteFillsAndRuns(t *testing.T) {
	for _, colorNumber := range []int{128, 256} {
		copies := spec.PaletteCopies(colorNumber)
		p := make([]byte, colorNumber*3*copies)
		for i := range p {
			p[i] = byte(i % 251)
		}
		interpolatePalette(p, colorNumber) // must not panic on a two-copy buffer
	}
	// A 64-color palette embeds every color, so interpolation must leave it
	// unchanged.
	p := make([]byte, 64*3*spec.PaletteCopies(64))
	for i := range p {
		p[i] = byte(i % 251)
	}
	before := append([]byte(nil), p...)
	interpolatePalette(p, 64)
	if !bytes.Equal(p, before) {
		t.Error("interpolatePalette changed a 64-color palette (should be a no-op)")
	}
}
