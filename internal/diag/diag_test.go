package diag

import (
	"bytes"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/spec"
)

func TestDiagDecodePrimaryHighColorPaletteCopies(t *testing.T) {
	for _, colors := range []int{128, 256} {
		img, err := encode.Run(encode.Config{Colors: colors, ModuleSize: 1, SymbolNumber: 1}, []byte("diag high color"))
		if err != nil {
			t.Fatalf("colors %d encode: %v", colors, err)
		}
		bm := core.BitmapFromImage(img)
		var sym core.DecodedSymbol
		var report bytes.Buffer
		if ret := diagDecodePrimary(&report, bm, &sym); ret != core.Success {
			t.Fatalf("colors %d diagDecodePrimary = %d\n%s", colors, ret, report.String())
		}
		wantLen := colors * 3 * spec.PaletteCopies(colors)
		if len(sym.Palette) != wantLen {
			t.Fatalf("colors %d palette len = %d, want %d", colors, len(sym.Palette), wantLen)
		}
	}
}

func TestDiagSymbolPaletteLayout(t *testing.T) {
	for _, colors := range []int{8, 128} {
		sym := &core.DecodedSymbol{
			Palette: make([]byte, colors*3*spec.PaletteCopies(colors)),
		}
		sym.Meta.NC = spec.Log2Int(colors) - 1
		gotColors, gotCopies, ok := diagSymbolPaletteLayout(sym)
		if !ok {
			t.Fatalf("colors %d layout rejected", colors)
		}
		if gotColors != colors || gotCopies != spec.PaletteCopies(colors) {
			t.Fatalf("colors %d layout = (%d,%d), want (%d,%d)",
				colors, gotColors, gotCopies, colors, spec.PaletteCopies(colors))
		}
		sym.Palette = sym.Palette[:len(sym.Palette)-1]
		if _, _, ok := diagSymbolPaletteLayout(sym); ok {
			t.Fatalf("colors %d truncated palette accepted", colors)
		}
	}
}
