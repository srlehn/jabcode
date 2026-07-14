//go:build jabcode_bsi

package encode_test

import (
	"fmt"
	"image"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/wire"
)

func TestBSIColorModesRoundTrip(t *testing.T) {
	for _, colors := range []int{4, 8, 16, 32, 64, 128, 256} {
		t.Run(fmt.Sprintf("%dc", colors), func(t *testing.T) {
			want := []byte(fmt.Sprintf("BSI %d color round trip", colors))
			rendered, err := encode.Render(encode.Config{
				Colors: colors, ModuleSize: 1, Profile: wire.BSI, SymbolNumber: 1,
			}, want)
			if err != nil {
				t.Fatal(err)
			}
			matrix := core.NewBitmap(rendered.SideSize.X, rendered.SideSize.Y, 4)
			for i, colorIndex := range rendered.Matrix {
				off := i * matrix.Channels
				paletteOffset := int(colorIndex) * 3
				copy(matrix.Pix[off:off+3], rendered.Palette[paletteOffset:paletteOffset+3])
				matrix.Pix[off+3] = 255
			}
			var symbol core.DecodedSymbol
			if result := decode.DecodeBSIPrimary(matrix, &symbol); result != core.Success {
				for i := range len(rendered.Palette) {
					if i >= len(symbol.Palette) {
						t.Logf("decoded palette ended at %d bytes, want %d", len(symbol.Palette), len(rendered.Palette))
						break
					}
					if symbol.Palette[i] != rendered.Palette[i] {
						t.Logf("palette[%d] = %d, want %d", i, symbol.Palette[i], rendered.Palette[i])
						break
					}
				}
				t.Fatalf("DecodeBSIPrimary = %d, want %d; metadata=%+v palette=%d data=%d", result, core.Success, symbol.Meta, len(symbol.Palette), len(symbol.Data))
			}
			got, ok := decode.DecodeDataProfile(symbol.Data, wire.BSI)
			if !ok {
				t.Fatal("DecodeDataProfile rejected the corrected payload")
			}
			if string(got) != string(want) {
				t.Fatalf("payload = %q, want %q", got, want)
			}
		})
	}
}

func TestBSIRectangleMetadataRoundTrip(t *testing.T) {
	for _, version := range []image.Point{
		image.Pt(3, 2),
		image.Pt(5, 2),
		image.Pt(9, 2),
		image.Pt(17, 2),
	} {
		t.Run(fmt.Sprintf("%dx%d", version.X, version.Y), func(t *testing.T) {
			want := []byte("BSI rectangle metadata")
			rendered, err := encode.Render(encode.Config{
				Colors: 8, ModuleSize: 1, Profile: wire.BSI, SymbolNumber: 1,
				SymbolVersions: []image.Point{version},
			}, want)
			if err != nil {
				t.Fatal(err)
			}
			matrix := core.NewBitmap(rendered.SideSize.X, rendered.SideSize.Y, 4)
			for i, colorIndex := range rendered.Matrix {
				off := i * matrix.Channels
				paletteOffset := int(colorIndex) * 3
				copy(matrix.Pix[off:off+3], rendered.Palette[paletteOffset:paletteOffset+3])
				matrix.Pix[off+3] = 255
			}
			var symbol core.DecodedSymbol
			if result := decode.DecodeBSIPrimary(matrix, &symbol); result != core.Success {
				t.Fatalf("DecodeBSIPrimary = %d, want %d; metadata=%+v", result, core.Success, symbol.Meta)
			}
			if symbol.Meta.SideVersion != version {
				t.Fatalf("side version = %v, want %v", symbol.Meta.SideVersion, version)
			}
			got, ok := decode.DecodeDataProfile(symbol.Data, wire.BSI)
			if !ok || string(got) != string(want) {
				t.Fatalf("payload = %q, ok=%v, want %q", got, ok, want)
			}
		})
	}
}
