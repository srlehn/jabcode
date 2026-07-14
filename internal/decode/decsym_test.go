package decode

import (
	"image"
	"slices"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
)

// TestDecodeSymbolStreamGarbage feeds decodeSymbolStream the garbage shapes a
// best-effort LDPC decode can emit: they must fail cleanly (never panic, never
// report a fatal status that would forfeit the alignment-pattern resample).
func TestDecodeSymbolStreamGarbage(t *testing.T) {
	cases := []struct {
		name string
		dec  []byte
	}{
		{"empty", nil},
		{"all-zero (no start flag)", make([]byte, 64)},
		{"flag only, no docked-position field", []byte{1}},
		{"all-ones (phantom docked secondaries, nothing to parse)", []byte{1, 1, 1, 1, 1}},
	}
	for _, c := range cases {
		var sym core.DecodedSymbol
		if got := decodeSymbolStream(c.dec, &sym, 0); got != core.Failure {
			t.Errorf("%s: got status %d, want core.Failure", c.name, got)
		}
	}
}

func TestMaskedExpansionMatchesInPlaceDemask(t *testing.T) {
	size := image.Pt(9, 7)
	dataMap := make([]byte, size.X*size.Y)
	moduleCount := 0
	for x := range size.X {
		for y := range size.Y {
			if (x+2*y)%5 == 0 {
				dataMap[y*size.X+x] = 1
				continue
			}
			moduleCount++
		}
	}
	for _, colorNumber := range []int{4, 8, 16} {
		bitsPerModule := 0
		for colors := colorNumber; colors > 1; colors >>= 1 {
			bitsPerModule++
		}
		raw := make([]byte, moduleCount)
		for i := range raw {
			raw[i] = byte((i*7 + 3) % colorNumber)
		}
		for maskType := range 8 {
			inPlace := append([]byte(nil), raw...)
			demaskSymbol(inPlace, dataMap, size, maskType, colorNumber)
			want := rawModuleData2RawData(inPlace, bitsPerModule)
			got := rawModuleData2MaskedRawData(raw, dataMap, size, maskType, colorNumber, bitsPerModule)
			if !slices.Equal(got, want) {
				t.Fatalf("colors=%d mask=%d: combined masked expansion differs", colorNumber, maskType)
			}
			if !slices.Equal(raw, makeRawModules(moduleCount, colorNumber)) {
				t.Fatalf("colors=%d mask=%d: neutral classifications mutated", colorNumber, maskType)
			}
		}
	}
}

func makeRawModules(moduleCount, colorNumber int) []byte {
	raw := make([]byte, moduleCount)
	for i := range raw {
		raw[i] = byte((i*7 + 3) % colorNumber)
	}
	return raw
}

// TestDecodeSymbolStreamValid parses a minimal well-formed stream: payload,
// an all-zero docked-position field, and the start flag.
func TestDecodeSymbolStreamValid(t *testing.T) {
	dec := []byte{1, 0, 1, 1, 0, 0, 0, 0, 1}
	var sym core.DecodedSymbol
	if got := decodeSymbolStream(dec, &sym, 0); got != core.Success {
		t.Fatalf("got status %d, want core.Success", got)
	}
	if sym.Meta.DockedPosition != 0 {
		t.Errorf("dockedPosition = %04b, want 0", sym.Meta.DockedPosition)
	}
	if want := []byte{1, 0, 1, 1}; !slices.Equal(sym.Data, want) {
		t.Errorf("data = %v, want %v", sym.Data, want)
	}
}
