package decode

import (
	"slices"
	"testing"
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
		var sym DecodedSymbol
		if got := decodeSymbolStream(c.dec, &sym, 0); got != Failure {
			t.Errorf("%s: got status %d, want Failure", c.name, got)
		}
	}
}

// TestDecodeSymbolStreamValid parses a minimal well-formed stream: payload,
// an all-zero docked-position field, and the start flag.
func TestDecodeSymbolStreamValid(t *testing.T) {
	dec := []byte{1, 0, 1, 1, 0, 0, 0, 0, 1}
	var sym DecodedSymbol
	if got := decodeSymbolStream(dec, &sym, 0); got != Success {
		t.Fatalf("got status %d, want Success", got)
	}
	if sym.Meta.DockedPosition != 0 {
		t.Errorf("dockedPosition = %04b, want 0", sym.Meta.DockedPosition)
	}
	if want := []byte{1, 0, 1, 1}; !slices.Equal(sym.Data, want) {
		t.Errorf("data = %v, want %v", sym.Data, want)
	}
}
