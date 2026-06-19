package jabcode

import (
	"image"
	"testing"
)

// TestEncodeMultiSymbolRoundTrip encodes a two-symbol (docked) JAB Code and
// decodes it back, exercising the full multi-symbol encode + decode pipeline.
func TestEncodeMultiSymbolRoundTrip(t *testing.T) {
	const s = "This is a longer message spanning two docked JAB Code symbols for round-trip testing of the encoder."
	enc := NewEncoder(WithSymbols(
		[]int{0, 2}, // primary at 0, secondary docked below
		[]image.Point{{X: 4, Y: 4}, {X: 4, Y: 4}},
		[]int{0, 0},
	))
	img, err := enc.Encode([]byte(s))
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	got, err := Decode(img)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if string(got) != s {
		t.Errorf("round-trip mismatch:\n got %q\nwant %q", got, s)
	}
}
