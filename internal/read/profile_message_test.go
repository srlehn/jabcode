package read

import (
	"bytes"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/wire"
)

func appendMessageBits(bits []byte, value, width int) []byte {
	for shift := width - 1; shift >= 0; shift-- {
		bits = append(bits, byte(value>>shift&1))
	}
	return bits
}

func TestDecodeSymbolsUsesWireProfileForMessageInterpretation(t *testing.T) {
	var bits []byte
	bits = appendMessageBits(bits, 28, 5) // latch lowercase
	bits = appendMessageBits(bits, 1, 5)
	bits = appendMessageBits(bits, 31, 5)
	bits = appendMessageBits(bits, 3, 2) // ISO numeric shift; C FNC1 sentinel
	bits = appendMessageBits(bits, 2, 4)
	bits = appendMessageBits(bits, 2, 5)

	symbols := []core.DecodedSymbol{{WireProfile: wire.ISO23634, Data: bits}}
	got, ok := decodeSymbolsTraced(nil, [3]*core.Bitmap{}, symbols, 1, nil)
	if !ok || !bytes.Equal(got, []byte("]j1a1b")) {
		t.Fatalf("ISO decode = (%q, %v), want (%q, true)", got, ok, "]j1a1b")
	}

	symbols[0].WireProfile = wire.CReference
	got, ok = decodeSymbolsTraced(nil, [3]*core.Bitmap{}, symbols, 1, nil)
	if !ok || !bytes.Equal(got, []byte("a")) {
		t.Fatalf("C-reference decode = (%q, %v), want (%q, true)", got, ok, "a")
	}
}

func TestDecodeSymbolsRejectsInvalidISOMessageControl(t *testing.T) {
	var bits []byte
	bits = appendMessageBits(bits, 31, 5)
	bits = appendMessageBits(bits, 3, 2)
	bits = appendMessageBits(bits, 6, 3) // reserved additional switch

	symbols := []core.DecodedSymbol{{WireProfile: wire.ISO23634, Data: bits}}
	got, ok := decodeSymbolsTraced(nil, [3]*core.Bitmap{}, symbols, 1, nil)
	if ok || got != nil {
		t.Fatalf("ISO decode = (%q, %v), want (nil, false)", got, ok)
	}
}
