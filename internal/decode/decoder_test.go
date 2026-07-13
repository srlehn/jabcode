package decode

import "testing"

// TestDecodeDataCReferenceUnimplementedModeLatch checks that a bit stream
// latching into the C reference's unimplemented ECI or FNC1 modes ends the
// message cleanly instead of indexing past the character-size table. Corrupted
// streams that still pass best-effort LDPC can produce such a latch.
func TestDecodeDataCReferenceUnimplementedModeLatch(t *testing.T) {
	cases := []struct {
		name string
		bits []byte
	}{
		// ModeUpper value 31 (11111) selects the extension read; its 2-bit
		// value 10 latches ECI; the trailing bits must not be interpreted.
		{"eci", []byte{1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1}},
		// ModeUpper 28 latches ModeLower; Lower 31 + extension 11 latches
		// FNC1.
		{"fnc1", []byte{1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}},
	}
	for _, c := range cases {
		if got := DecodeData(c.bits); len(got) != 0 {
			t.Errorf("%s: DecodeData = %q, want empty", c.name, got)
		}
	}
}

func TestDecodeDataCReferenceOutputUnchanged(t *testing.T) {
	var bits messageBits
	bits.upper(1)
	bits.byteRun('\\')
	bits.upper(2)
	if got := DecodeData(bits); string(got) != "A\\B" {
		t.Fatalf("DecodeData = %q, want %q", got, "A\\B")
	}
}
