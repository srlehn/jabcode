package decode

import "testing"

// TestDecodeDataUnimplementedModeLatch checks that a bit stream latching into
// the unimplemented ECI or FNC1 modes ends the message cleanly instead of
// indexing past the character-size table. Corrupted streams that still pass
// the best-effort LDPC can produce such a latch on degraded captures.
func TestDecodeDataUnimplementedModeLatch(t *testing.T) {
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
