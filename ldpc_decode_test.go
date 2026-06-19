package jabcode

import "testing"

// TestLDPCRoundTrip checks that decodeLDPChd recovers the original message from
// an error-free encodeLDPC codeword, across the same configurations used for the
// encode golden test (including the large multi-sub-block case).
func TestLDPCRoundTrip(t *testing.T) {
	cases := []struct{ Pn, wc, wr int }{
		{40, 4, 9}, {100, 4, 9}, {64, 3, 7}, {36, 3, 6},
		{252, 4, 7}, {515, 3, 8}, {33, 4, 5}, {3500, 4, 9},
	}
	for _, c := range cases {
		in := make([]byte, c.Pn)
		for i := range in {
			in[i] = ldpcInputBit(i)
		}
		ecc := encodeLDPC(in, c.wc, c.wr)
		got := decodeLDPChd(ecc, c.wc, c.wr)
		if len(got) != c.Pn {
			t.Errorf("Pn=%d wc=%d wr=%d: decoded length %d, want %d", c.Pn, c.wc, c.wr, len(got), c.Pn)
			continue
		}
		for i := range in {
			if got[i] != in[i] {
				t.Errorf("Pn=%d wc=%d wr=%d: bit[%d]=%d, want %d", c.Pn, c.wc, c.wr, i, got[i], in[i])
				break
			}
		}
	}
}

// TestLDPCCorrectsSingleError checks that hard-decision decoding repairs a
// single flipped bit in a sufficiently large codeword (length >= 36, where the
// corrector is deterministic).
func TestLDPCCorrectsSingleError(t *testing.T) {
	const Pn, wc, wr = 100, 4, 9
	in := make([]byte, Pn)
	for i := range in {
		in[i] = ldpcInputBit(i)
	}
	ecc := encodeLDPC(in, wc, wr)
	ecc[7] ^= 1 // inject a single-bit error

	got := decodeLDPChd(ecc, wc, wr)
	for i := range in {
		if got[i] != in[i] {
			t.Fatalf("uncorrected: bit[%d]=%d, want %d", i, got[i], in[i])
		}
	}
}
