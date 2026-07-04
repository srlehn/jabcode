package ecc

import "testing"

// TestLDPCRoundTrip checks that DecodeLDPCHard recovers the original message from
// an error-free EncodeLDPC codeword, across the same configurations used for the
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
		ecc := EncodeLDPC(in, c.wc, c.wr)
		got, ok := DecodeLDPCHard(ecc, c.wc, c.wr)
		if !ok {
			t.Errorf("Pn=%d wc=%d wr=%d: clean codeword failed the syndrome check", c.Pn, c.wc, c.wr)
		}
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
// corrector is deterministic), and that the repaired codeword passes the
// post-correction syndrome check.
func TestLDPCCorrectsSingleError(t *testing.T) {
	const Pn, wc, wr = 100, 4, 9
	in := make([]byte, Pn)
	for i := range in {
		in[i] = ldpcInputBit(i)
	}
	ecc := EncodeLDPC(in, wc, wr)
	ecc[7] ^= 1 // inject a single-bit error

	got, ok := DecodeLDPCHard(ecc, wc, wr)
	if !ok {
		t.Fatal("corrected codeword failed the syndrome check")
	}
	for i := range in {
		if got[i] != in[i] {
			t.Fatalf("uncorrected: bit[%d]=%d, want %d", i, got[i], in[i])
		}
	}
}

// TestLDPCReportsUncorrectable checks that corruption beyond the corrector's
// reach reports ok=false instead of passing garbage as a success, while the
// best-effort message keeps its shape for callers that ignore ok.
func TestLDPCReportsUncorrectable(t *testing.T) {
	const Pn, wc, wr = 100, 4, 9
	in := make([]byte, Pn)
	for i := range in {
		in[i] = ldpcInputBit(i)
	}
	cw := EncodeLDPC(in, wc, wr)
	for i := 0; i < len(cw); i += 2 {
		cw[i] ^= 1
	}

	got, ok := DecodeLDPCHard(cw, wc, wr)
	if ok {
		t.Fatal("heavily corrupted codeword passed the syndrome check")
	}
	if len(got) != Pn {
		t.Fatalf("decoded length %d, want %d", len(got), Pn)
	}
}
