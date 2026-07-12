package ecc

import (
	"math"
	"slices"
	"testing"
)

// TestDecodeLDPCSoft checks the soft-decision decoder on a clean codeword and on
// one with a single flipped bit (exercising belief propagation).
func TestDecodeLDPCSoft(t *testing.T) {
	const Pn, wc, wr = 100, 4, 9
	in := make([]byte, Pn)
	for i := range in {
		in[i] = ldpcInputBit(i)
	}
	ecc := EncodeLDPC(in, wc, wr)
	gross := len(ecc)

	// reliab builds confident, correct soft inputs, optionally corrupting one bit
	// to a low-confidence wrong value (as a noisy channel would).
	reliab := func(flip int) ([]float64, []byte) {
		enc := make([]float64, gross)
		dec := make([]byte, gross)
		for i, b := range ecc {
			if b == 0 {
				enc[i] = 4.0
			} else {
				enc[i] = -4.0
			}
			dec[i] = b
		}
		if flip >= 0 {
			enc[flip] = -enc[flip] / 16 // small magnitude, wrong sign
			dec[flip] ^= 1
		}
		return enc, dec
	}

	// Clean input: belief propagation is not invoked and the message is
	// recovered exactly (matches the reference decodeLDPC).
	enc, dec := reliab(-1)
	if got := decodeLDPC(enc, gross, wc, wr, dec); got != Pn {
		t.Fatalf("clean: decoded length %d, want %d", got, Pn)
	}
	for i := 0; i < Pn; i++ {
		if dec[i] != in[i] {
			t.Fatalf("clean: bit %d = %d, want %d", i, dec[i], in[i])
		}
	}

	// Pathological single low-confidence flip: belief propagation runs and
	// collapses to the all-zero codeword — verified to match the C reference
	// oracle bit for bit, confirming a faithful port.
	enc, dec = reliab(3)
	if got := decodeLDPC(enc, gross, wc, wr, dec); got != Pn {
		t.Fatalf("flip: decoded length %d, want %d", got, Pn)
	}
	for i := 0; i < Pn; i++ {
		if dec[i] != 0 {
			t.Fatalf("flip: bit %d = %d, want 0 (reference behavior)", i, dec[i])
		}
	}
}

// TestDecodeLDPCSigned exercises the additive-evidence entry directly. Its
// input is already a signed channel value, positive for zero and negative for
// one, so the decoder must neither infer a scale from unrelated positions nor
// mutate the retained evidence supplied by an accumulator.
func TestDecodeLDPCSigned(t *testing.T) {
	const Pn, wc, wr = 100, 4, 9
	in := make([]byte, Pn)
	for i := range in {
		in[i] = ldpcInputBit(i)
	}
	codeword := EncodeLDPC(in, wc, wr)
	llr := make([]float64, len(codeword))
	for i, bit := range codeword {
		llr[i] = 4
		if bit != 0 {
			llr[i] = -4
		}
	}
	// One low-confidence wrong bit forces belief propagation while the rest
	// of the signed channel remains trustworthy.
	llr[17] = -llr[17] / 32
	wantInput := slices.Clone(llr)

	dec, ok := DecodeLDPCSigned(llr, wc, wr)
	if !ok {
		t.Fatal("signed decoder rejected a correctable codeword")
	}
	if !slices.Equal(dec, in) {
		t.Fatal("signed decoder recovered the wrong message")
	}
	for i := range llr {
		if math.Float64bits(llr[i]) != math.Float64bits(wantInput[i]) {
			t.Fatalf("signed decoder mutated input at %d", i)
		}
	}
}

func TestDecodeLDPCSignedSubBlocks(t *testing.T) {
	const pn, wc, wr = 3000, 4, 9
	in := make([]byte, pn)
	for i := range in {
		in[i] = ldpcInputBit(i)
	}
	codeword := EncodeLDPC(in, wc, wr)
	llr := make([]float64, len(codeword))
	for i, bit := range codeword {
		llr[i] = 4
		if bit != 0 {
			llr[i] = -4
		}
	}
	// The 5400-bit gross codeword splits into three independently decoded
	// blocks. Give every block one low-confidence wrong bit.
	for _, i := range []int{17, 1817, 3617} {
		llr[i] = -llr[i] / 32
	}
	wantInput := slices.Clone(llr)

	dec, ok := DecodeLDPCSigned(llr, wc, wr)
	if !ok || !slices.Equal(dec, in) {
		t.Fatal("signed decoder failed the multi-block codeword")
	}
	for i := range llr {
		if math.Float64bits(llr[i]) != math.Float64bits(wantInput[i]) {
			t.Fatalf("signed multi-block decoder mutated input at %d", i)
		}
	}
}
