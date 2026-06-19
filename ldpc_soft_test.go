package jabcode

import "testing"

// TestDecodeLDPCSoft checks the soft-decision decoder on a clean codeword and on
// one with a single flipped bit (exercising belief propagation).
func TestDecodeLDPCSoft(t *testing.T) {
	const Pn, wc, wr = 100, 4, 9
	in := make([]byte, Pn)
	for i := range in {
		in[i] = ldpcInputBit(i)
	}
	ecc := encodeLDPC(in, wc, wr)
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
	// (non_repo_misc/oracle_ldpc_soft) bit for bit, confirming a faithful port.
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
