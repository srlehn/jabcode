// Package ecc implements the JAB Code forward-error-correction stage: systematic
// LDPC coding (hard- and soft-decision), the fixed byte (de)interleaving
// permutation, and the seeded PRNG they share. It is wire-compatible with the
// JAB Code reference library.
package ecc

import (
	"iter"
	"math"

	"github.com/srlehn/jabcode/internal/wire"
)

// lcgValues returns an endless sequence of the JAB Code pseudo-random
// generator's 32-bit outputs, seeded with seed. The generator is a 64-bit
// linear congruential generator whose high word is run through MT-style
// tempering; it reproduces the reference library's pseudo_random.c bit for bit.
//
// Modeling the generator as an iter.Seq keeps it a first-class, composable
// stream with no mutable shared state: each call yields an independent
// generator. Sequential consumers range over it; consumers needing on-demand
// pull semantics can bridge with iter.Pull.
func lcgValues(seed uint64) iter.Seq[uint32] {
	return func(yield func(uint32) bool) {
		s := seed
		for {
			s = 6364136223846793005*s + 1
			if !yield(temper(uint32(s >> 32))) {
				return
			}
		}
	}
}

// isoValues returns the ISO/IEC 23634 Annex F pseudo-random sequence. The
// uint32 state reproduces the specified unsigned-long low-word arithmetic.
func isoValues(seed uint64) iter.Seq[uint32] {
	return func(yield func(uint32) bool) {
		next := uint32(seed)
		for {
			next = next*1103515245 + 12345
			if !yield((next / 65536) % 32768) {
				return
			}
		}
	}
}

func randomValues(profile wire.Profile, seed uint64) iter.Seq[uint32] {
	if profile == wire.ISO23634 {
		return isoValues(seed)
	}
	return lcgValues(seed)
}

// temper applies the tempering transform to x (identical to MT19937 tempering).
func temper(x uint32) uint32 {
	x ^= x >> 11
	x ^= (x << 7) & 0x9D2C5680
	x ^= (x << 15) & 0xEFC60000
	x ^= x >> 18
	return x
}

// randIndex maps a 32-bit generator output x to an index in [0, n). It uses the
// same float32 scaling as the reference, so every PRNG-driven permutation in the
// format (interleaving, LDPC matrix construction) is reproduced exactly.
func randIndex(x uint32, n int) int {
	return int(float32(x) / float32(math.MaxUint32) * float32(n))
}

func profileRandIndex(profile wire.Profile, x uint32, n int) int {
	if profile == wire.ISO23634 {
		// Annex F requires an index in [0,n), while its rand routine returns
		// one of 32768 values in [0,32767].
		return int(float32(x) / 32768 * float32(n))
	}
	return randIndex(x, n)
}
