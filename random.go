// Package jabcode is a pure-Go port of the JAB Code (Just Another Bar Code)
// reference library, a high-capacity 2D color matrix symbology standardized as
// ISO/IEC 23634:2022.
//
// The port mirrors the C reference implementation
// (https://github.com/jabcode/jabcode) closely enough to be bitstream- and
// image-compatible: codes produced here decode with the reference reader and
// vice versa.
package jabcode

import (
	"iter"
	"math"
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
