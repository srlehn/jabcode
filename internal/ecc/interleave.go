package ecc

import "github.com/srlehn/jabcode/internal/wire"

// interleaveSeed is the fixed LCG seed used to derive the (de)interleaving
// permutation (INTERLEAVE_SEED in interleave.c).
const interleaveSeed = 226759

// Interleave shuffles data in place using a deterministic back-to-front
// Fisher-Yates pass driven by the seeded generator.
func Interleave(data []byte) {
	InterleaveProfile(data, wire.ISO23634)
}

// InterleaveProfile is Interleave under the selected wire-format profile.
func InterleaveProfile(data []byte, profile wire.Profile) {
	// Ports interleaveData in interleave.c.
	n := len(data)
	if n == 0 {
		return
	}
	i := 0
	for x := range randomValues(profile, interleaveSeed) {
		pos := profileRandIndex(profile, x, n-i)
		j := n - 1 - i
		data[j], data[pos] = data[pos], data[j]
		if i++; i == n {
			break
		}
	}
}

// Deinterleave inverts Interleave in place.
func Deinterleave(data []byte) {
	// Ports deinterleaveData in interleave.c.
	DeinterleaveProfile(data, wire.ISO23634)
}

// DeinterleaveProfile is Deinterleave under the selected wire-format profile.
func DeinterleaveProfile(data []byte, profile wire.Profile) {
	deinterleave(data, profile)
}

// DeinterleaveFloat applies the byte-deinterleaving permutation to a parallel
// slice, so soft-decision per-bit reliabilities track the bits they describe
// through the same shuffle.
func DeinterleaveFloat(data []float64) {
	DeinterleaveFloatProfile(data, wire.ISO23634)
}

// DeinterleaveFloatProfile applies the selected profile's byte-deinterleaving
// permutation to a parallel float slice.
func DeinterleaveFloatProfile(data []float64, profile wire.Profile) {
	deinterleave(data, profile)
}

// deinterleave inverts Interleave in place for any element type: it replays the
// interleaving permutation on an index array, then scatters the data back to its
// original positions.
func deinterleave[T any](data []T, profile wire.Profile) {
	n := len(data)
	if n == 0 {
		return
	}
	index := make([]int, n)
	for i := range index {
		index[i] = i
	}
	i := 0
	for x := range randomValues(profile, interleaveSeed) {
		pos := profileRandIndex(profile, x, n-i)
		j := n - 1 - i
		index[j], index[pos] = index[pos], index[j]
		if i++; i == n {
			break
		}
	}
	tmp := make([]T, n)
	copy(tmp, data)
	for i := range n {
		data[index[i]] = tmp[i]
	}
}
