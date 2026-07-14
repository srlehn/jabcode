package ecc

import "github.com/srlehn/jabcode/internal/wire"

// interleaveSeed is the fixed LCG seed used to derive the (de)interleaving
// permutation (INTERLEAVE_SEED in interleave.c).
const interleaveSeed = 226759

// Interleave shuffles data in place using a deterministic back-to-front
// Fisher-Yates pass driven by the seeded generator.
func Interleave(data []byte) {
	InterleaveVariant(data, wire.ISO23634)
}

// InterleaveVariant is Interleave under the selected wire-format variant.
func InterleaveVariant(data []byte, variant wire.Variant) {
	// Ports interleaveData in interleave.c.
	n := len(data)
	if n == 0 {
		return
	}
	i := 0
	for x := range randomValues(variant, interleaveSeed) {
		pos := variantRandIndex(variant, x, n-i)
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
	DeinterleaveVariant(data, wire.ISO23634)
}

// DeinterleaveVariant is Deinterleave under the selected wire-format variant.
func DeinterleaveVariant(data []byte, variant wire.Variant) {
	deinterleave(data, variant)
}

// DeinterleaveFloat applies the byte-deinterleaving permutation to a parallel
// slice, so soft-decision per-bit reliabilities track the bits they describe
// through the same shuffle.
func DeinterleaveFloat(data []float64) {
	DeinterleaveFloatVariant(data, wire.ISO23634)
}

// DeinterleaveFloatVariant applies the selected variant's byte-deinterleaving
// permutation to a parallel float slice.
func DeinterleaveFloatVariant(data []float64, variant wire.Variant) {
	deinterleave(data, variant)
}

// deinterleave inverts Interleave in place for any element type: it replays the
// interleaving permutation on an index array, then scatters the data back to its
// original positions.
func deinterleave[T any](data []T, variant wire.Variant) {
	n := len(data)
	if n == 0 {
		return
	}
	index := make([]int, n)
	for i := range index {
		index[i] = i
	}
	i := 0
	for x := range randomValues(variant, interleaveSeed) {
		pos := variantRandIndex(variant, x, n-i)
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
