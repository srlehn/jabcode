package ecc

import "sync"

// sysKey identifies a systematic parity-check matrix. The constructions are
// seeded by the fixed standard constants, so the matrix depends only on the
// code weights, the capacity and which systematic rearrangement (encoder or
// decoder) is wanted.
type sysKey struct {
	wc, wr, capacity int
	encode           bool
}

// sysEntry is a cached systematic matrix with its rank. The matrix is shared
// across callers and goroutines and must never be modified.
type sysEntry struct {
	A    *bitMatrix
	rank int
}

var (
	sysMu    sync.RWMutex
	sysCache = map[sysKey]sysEntry{}
)

// systematicParityCheck returns the systematic-form parity-check matrix and
// its rank for the given code, memoized: construction plus Gauss-Jordan
// elimination dominate decode time, a cascade repeats them per symbol, and a
// camera stream repeats them per frame. Sound because every consumer only
// reads the matrix (syndrome checks, bit-flip implication counts, generator
// derivation). The bounded symbol-version and ECC-level spaces bound the key
// space, so the cache needs no eviction; concurrent misses build identical
// entries, so the last write winning is harmless.
func systematicParityCheck(wc, wr, capacity int, encode bool) (*bitMatrix, int) {
	key := sysKey{wc, wr, capacity, encode}
	sysMu.RLock()
	e, ok := sysCache[key]
	sysMu.RUnlock()
	if ok {
		return e.A, e.rank
	}
	A := parityCheckMatrix(wc, wr, capacity)
	rank := gaussJordan(A, encode)
	sysMu.Lock()
	sysCache[key] = sysEntry{A, rank}
	sysMu.Unlock()
	return A, rank
}
