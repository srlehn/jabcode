package ecc

import (
	"math/bits"
	"sync"
)

// sysKey identifies a systematic parity-check matrix. The constructions are
// seeded by the fixed standard constants, so the matrix depends only on the
// code weights, the capacity and which systematic rearrangement (encoder or
// decoder) is wanted.
type sysKey struct {
	wc, wr, capacity int
	encode           bool
}

// sysEntry is a cached systematic matrix with its rank, plus - for decoder
// matrices - the edge adjacency the iterative decoders walk. The entry is
// shared across callers and goroutines and must never be modified.
type sysEntry struct {
	A    *bitMatrix
	rank int
	idx  *ldpcIndex // decoder entries only (encode: nil)
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
	e := systematicEntry(wc, wr, capacity, encode)
	return e.A, e.rank
}

// systematicParityCheckIndexed is systematicParityCheck for the decoder
// rearrangement, also returning the matrix's edge adjacency (built once and
// cached with the matrix).
func systematicParityCheckIndexed(wc, wr, capacity int) (*bitMatrix, int, *ldpcIndex) {
	e := systematicEntry(wc, wr, capacity, false)
	return e.A, e.rank, e.idx
}

func systematicEntry(wc, wr, capacity int, encode bool) sysEntry {
	key := sysKey{wc, wr, capacity, encode}
	sysMu.RLock()
	e, ok := sysCache[key]
	sysMu.RUnlock()
	if ok {
		return e
	}
	A := parityCheckMatrix(wc, wr, capacity)
	rank := gaussJordan(A, encode)
	e = sysEntry{A: A, rank: rank}
	if !encode {
		e.idx = newLDPCIndex(A)
	}
	sysMu.Lock()
	sysCache[key] = e
	sysMu.Unlock()
	return e
}

// ldpcIndex is the edge adjacency of a decoder parity-check matrix: per row
// the ascending list of set columns, and per column the ascending list of
// rows containing it with the matching slot in that row's list. The decoder
// rearrangement only permutes rows and columns of the Gallager construction,
// so a message-code row holds exactly wr set bits and a column exactly wc -
// walking these lists replaces the dense (row x column) bit probing of the
// iterative decoders with identical results at a fraction of the work, and
// lets belief propagation store its per-edge messages contiguously
// (rowOff[r]+slot) instead of in a rows x columns buffer.
type ldpcIndex struct {
	rowCols   [][]int32 // per row: ascending set columns
	rowOff    []int32   // per row: cumulative edge offset; rowOff[rows] = total edges
	colRows   [][]int32 // per column: ascending rows containing the column
	colSlots  [][]int32 // per column: slot of the column in that row's rowCols
	maxColDeg int
}

func newLDPCIndex(A *bitMatrix) *ldpcIndex {
	idx := &ldpcIndex{
		rowCols:  make([][]int32, A.rows),
		rowOff:   make([]int32, A.rows+1),
		colRows:  make([][]int32, A.cols),
		colSlots: make([][]int32, A.cols),
	}
	for r := range A.rows {
		var cols []int32
		for k, w := range A.row(r) {
			for ; w != 0; w &= w - 1 {
				cols = append(cols, int32(k*64+bits.TrailingZeros64(w)))
			}
		}
		idx.rowCols[r] = cols
		idx.rowOff[r+1] = idx.rowOff[r] + int32(len(cols))
		for slot, c := range cols {
			idx.colRows[c] = append(idx.colRows[c], int32(r))
			idx.colSlots[c] = append(idx.colSlots[c], int32(slot))
			idx.maxColDeg = max(idx.maxColDeg, len(idx.colRows[c]))
		}
	}
	return idx
}
