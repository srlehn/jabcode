package ecc

import (
	"math/bits"
	"sync"

	"github.com/srlehn/jabcode/internal/wire"
)

// sysKey identifies a systematic parity-check matrix. The constructions are
// seeded by the fixed standard constants, so the matrix depends only on the
// code weights, the capacity and which systematic rearrangement (encoder or
// decoder) is wanted.
type sysKey struct {
	wc, wr, capacity int
	encode           bool
	profile          wire.Profile
}

// sysEntry is a cached systematic matrix with its rank, plus - for decoder
// matrices - the edge adjacency the iterative decoders walk. The matrix,
// rank and adjacency are shared across callers and goroutines and must never
// be modified; use is the eviction stamp, guarded by sysMu.
type sysEntry struct {
	A    *bitMatrix
	rank int
	idx  *ldpcIndex // decoder entries only (encode: nil)
	use  int64      // last-use stamp for LRU eviction
}

var (
	sysMu    sync.Mutex
	sysCache = map[sysKey]*sysEntry{}
	sysUse   int64
)

// sysCacheMax bounds the entry count. Legitimate keys come from the small
// symbol-version and ECC-level spaces, but garbage detections (phantom quads
// whose metadata parses into arbitrary legal-looking parameters) mint junk
// keys indefinitely on a long camera stream. Past the cap the
// least-recently-used entry is evicted, so the keys a session actually
// replays - the real symbol's codes, requested every frame - stay resident
// even after junk has filled the map. Cache content only affects
// construction cost, never results: a rebuilt matrix is identical.
const sysCacheMax = 64

// systematicParityCheck returns the systematic-form parity-check matrix and
// its rank for the given code, memoized: construction plus Gauss-Jordan
// elimination dominate decode time, a cascade repeats them per symbol, and a
// camera stream repeats them per frame. Sound because every consumer only
// reads the matrix (syndrome checks, bit-flip implication counts, generator
// derivation).
func systematicParityCheck(wc, wr, capacity int, encode bool) (*bitMatrix, int) {
	return systematicParityCheckProfile(wc, wr, capacity, encode, wire.ISO23634)
}

func systematicParityCheckProfile(wc, wr, capacity int, encode bool, profile wire.Profile) (*bitMatrix, int) {
	e := systematicEntryProfile(wc, wr, capacity, encode, profile)
	return e.A, e.rank
}

// systematicParityCheckIndexed is systematicParityCheck for the decoder
// rearrangement, also returning the matrix's edge adjacency (built once and
// cached with the matrix).
func systematicParityCheckIndexed(wc, wr, capacity int) (*bitMatrix, int, *ldpcIndex) {
	return systematicParityCheckIndexedProfile(wc, wr, capacity, wire.ISO23634)
}

func systematicParityCheckIndexedProfile(wc, wr, capacity int, profile wire.Profile) (*bitMatrix, int, *ldpcIndex) {
	e := systematicEntryProfile(wc, wr, capacity, false, profile)
	return e.A, e.rank, e.idx
}

func systematicEntry(wc, wr, capacity int, encode bool) *sysEntry {
	return systematicEntryProfile(wc, wr, capacity, encode, wire.ISO23634)
}

func systematicEntryProfile(wc, wr, capacity int, encode bool, profile wire.Profile) *sysEntry {
	key := sysKey{wc: wc, wr: wr, capacity: capacity, encode: encode, profile: profile}
	sysMu.Lock()
	if e, ok := sysCache[key]; ok {
		sysUse++
		e.use = sysUse
		sysMu.Unlock()
		return e
	}
	sysMu.Unlock()

	// Build outside the lock; concurrent misses build identical entries, so
	// whichever insert wins is harmless.
	A := parityCheckMatrix(profile, wc, wr, capacity)
	rank := A.gaussJordan(encode)
	e := &sysEntry{A: A, rank: rank}
	if !encode {
		e.idx = A.decoderIndex()
	}

	sysMu.Lock()
	if cached, ok := sysCache[key]; ok {
		sysUse++
		cached.use = sysUse
		sysMu.Unlock()
		return cached
	}
	if len(sysCache) >= sysCacheMax {
		var oldest sysKey
		first := true
		for k, c := range sysCache {
			if first || c.use < sysCache[oldest].use {
				oldest, first = k, false
			}
		}
		delete(sysCache, oldest)
	}
	sysUse++
	e.use = sysUse
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

func (m *bitMatrix) decoderIndex() *ldpcIndex {
	idx := &ldpcIndex{
		rowCols:  make([][]int32, m.rows),
		rowOff:   make([]int32, m.rows+1),
		colRows:  make([][]int32, m.cols),
		colSlots: make([][]int32, m.cols),
	}
	for r := range m.rows {
		var cols []int32
		for k, w := range m.row(r) {
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
