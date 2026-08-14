package ecc

import "github.com/srlehn/jabcode/internal/wire"

// PivotSweep reports the pivot column each parity row of a message code takes
// during systematic reduction, or PivotNone where the row proved dependent.
//
// The sweep is the only expensive part of building a decoder parity-check
// matrix, and it depends on nothing but the four arguments. Exposing it lets a
// consumer precompute the sequence once and rebuild the systematic matrix from
// the sparse Gallager source alone, which is cheap.
func PivotSweep(wc, wr, capacity int, variant wire.Variant) []uint16 {
	pivots, _ := messageMatrix(variant, wc, wr, capacity).pivotSweep()
	return pivots
}

// LeadingPivot is the pivot column of parity row i in the leading Gallager
// block, where the block holds capacity/wr rows of wr consecutive ones.
//
// Those rows have pairwise disjoint support, so none of them is ever modified
// before the sweep reaches it and each pivots at its own first column. A
// transcript therefore stores only the rows the permuted copies contribute.
func LeadingPivot(i, wr int) int { return i * wr }
