package ecc

import (
	"iter"
	"math"
	"math/bits"

	"github.com/srlehn/jabcode/internal/wire"
)

// LDPC error-correction seeds (ldpc.h).
const (
	ldpcMetadataSeed = 38545
	ldpcMessageSeed  = 785465
)

// ceilF32 returns ceil(x) as an int. The argument is computed in float32 so the
// rounding matches the reference's jab_float (float) arithmetic.
func ceilF32(x float32) int { return int(math.Ceil(float64(x))) }

// parityCheckRows returns the number of parity-check rows for a code of the
// given width and column/row weights (the nb_pcb quantity in ldpc.c).
func parityCheckRows(wc, wr, capacity int) int {
	if wr < 4 {
		return capacity / 2
	}
	return capacity / wr * wc
}

// messageMatrix builds the LDPC parity-check matrix for message data using
// Gallager's construction: a block of consecutive-ones rows followed by wc-1
// pseudo-randomly column-permuted copies of it.
func messageMatrix(profile wire.Profile, wc, wr, capacity int) *bitMatrix {
	// Ports createMatrixA in ldpc.c.
	rows := parityCheckRows(wc, wr, capacity)
	blocks := capacity / wr // rows per consecutive-ones block
	A := newBitMatrix(rows, capacity)

	for i := range blocks {
		for j := range wr {
			A.set(i, i*wr+j)
		}
	}

	perm := newIdentityPerm(capacity)
	next, stop := iter.Pull(randomValues(profile, ldpcMessageSeed))
	defer stop()
	for i := 1; i < wc; i++ {
		dstBlock := i * blocks
		for j := range capacity {
			x, _ := next()
			pos := profileRandIndex(profile, x, capacity-j)
			for k := range blocks {
				if A.get(k, perm[pos]) {
					A.set(dstBlock+k, j)
				}
			}
			perm.swap(capacity-1-j, pos)
		}
	}
	return A
}

// metadataMatrix builds the LDPC parity-check matrix for metadata, used for the
// wr == 0 code rate.
func metadataMatrix(profile wire.Profile, wc, capacity int) *bitMatrix {
	// Ports createMetadataMatrixA in ldpc.c.
	rows := capacity / 2
	A := newBitMatrix(rows, capacity)

	perm := newIdentityPerm(capacity)
	next, stop := iter.Pull(randomValues(profile, ldpcMetadataSeed))
	defer stop()

	onesPerRow := int(float32(capacity*rows)/float32(wc)+3) / rows
	for i := range rows {
		for j := range onesPerRow {
			x, _ := next()
			pos := profileRandIndex(profile, x, capacity-j)
			A.set(i, perm[pos])
			perm.swap(capacity-1-j, pos)
		}
	}
	return A
}

// parityCheckMatrix selects the message or metadata builder (the wr>0 branch in
// the reference encoder/decoder).
func parityCheckMatrix(profile wire.Profile, wc, wr, capacity int) *bitMatrix {
	if wr > 0 {
		return messageMatrix(profile, wc, wr, capacity)
	}
	return metadataMatrix(profile, wc, capacity)
}

// perm is a column permutation used while building Gallager matrices.
type perm []int

func newIdentityPerm(n int) perm {
	p := make(perm, n)
	for i := range p {
		p[i] = i
	}
	return p
}

func (p perm) swap(i, j int) { p[i], p[j] = p[j], p[i] }

// gaussJordan reduces parity-check matrix A to systematic form over GF(2) and
// returns its rank. A is replaced in place by the rearranged systematic matrix.
// encode selects the encoder path (rearrange the reduced matrix) versus the
// decoder path (rearrange the original matrix); the distinction matters because
// the two callers need different but related systematic forms.
func (m *bitMatrix) gaussJordan(encode bool) int {
	// Ports GaussJordan in ldpc.c.
	rows, cols := m.rows, m.cols
	reduced := m.clone()

	arrangement := make([]int, cols)
	processed := make([]bool, cols)
	var zeroLines []int
	swaps := make([][2]int, 0, cols)

	// Forward elimination: for each row find its pivot column and clear that
	// column in all other rows. Pivots beyond the parity-check region are
	// recorded so their columns can be swapped into place afterwards.
	for i := range rows {
		pivot := reduced.firstSetCol(i)
		if pivot < 0 {
			zeroLines = append(zeroLines, i)
			continue
		}
		processed[pivot] = true
		arrangement[pivot] = i
		if pivot >= rows {
			swaps = append(swaps, [2]int{pivot, 0})
		}
		for j := range rows {
			if j != i && reduced.get(j, pivot) {
				reduced.xorRow(j, i)
			}
		}
	}
	rank := rows - len(zeroLines)

	// Move pivots that landed past the rank back into the identity region.
	relocated := 0
	for i := rank; i < rows; i++ {
		if arrangement[i] > 0 {
			for j := range rows {
				if !processed[j] {
					arrangement[j] = arrangement[i]
					processed[j] = true
					processed[i] = false
					swaps = append(swaps, [2]int{i, j})
					arrangement[i] = j
					relocated++
					break
				}
			}
		}
	}

	// Pair the out-of-region pivots recorded above with free columns.
	free := 0
	for c := 0; c < rows && free < len(swaps)-relocated; c++ {
		if !processed[c] {
			arrangement[c] = arrangement[swaps[free][0]]
			processed[c] = true
			swaps[free][1] = c
			free++
		}
	}

	// Assign the remaining columns to the zero (dependent) rows.
	z := 0
	for c := range rows {
		if !processed[c] {
			arrangement[c] = zeroLines[z]
			z++
		}
	}

	// Rearrange rows then apply the recorded column swaps. The encoder works
	// from the reduced matrix; the decoder from the original.
	source := m
	if encode {
		source = reduced
	}
	out := newBitMatrix(rows, cols)
	for i := range rows {
		out.copyRowFrom(i, source, arrangement[i])
	}
	for _, s := range swaps {
		out.swapCols(s[0], s[1])
	}
	*m = *out
	return rank
}

// generatorMatrix derives the systematic generator G = [Cᵀ ; I] from the
// systematic parity-check matrix A, where Pn is the number of net message bits.
func (m *bitMatrix) generatorMatrix(capacity, Pn int) *bitMatrix {
	// Ports createGeneratorMatrix in ldpc.c.
	G := newBitMatrix(capacity, Pn)
	for c := range Pn { // identity block (bottom Pn rows)
		G.set(capacity-Pn+c, c)
	}
	for r := 0; r < capacity-Pn; r++ { // Cᵀ block (top rows)
		for c := range Pn {
			if m.get(r, capacity-Pn+c) {
				G.set(r, c)
			}
		}
	}
	return G
}

// subBlockCount splits a gross length into encoding sub-blocks, each below the
// 2700-bit working limit the reference uses to bound matrix sizes.
func subBlockCount(Pg int) int {
	for i := 1; i < 10000; i++ {
		if Pg/i < 2700 {
			return i
		}
	}
	return 0
}

// EncodeLDPC applies systematic LDPC encoding to a message of bit-per-byte
// values, returning the gross (parity followed by message) codeword, also as
// bit-per-byte values. wc and wr are the column and row weights of the
// parity-check matrix. Large messages are split into sub-blocks, exactly as the
// reference encoder does.
func EncodeLDPC(data []byte, wc, wr int) []byte {
	return EncodeLDPCProfile(data, wc, wr, wire.CReference)
}

// EncodeLDPCProfile is EncodeLDPC under the selected wire-format profile.
func EncodeLDPCProfile(data []byte, wc, wr int, profile wire.Profile) []byte {
	// Ports encodeLDPC in ldpc.c.
	Pn := len(data)
	var Pg int // gross length
	if wr > 0 {
		Pg = ceilF32(float32(Pn*wr) / float32(wr-wc))
		Pg = wr * ceilF32(float32(Pg)/float32(wr)) // round up to a multiple of wr
	} else {
		Pg = Pn * 2
	}

	blocks := subBlockCount(Pg)
	var grossSub, netSub int
	if wr > 0 {
		grossSub = ((Pg / blocks) / wr) * wr
		netSub = grossSub * (wr - wc) / wr
	} else {
		grossSub = Pg
		netSub = Pn
	}
	iterations := Pg / grossSub
	blocks = iterations
	if netSub*blocks < Pn {
		iterations--
	}

	ecc := make([]byte, Pg)
	encodeBlocks(ecc, data, wc, wr, grossSub, netSub, iterations, profile)

	if iterations != blocks {
		// Encode the shorter trailing block separately.
		start := iterations * netSub
		base := iterations * grossSub
		grossSub = Pg - iterations*grossSub
		encodeOneBlock(ecc[base:], data[start:], wc, wr, grossSub, profile)
	}
	return ecc
}

// encodeBlocks encodes the first `iterations` equal-size sub-blocks in place.
func encodeBlocks(ecc, data []byte, wc, wr, grossSub, netSub, iterations int, profile wire.Profile) {
	A, rank := systematicParityCheckProfile(wc, wr, grossSub, true, profile)
	G := A.generatorMatrix(grossSub, grossSub-rank)
	for it := range iterations {
		multiplyBlock(ecc[it*grossSub:], data[it*netSub:(it+1)*netSub], G, grossSub)
	}
}

// encodeOneBlock encodes a single sub-block of the given gross size, consuming
// all of msg as the message bits.
func encodeOneBlock(ecc, msg []byte, wc, wr, grossSub int, profile wire.Profile) {
	A, rank := systematicParityCheckProfile(wc, wr, grossSub, true, profile)
	G := A.generatorMatrix(grossSub, grossSub-rank)
	multiplyBlock(ecc, msg, G, grossSub)
}

// multiplyBlock computes ecc[0:grossSub] = G · msg over GF(2), writing one gross
// codeword bit per row of G.
func multiplyBlock(ecc, msg []byte, G *bitMatrix, grossSub int) {
	for i := range grossSub {
		var bit byte
		for j := range msg {
			if G.get(i, j) && msg[j]&1 == 1 {
				bit ^= 1
			}
		}
		ecc[i] = bit
	}
}

// --- Decoding ---

// DecodeLDPCHard decodes a gross LDPC codeword of bit-per-byte values using
// hard-decision decoding and returns the recovered net message. It handles
// sub-blocks the same way as encoding; data is not modified.
//
// A best-effort correction is attempted when a sub-block's syndrome fails; ok
// reports whether every sub-block satisfies its parity checks afterwards.
// When ok is false the message is known to be unreliable - the correction
// gave up, and parsing it can still succeed with a corrupted payload.
func DecodeLDPCHard(data []byte, wc, wr int) (dec []byte, ok bool) {
	return DecodeLDPCHardProfile(data, wc, wr, wire.CReference)
}

// DecodeLDPCHardProfile is DecodeLDPCHard under the selected wire-format
// profile.
func DecodeLDPCHardProfile(data []byte, wc, wr int, profile wire.Profile) (dec []byte, ok bool) {
	// Ports decodeLDPChd in ldpc.c; the post-correction syndrome re-check is
	// an addition (the reference never verifies what decodeMessage produced).
	length := len(data)
	const maxIter = 25

	var Pg, Pn int
	if wr > 3 {
		Pg = wr * (length / wr)
		Pn = Pg * (wr - wc) / wr
	} else {
		Pg = length
		Pn = length / 2
		wc = 2
		if Pn > 36 {
			wc = 3
		}
	}
	decodedLen := Pn
	ok = true

	work := make([]byte, length)
	copy(work, data)

	blocks := subBlockCount(Pg)
	var grossSub, netSub int
	if wr > 3 {
		grossSub = ((Pg / blocks) / wr) * wr
		netSub = grossSub * (wr - wc) / wr
	} else {
		grossSub = Pg
		netSub = Pn
	}
	iterations := Pg / grossSub
	blocks = iterations
	if netSub*blocks < Pn {
		iterations--
	}

	A, rank, idx := systematicParityCheckIndexedProfile(wc, wr, grossSub, profile)
	oldGrossSub, oldNetSub := grossSub, netSub

	for it := 0; it < blocks; it++ {
		if iterations != blocks && it == iterations {
			// Trailing block is shorter: rebuild its parity-check matrix.
			grossSub = Pg - iterations*grossSub
			netSub = grossSub * (wr - wc) / wr
			A, rank, idx = systematicParityCheckIndexedProfile(wc, wr, grossSub, profile)
		}
		start := it * oldGrossSub
		if !syndromeOK(work, A, grossSub, rank, start) {
			decodeMessage(work, idx, grossSub, rank, maxIter, start)
			if !syndromeOK(work, A, grossSub, rank, start) {
				ok = false
			}
		}
		// Compact the systematic message bits to the front of work.
		for loop := 0; loop < netSub; loop++ {
			work[it*oldNetSub+loop] = work[start+rank+loop]
		}
	}
	return work[:decodedLen], ok
}

// syndromeOK reports whether the first rank parity checks of A are satisfied by
// the sub-block at data[startPos:startPos+length]. The sub-block is packed into
// words once so each parity row is popcount(AND) over A's packed rows instead
// of a bit-at-a-time walk; length always equals A.cols at the call sites, so
// the packed widths line up.
func syndromeOK(data []byte, A *bitMatrix, length, rank, startPos int) bool {
	return syndromeWeight(data, A, length, rank, startPos) == 0
}

// syndromeWeight returns how many of the first rank parity checks are
// unsatisfied by one gross sub-block. It is read-only and uses the same packed
// parity walk as syndromeOK.
func syndromeWeight(data []byte, A *bitMatrix, length, rank, startPos int) int {
	packed := make([]uint64, (length+63)/64)
	for j, b := range data[startPos : startPos+length] {
		if b&1 == 1 {
			packed[j/64] |= 1 << (uint(j) % 64)
		}
	}
	unsatisfied := 0
	for i := range rank {
		ones := 0
		for k, w := range A.row(i) {
			ones += bits.OnesCount64(w & packed[k])
		}
		if ones&1 != 0 {
			unsatisfied++
		}
	}
	return unsatisfied
}

// LDPCSyndromeWeight measures a complete gross message codeword without
// correcting or mutating it. It returns the number of unsatisfied independent
// parity checks, the number checked, and whether the code parameters and
// gross length form a legal data codeword.
func LDPCSyndromeWeight(data []byte, wc, wr int) (unsatisfied, checks int, valid bool) {
	if len(data) == 0 || wc < 3 || wc >= wr || wr > 11 || len(data)%wr != 0 {
		return 0, 0, false
	}
	pg := len(data)
	blocks := subBlockCount(pg)
	if blocks <= 0 {
		return 0, 0, false
	}
	grossSub := ((pg / blocks) / wr) * wr
	if grossSub <= 0 {
		return 0, 0, false
	}
	netSub := grossSub * (wr - wc) / wr
	pn := pg * (wr - wc) / wr
	iterations := pg / grossSub
	blocks = iterations
	if netSub*blocks < pn {
		iterations--
	}
	A, rank, _ := systematicParityCheckIndexed(wc, wr, grossSub)
	oldGrossSub := grossSub
	for block := 0; block < blocks; block++ {
		if iterations != blocks && block == iterations {
			grossSub = pg - iterations*grossSub
			A, rank, _ = systematicParityCheckIndexed(wc, wr, grossSub)
		}
		start := block * oldGrossSub
		unsatisfied += syndromeWeight(data, A, grossSub, rank, start)
		checks += rank
	}
	return unsatisfied, checks, true
}

// decodeMessage performs iterative hard-decision bit-flipping correction on the
// sub-block at data[startPos:startPos+length], using the first `height` rows of
// the parity-check matrix, given as its edge adjacency. The reference probes
// every (row, column) bit; walking the set-bit lists computes the same syndrome
// parities and implication counts.
//
// NOTE: for length < 36 the reference breaks ties between equally-likely error
// positions with C rand() (non-deterministic); we deterministically pick the
// first candidate. This affects only the correction of actual errors in very
// short codewords; error-free decoding is identical.
func decodeMessage(data []byte, idx *ldpcIndex, length, height, maxIter, startPos int) {
	// Ports decodeMessage in ldpc.c.
	maxVal := make([]int, length)
	used := make([]bool, length) // bits flipped by the previous iteration
	var prevIndex []int

	for range maxIter {
		for j := range height {
			ones := 0
			for _, i := range idx.rowCols[j] {
				ones += int(data[startPos+int(i)] & 1)
			}
			if ones%2 == 1 { // unsatisfied check: implicate its bits
				for _, k := range idx.rowCols[j] {
					maxVal[k]++
				}
			}
		}

		// Collect the most-implicated bit positions not already flipped.
		best := 0
		var candidates []int
		for j := range length {
			if maxVal[j] >= best && !used[j] {
				if maxVal[j] != best {
					candidates = candidates[:0]
				}
				best = maxVal[j]
				candidates = append(candidates, j)
			}
			maxVal[j] = 0
		}

		if best == 0 {
			break
		}
		for _, j := range prevIndex {
			used[j] = false
		}
		prevIndex = prevIndex[:0]
		if length < 36 {
			j := candidates[0] // deterministic tie-break (see note)
			prevIndex = append(prevIndex, j)
			used[j] = true
			data[startPos+j] ^= 1
		} else {
			for _, j := range candidates {
				prevIndex = append(prevIndex, j)
				used[j] = true
				data[startPos+j] ^= 1
			}
		}
	}
}
