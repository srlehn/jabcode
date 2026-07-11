package ecc

import "math"

// decodeMessageBP performs iterative log-domain belief-propagation decoding of a
// sub-block, refining the hard decisions in dec from the per-bit reliabilities in
// enc. length is the sub-block size, checkbits the matrix rank, height the number
// of parity-check rows; the matrix is given as its edge adjacency.
//
// Messages live only on the edges of the Tanner graph, so they are stored per
// edge (idx.rowOff[row]+slot) instead of in a rows x columns buffer; entries
// off the edges would stay exactly zero, so the variable-node sums over the
// column's edge list equal the reference's sums over every row.
func decodeMessageBP(enc []float64, idx *ldpcIndex, length, checkbits, height, maxIter int, isCorrect *bool, startPos int, dec []byte) int {
	// Ports decodeMessageBP in ldpc.c.
	lambda := make([]float64, length)
	oldNu := make([]float64, idx.maxColDeg)
	nu := make([]float64, idx.rowOff[height])

	// Fix the parity bits that carry no information.
	for i := length - 1; i >= length-(height-checkbits); i-- {
		enc[startPos+i] = 1.0
		dec[startPos+i] = 0
	}
	meansum := 0.0
	for i := 0; i < length; i++ {
		meansum += enc[startPos+i]
	}
	meansum /= float64(length)
	variance := 0.0
	for i := 0; i < length; i++ {
		d := enc[startPos+i] - meansum
		variance += d * d
	}
	variance /= float64(length - 1)
	for i := 0; i < length; i++ {
		if dec[startPos+i] != 0 {
			enc[startPos+i] = -enc[startPos+i]
		}
		lambda[i] = 2.0 * enc[startPos+i] / variance
	}

	for kl := 0; kl < maxIter; kl++ {
		for j := 0; j < height; j++ {
			product := 1.0
			row := idx.rowCols[j]
			base := int(idx.rowOff[j])
			for s, i := range row {
				if kl == 0 {
					product *= math.Tanh(lambda[i] * 0.5)
				} else {
					product *= math.Tanh(nu[base+s] * 0.5)
				}
			}
			for s, i := range row {
				var num, denum float64
				switch {
				case kl > 0 && math.Tanh(nu[base+s]*0.5) != 0.0:
					t := math.Tanh(nu[base+s] * 0.5)
					num, denum = 1+product/t, 1-product/t
				case kl == 0 && math.Tanh(lambda[i]*0.5) != 0.0:
					t := math.Tanh(lambda[i] * 0.5)
					num, denum = 1+product/t, 1-product/t
				default:
					num, denum = 1+product, 1-product
				}
				switch {
				case num == 0.0:
					nu[base+s] = -1
				case denum == 0.0:
					nu[base+s] = 1
				default:
					nu[base+s] = math.Log(num / denum)
				}
			}
		}

		for i := 0; i < length; i++ {
			rows, slots := idx.colRows[i], idx.colSlots[i]
			// The reference sums rows below height only; the adjacency
			// covers the whole matrix, so cut the ascending row list there
			// (height is the full row count at every current call site).
			for len(rows) > 0 && int(rows[len(rows)-1]) >= height {
				rows, slots = rows[:len(rows)-1], slots[:len(slots)-1]
			}
			sum := 0.0
			for e, k := range rows {
				v := nu[int(idx.rowOff[k])+int(slots[e])]
				sum += v
				oldNu[e] = v
			}
			for e, k := range rows {
				nu[int(idx.rowOff[k])+int(slots[e])] = lambda[i] + (sum - oldNu[e])
			}
			lambda[i] = 2.0*enc[startPos+i]/variance + sum
			if lambda[i] < 0 {
				dec[startPos+i] = 1
			} else {
				dec[startPos+i] = 0
			}
		}

		*isCorrect = true
		for i := 0; i < height; i++ {
			temp := 0
			for _, j := range idx.rowCols[i] {
				temp ^= int(dec[startPos+int(j)] & 1)
			}
			if temp != 0 {
				*isCorrect = false
				break
			}
		}
		if !*isCorrect && kl < maxIter-1 {
			*isCorrect = true
		} else {
			break
		}
	}
	return 1
}

// DecodeLDPCSoft runs soft-decision (belief-propagation) decoding of a gross
// codeword: hard holds the initial hard bit decisions (one bit per byte) and rel
// the matching non-negative per-bit reliabilities. It returns the recovered net
// message (written to the front of hard) and whether every sub-block satisfied
// its parity checks afterwards. rel and hard must share the gross length; both
// are modified in place.
//
// The data path calls this only after hard-decision decoding gives up, so a
// clean capture never reaches it.
func DecodeLDPCSoft(rel []float64, hard []byte, wc, wr int) (dec []byte, ok bool) {
	if len(rel) != len(hard) || len(hard) == 0 {
		return nil, false
	}
	n := decodeLDPC(rel, len(hard), wc, wr, hard)
	if n <= 0 || n > len(hard) {
		return nil, false
	}
	return hard[:n], true
}

// decodeLDPC decodes a gross codeword using soft-decision (belief propagation)
// decoding, given per-bit reliabilities enc and initial hard decisions dec, and
// returns the recovered net message length, or 0 when a sub-block cannot be
// satisfied. The decoded message is written to the front of dec.
func decodeLDPC(enc []float64, length, wc, wr int, dec []byte) int {
	// Ports decodeLDPC in ldpc.c.
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

	A, rank, idx := systematicParityCheckIndexed(wc, wr, grossSub)
	oldGrossSub, oldNetSub := grossSub, netSub

	for it := 0; it < blocks; it++ {
		if iterations != blocks && it == iterations {
			grossSub = Pg - iterations*grossSub
			netSub = grossSub * (wr - wc) / wr
			A, rank, idx = systematicParityCheckIndexed(wc, wr, grossSub)
		}
		start := it * oldGrossSub
		if !syndromeOK(dec, A, grossSub, rank, start) {
			height := grossSub / wr * wc
			if wr < 4 {
				height = grossSub / 2
			}
			var isCorrect bool
			decodeMessageBP(enc, idx, grossSub, rank, height, maxIter, &isCorrect, start, dec)
			if !syndromeOK(dec, A, grossSub, rank, start) {
				return 0
			}
		}
		for loop := 0; loop < netSub; loop++ {
			dec[it*oldNetSub+loop] = dec[start+rank+loop]
		}
	}
	return decodedLen
}
