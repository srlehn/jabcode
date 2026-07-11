package ecc

import (
	"math"
	"math/rand"
	"slices"
	"testing"
)

// The iterative decoders were rewritten from dense (row x column) matrix
// probing to edge-adjacency walks. These tests pin the equivalence: the dense
// bodies below are verbatim copies of the pre-rewrite implementations, and
// both variants must produce bit-identical outputs across code families
// (Gallager data codes, metadata codes, sub-36-bit tie-break codes), across
// error patterns (garbage, all-zero, near-valid with few flips), including
// the non-converging paths.

// denseDecodeMessage is the pre-rewrite decodeMessage.
func denseDecodeMessage(data []byte, matrix *bitMatrix, length, height, maxIter, startPos int) {
	maxVal := make([]int, length)
	var prevIndex []int

	for range maxIter {
		for j := range height {
			ones := 0
			for i := range length {
				if matrix.get(j, i) && data[startPos+i]&1 == 1 {
					ones++
				}
			}
			if ones%2 == 1 {
				for k := range length {
					if matrix.get(j, k) {
						maxVal[k]++
					}
				}
			}
		}

		best := 0
		var candidates []int
		for j := range length {
			used := slices.Contains(prevIndex, j)
			if maxVal[j] >= best && !used {
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
		prevIndex = prevIndex[:0]
		if length < 36 {
			idx := candidates[0]
			prevIndex = append(prevIndex, idx)
			data[startPos+idx] ^= 1
		} else {
			for _, idx := range candidates {
				prevIndex = append(prevIndex, idx)
				data[startPos+idx] ^= 1
			}
		}
	}
}

// denseDecodeMessageBP is the pre-rewrite decodeMessageBP.
func denseDecodeMessageBP(enc []float64, matrix *bitMatrix, length, checkbits, height, maxIter int, isCorrect *bool, startPos int, dec []byte) int {
	lambda := make([]float64, length)
	oldNuRow := make([]float64, length)
	nu := make([]float64, length*height)
	index := make([]int, length)

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
			count := 0
			for i := 0; i < length; i++ {
				if matrix.get(j, i) {
					if kl == 0 {
						product *= math.Tanh(lambda[i] * 0.5)
					} else {
						product *= math.Tanh(nu[j*length+i] * 0.5)
					}
					index[count] = i
					count++
				}
			}
			for i := 0; i < count; i++ {
				idx := index[i]
				var num, denum float64
				switch {
				case matrix.get(j, idx) && math.Tanh(nu[j*length+idx]*0.5) != 0.0 && kl > 0:
					t := math.Tanh(nu[j*length+idx] * 0.5)
					num, denum = 1+product/t, 1-product/t
				case matrix.get(j, idx) && math.Tanh(lambda[idx]*0.5) != 0.0 && kl == 0:
					t := math.Tanh(lambda[idx] * 0.5)
					num, denum = 1+product/t, 1-product/t
				default:
					num, denum = 1+product, 1-product
				}
				switch {
				case num == 0.0:
					nu[j*length+idx] = -1
				case denum == 0.0:
					nu[j*length+idx] = 1
				default:
					nu[j*length+idx] = math.Log(num / denum)
				}
			}
		}

		for i := 0; i < length; i++ {
			sum := 0.0
			for k := 0; k < height; k++ {
				sum += nu[k*length+i]
				oldNuRow[k] = nu[k*length+i]
			}
			for k := 0; k < height; k++ {
				if matrix.get(k, i) {
					nu[k*length+i] = lambda[i] + (sum - oldNuRow[k])
				}
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
			for j := 0; j < length; j++ {
				if matrix.get(i, j) && dec[startPos+j]&1 == 1 {
					temp ^= 1
				}
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

// equivCode names one (wc, wr, capacity) family under test. bpHeight mirrors
// decodeLDPC's height choice; hard decoding uses the matrix rank as height,
// mirroring DecodeLDPCHard.
type equivCode struct {
	name             string
	wc, wr, capacity int
}

func (c equivCode) bpHeight() int {
	if c.wr > 3 {
		return c.capacity / c.wr * c.wc
	}
	return c.capacity / 2
}

var equivCodes = []equivCode{
	{"short-metadata", 2, 0, 24}, // length < 36: single-flip tie-break branch
	{"metadata", 3, 0, 60},
	{"data-small", 6, 7, 700},
	{"data-large", 6, 7, 2100},
}

// equivVectors yields deterministic test vectors per code: garbage, all-zero
// (valid codeword of any linear code), and all-zero with 1..8 flips.
func equivVectors(rng *rand.Rand, capacity int) [][]byte {
	var out [][]byte
	garbage := make([]byte, capacity)
	for i := range garbage {
		garbage[i] = byte(rng.Intn(2))
	}
	out = append(out, garbage)
	out = append(out, make([]byte, capacity))
	for _, flips := range []int{1, 3, 8} {
		v := make([]byte, capacity)
		for range flips {
			v[rng.Intn(capacity)] = 1
		}
		out = append(out, v)
	}
	return out
}

func TestDecodeMessageSparseEquivalence(t *testing.T) {
	for _, c := range equivCodes {
		rng := rand.New(rand.NewSource(1))
		A, rank, idx := systematicParityCheckIndexed(c.wc, c.wr, c.capacity)
		for vi, vec := range equivVectors(rng, c.capacity) {
			const maxIter = 25
			dense := slices.Clone(vec)
			sparse := slices.Clone(vec)
			denseDecodeMessage(dense, A, c.capacity, rank, maxIter, 0)
			decodeMessage(sparse, idx, c.capacity, rank, maxIter, 0)
			if !slices.Equal(dense, sparse) {
				t.Errorf("%s vector %d: hard decode diverges", c.name, vi)
			}
		}
	}
}

func TestDecodeMessageBPSparseEquivalence(t *testing.T) {
	for _, c := range equivCodes {
		if c.capacity > 1000 {
			continue // the dense reference allocates length*height floats
		}
		rng := rand.New(rand.NewSource(2))
		A, rank, idx := systematicParityCheckIndexed(c.wc, c.wr, c.capacity)
		height := c.bpHeight()
		for vi, vec := range equivVectors(rng, c.capacity) {
			const maxIter = 25
			rel := make([]float64, c.capacity)
			for i := range rel {
				rel[i] = rng.Float64()
			}
			denseEnc, sparseEnc := slices.Clone(rel), slices.Clone(rel)
			denseDec, sparseDec := slices.Clone(vec), slices.Clone(vec)
			var denseOK, sparseOK bool
			denseDecodeMessageBP(denseEnc, A, c.capacity, rank, height, maxIter, &denseOK, 0, denseDec)
			decodeMessageBP(sparseEnc, idx, c.capacity, rank, height, maxIter, &sparseOK, 0, sparseDec)
			if denseOK != sparseOK {
				t.Errorf("%s vector %d: isCorrect diverges (dense %v, sparse %v)", c.name, vi, denseOK, sparseOK)
			}
			if !slices.Equal(denseDec, sparseDec) {
				t.Errorf("%s vector %d: BP hard decisions diverge", c.name, vi)
			}
			for i := range denseEnc {
				if math.Float64bits(denseEnc[i]) != math.Float64bits(sparseEnc[i]) {
					t.Errorf("%s vector %d: enc[%d] diverges bitwise (dense %x, sparse %x)",
						c.name, vi, i, math.Float64bits(denseEnc[i]), math.Float64bits(sparseEnc[i]))
					break
				}
			}
		}
	}
}
