package jabcode

import "image"

// Mask evaluation weights and pattern count (mask.c, jabcode.h).
const (
	maskW1               = 100
	maskW2               = 3
	maskW3               = 3
	numberOfMaskPatterns = 8
)

// codeParams holds geometry needed to mask and render a code (jab_code). For a
// single symbol the code size equals the symbol side size.
type codeParams struct {
	dimension int
	codeSize  image.Point
}

// getCodePara computes the code parameters for the (single) symbol.
func (e *Encoder) getCodePara() codeParams {
	return codeParams{dimension: e.moduleSize, codeSize: e.symbols[0].sideSize}
}

// maskSymbolsToBuffer applies mask pattern maskType to the data modules and
// returns the whole code as an int buffer (non-code cells are -1), used for
// penalty evaluation (maskSymbols with a target buffer, mask.c).
func (e *Encoder) maskSymbolsToBuffer(maskType int, cp codeParams) []int {
	masked := make([]int, cp.codeSize.X*cp.codeSize.Y)
	for i := range masked {
		masked[i] = -1
	}
	s := &e.symbols[0]
	w, h := s.sideSize.X, s.sideSize.Y
	for y := range h {
		for x := range w {
			idx := int(s.matrix[y*w+x])
			if s.dataMap[y*w+x] != 0 {
				idx ^= maskValue(maskType, x, y) % e.colors
			}
			masked[y*cp.codeSize.X+x] = idx
		}
	}
	return masked
}

// maskCode evaluates all mask patterns, applies the lowest-penalty one in place,
// and returns its reference (maskCode in mask.c).
func (e *Encoder) maskCode(cp codeParams) int {
	maskType := 0
	minPenalty := 10000
	for t := range numberOfMaskPatterns {
		masked := e.maskSymbolsToBuffer(t, cp)
		if p := evaluateMask(masked, cp.codeSize.X, cp.codeSize.Y, e.colors); p < minPenalty {
			maskType = t
			minPenalty = p
		}
	}
	e.maskSymbol(0, maskType)
	return maskType
}

// evaluateMask sums the three masking penalty rules (evaluateMask in mask.c).
func evaluateMask(matrix []int, width, height, colorNumber int) int {
	return applyRule1(matrix, width, height, colorNumber) +
		applyRule2(matrix, width, height) +
		applyRule3(matrix, width, height)
}

// applyRule1 penalizes finder-pattern-like 5-module runs through any cell
// (applyRule1 in mask.c).
func applyRule1(matrix []int, width, height, colorNumber int) int {
	var c1, c2 [4]int
	switch colorNumber {
	case 2:
		c1 = [4]int{0, 1, 1, 1}
		c2 = [4]int{1, 0, 0, 0}
	case 4:
		c1 = [4]int{0, 1, 2, 3}
		c2 = [4]int{3, 2, 1, 0}
	default:
		c1 = [4]int{fp0CoreColor, fp1CoreColor, fp2CoreColor, fp3CoreColor}
		for i := range c2 {
			c2[i] = 7 - c1[i]
		}
	}
	at := func(x, y int) int { return matrix[y*width+x] }
	score := 0
	for i := range height {
		for j := range width {
			if j < 2 || j > width-3 || i < 2 || i > height-3 {
				continue
			}
			for fp := range 4 {
				a, b := c1[fp], c2[fp]
				if at(j-2, i) == a && at(j-1, i) == b && at(j, i) == a && at(j+1, i) == b && at(j+2, i) == a &&
					at(j, i-2) == a && at(j, i-1) == b && at(j, i) == a && at(j, i+1) == b && at(j, i+2) == a {
					score++
					break
				}
			}
		}
	}
	return maskW1 * score
}

// applyRule2 penalizes 2x2 blocks of one color (applyRule2 in mask.c).
func applyRule2(matrix []int, width, height int) int {
	score := 0
	for i := 0; i < height-1; i++ {
		for j := 0; j < width-1; j++ {
			a := matrix[i*width+j]
			b := matrix[i*width+(j+1)]
			c := matrix[(i+1)*width+j]
			d := matrix[(i+1)*width+(j+1)]
			if a != -1 && b != -1 && c != -1 && d != -1 && a == b && a == c && a == d {
				score++
			}
		}
	}
	return maskW2 * score
}

// applyRule3 penalizes long same-color runs in rows and columns (applyRule3).
func applyRule3(matrix []int, width, height int) int {
	score := 0
	for k := range 2 {
		maxi, maxj := height, width
		if k == 1 {
			maxi, maxj = width, height
		}
		for i := 0; i < maxi; i++ {
			run := 0
			prev := -1
			for j := 0; j < maxj; j++ {
				cur := matrix[i*width+j]
				if k == 1 {
					cur = matrix[j*width+i]
				}
				if cur != -1 {
					if cur == prev {
						run++
					} else {
						if run >= 5 {
							score += maskW3 + (run - 5)
						}
						run = 1
						prev = cur
					}
				} else {
					if run >= 5 {
						score += maskW3 + (run - 5)
					}
					run = 0
					prev = -1
				}
			}
			if run >= 5 {
				score += maskW3 + (run - 5)
			}
		}
	}
	return score
}
