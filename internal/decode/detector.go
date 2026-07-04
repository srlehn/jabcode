package decode

import "math"

// checkPatternCross checks whether the five run-lengths of a candidate finder-
// pattern scanline follow the expected n-1-1-1-m proportion and returns the
// estimated module size.
func checkPatternCross(stateCount [5]int) (moduleSize float64, ok bool) {
	// Ports checkPatternCross in detector.c.
	inside := 0
	for i := 1; i < 4; i++ {
		if stateCount[i] == 0 {
			return 0, false
		}
		inside += stateCount[i]
	}
	layerSize := float64(inside) / 3.0
	moduleSize = layerSize
	tol := layerSize / 2.0
	ok = math.Abs(layerSize-float64(stateCount[1])) < tol &&
		math.Abs(layerSize-float64(stateCount[2])) < tol &&
		math.Abs(layerSize-float64(stateCount[3])) < tol &&
		float64(stateCount[0]) > 0.5*tol && // the two outer layers may be larger
		float64(stateCount[4]) > 0.5*tol &&
		math.Abs(float64(stateCount[1]-stateCount[3])) < tol // layers 1 and 3 equal
	return moduleSize, ok
}

// checkModuleSize3 reports whether three module-size estimates are consistent.
func checkModuleSize3(r, g, b float64) bool {
	mean := (r + g + b) / 3.0
	tol := mean / 2.5
	return math.Abs(mean-r) < tol && math.Abs(mean-g) < tol && math.Abs(mean-b) < tol
}

// checkModuleSize2 reports whether two module-size estimates are consistent.
func checkModuleSize2(s1, s2 float64) bool {
	mean := (s1 + s2) / 2.0
	tol := mean / 2.5
	return math.Abs(mean-s1) < tol && math.Abs(mean-s2) < tol
}

// patternScan is the result of a finder-pattern scanline search.
type patternScan struct {
	start, end int
	Center     float64
	ModuleSize float64
	skip       int
	ok         bool
}

// scanlinePixel reads pixel p of channel ch along a horizontal (row>=0) or
// vertical (col>=0) scanline.
func scanlinePixel(ch *Bitmap, row, col, p int) byte {
	if row >= 0 {
		return ch.Pix[row*ch.Width+p]
	}
	return ch.Pix[p*ch.Width+col]
}

// seekPattern scans a row (row>=0) or column (col>=0) of a binary channel for a
// finder-pattern cross signature within [start, end) (seekPattern in
// detector.c). The five-state run-length machine mirrors the reference exactly.
func seekPattern(ch *Bitmap, row, col, start, end int) patternScan {
	const stateNumber = 5
	curState := 0
	var stateCount [5]int

	min, max := start, end
	res := patternScan{start: start, end: end}
	for p := min; p < max; p++ {
		if p == min {
			stateCount[curState]++
			res.start = p
			continue
		}
		prev := scanlinePixel(ch, row, col, p-1)
		curr := scanlinePixel(ch, row, col, p)
		if curr == prev {
			stateCount[curState]++
		}
		if curr != prev || p == max-1 {
			if curState < stateNumber-1 {
				if stateCount[curState] < 3 {
					if curState == 0 {
						stateCount[curState] = 1
						res.start = p
					} else {
						stateCount[curState-1] += stateCount[curState]
						stateCount[curState] = 0
						curState--
						stateCount[curState]++
					}
				} else {
					curState++
					stateCount[curState]++
				}
			} else {
				if stateCount[curState] < 3 {
					stateCount[curState-1] += stateCount[curState]
					stateCount[curState] = 0
					curState--
					stateCount[curState]++
					continue
				}
				if ms, ok := checkPatternCross(stateCount); ok {
					res.end = p + 1
					res.skip = stateCount[0]
					res.ModuleSize = ms
					endPos := p
					if p == max-1 && curr == prev {
						endPos = p + 1
					}
					res.Center = float64(endPos-stateCount[4]-stateCount[3]) - float64(stateCount[2])/2.0
					res.ok = true
					return res
				}
				// check failed: shift the state window and keep scanning
				res.start += stateCount[0]
				for k := range stateNumber - 1 {
					stateCount[k] = stateCount[k+1]
				}
				stateCount[stateNumber-1] = 1
				curState = stateNumber - 1
			}
		}
	}
	res.end = max
	return res
}

// seekPatternHorizontal is seekPattern specialized to a single image row slice.
func seekPatternHorizontal(row []byte, start, end int) patternScan {
	// Ports seekPatternHorizontal in detector.c.
	const stateNumber = 5
	curState := 0
	var stateCount [5]int

	min, max := start, end
	res := patternScan{start: start, end: end}
	for j := min; j < max; j++ {
		if j == min {
			stateCount[curState]++
			res.start = j
			continue
		}
		if row[j] == row[j-1] {
			stateCount[curState]++
		}
		if row[j] != row[j-1] || j == max-1 {
			if curState < stateNumber-1 {
				if stateCount[curState] < 3 {
					if curState == 0 {
						stateCount[curState] = 1
						res.start = j
					} else {
						stateCount[curState-1] += stateCount[curState]
						stateCount[curState] = 0
						curState--
						stateCount[curState]++
					}
				} else {
					curState++
					stateCount[curState]++
				}
			} else {
				if stateCount[curState] < 3 {
					stateCount[curState-1] += stateCount[curState]
					stateCount[curState] = 0
					curState--
					stateCount[curState]++
					continue
				}
				if ms, ok := checkPatternCross(stateCount); ok {
					res.end = j + 1
					res.skip = stateCount[0]
					res.ModuleSize = ms
					endPos := j
					if j == max-1 && row[j] == row[j-1] {
						endPos = j + 1
					}
					res.Center = float64(endPos-stateCount[4]-stateCount[3]) - float64(stateCount[2])/2.0
					res.ok = true
					return res
				}
				res.start += stateCount[0]
				for k := range stateNumber - 1 {
					stateCount[k] = stateCount[k+1]
				}
				stateCount[stateNumber-1] = 1
				curState = stateNumber - 1
			}
		}
	}
	res.end = max
	return res
}
