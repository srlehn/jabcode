package detect

import (
	"math"
	"slices"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/spec"
)

// Finder-pattern types, in the order ISO/IEC 23634:2022 4.3.7 lists them.
//
// A JAB finder is NOT the concentric ring pattern a QR finder is, and assuming
// it is has repeatedly produced wrong reasoning about rotation. Per 4.3.7 and
// its Figures 5 and 6, each finder is two equal 3x3 square references joined at
// a single overlapping module, the core, laid out along a *diagonal*:
//
//	fp0  UL  black outer, cyan inner,  black core   | on one diagonal
//	fp1  UR  black outer, yellow inner, black core  |
//	fp2  LR  yellow outer, black inner, yellow core | on the other diagonal
//	fp3  LL  cyan outer, black inner, cyan core     |
//
// The consequence that matters for any orientation work: the shape's point
// symmetry is order 2, a half turn about the core exchanging the two references,
// plus a reflection in the joining diagonal. It has no quarter-turn symmetry. A
// quarter turn carries the {UL,UR} diagonal onto the {LR,LL} diagonal, so it
// permutes finder types rather than fixing them - a rotated UL presents the
// geometry of the LR/LL class while keeping UL's colours, and only the colours
// still identify it.
//
// So a scan direction folds modulo 180 degrees, never modulo 90, and a
// quarter-turn search rung carries real detection evidence rather than only a
// corner relabelling.
const (
	fp0 = 0
	fp1 = 1
	fp2 = 2
	fp3 = 3
)

// FinderPattern is a detected finder or alignment pattern.
type FinderPattern struct {
	Typ        int
	ModuleSize float64
	Center     core.PointF
	FoundCount int
	direction  int
}

// crossCheckPatternDiagonal validates a finder-pattern candidate along a
// diagonal and refines its center, returning the number of confirmed diagonals.
func crossCheckPatternDiagonal(image *core.Bitmap, typ int, moduleSizeMax float64, centerx, centery, moduleSize *float64, dir *int, bothDir bool, slack int) int {
	// Ports crossCheckPatternDiagonal in detector.c.
	const stateMiddle = 2
	var offsetX, offsetY int
	fixDir := false
	switch {
	case *dir != 0:
		if *dir > 0 {
			offsetX, offsetY, *dir = -1, -1, 1
		} else {
			offsetX, offsetY, *dir = 1, -1, -1
		}
		fixDir = true
	case typ == fp0 || typ == fp1:
		offsetX, offsetY, *dir = -1, -1, 1
	default:
		offsetX, offsetY, *dir = 1, -1, -1
	}

	confirmed := 0
	tryCount := 0
	tmpModuleSize := 0.0
	for {
		flag := false
		tryCount++
		var i, stateIndex int
		var stateCount [5]int
		startx := int(*centerx)
		starty := int(*centery)

		stateCount[stateMiddle]++
		for j := 1; starty+j*offsetY >= 0 && starty+j*offsetY < image.Height && startx+j*offsetX >= 0 && startx+j*offsetX < image.Width && stateIndex <= stateMiddle; j++ {
			if image.Pix[(starty+j*offsetY)*image.Width+(startx+j*offsetX)] == image.Pix[(starty+(j-1)*offsetY)*image.Width+(startx+(j-1)*offsetX)] {
				stateCount[stateMiddle-stateIndex]++
			} else if stateIndex > 0 && stateCount[stateMiddle-stateIndex] < slack {
				stateCount[stateMiddle-(stateIndex-1)] += stateCount[stateMiddle-stateIndex]
				stateCount[stateMiddle-stateIndex] = 0
				stateIndex--
				stateCount[stateMiddle-stateIndex]++
			} else {
				stateIndex++
				if stateIndex > stateMiddle {
					break
				}
				stateCount[stateMiddle-stateIndex]++
			}
		}
		if stateIndex < stateMiddle {
			if tryCount == 1 {
				flag = true
				offsetX = -offsetX
				*dir = -(*dir)
			} else {
				return confirmed
			}
		}

		if !flag {
			stateIndex = 0
			for i = 1; starty-i*offsetY >= 0 && starty-i*offsetY < image.Height && startx-i*offsetX >= 0 && startx-i*offsetX < image.Width && stateIndex <= stateMiddle; i++ {
				if image.Pix[(starty-i*offsetY)*image.Width+(startx-i*offsetX)] == image.Pix[(starty-(i-1)*offsetY)*image.Width+(startx-(i-1)*offsetX)] {
					stateCount[stateMiddle+stateIndex]++
				} else if stateIndex > 0 && stateCount[stateMiddle+stateIndex] < slack {
					stateCount[stateMiddle+(stateIndex-1)] += stateCount[stateMiddle+stateIndex]
					stateCount[stateMiddle+stateIndex] = 0
					stateIndex--
					stateCount[stateMiddle+stateIndex]++
				} else {
					stateIndex++
					if stateIndex > stateMiddle {
						break
					}
					stateCount[stateMiddle+stateIndex]++
				}
			}
			if stateIndex < stateMiddle {
				if tryCount == 1 {
					flag = true
					offsetX = -offsetX
					*dir = -(*dir)
				} else {
					return confirmed
				}
			}
		}

		if !flag {
			ms, ret := checkPatternCross(stateCount)
			*moduleSize = ms
			if ret && *moduleSize <= moduleSizeMax {
				if tmpModuleSize > 0 {
					*moduleSize = (*moduleSize + tmpModuleSize) / 2.0
				} else {
					tmpModuleSize = *moduleSize
				}
				*centerx = float64(startx+i-stateCount[4]-stateCount[3]) - float64(stateCount[2])/2.0
				*centery = float64(starty+i-stateCount[4]-stateCount[3]) - float64(stateCount[2])/2.0
				confirmed++
				if !bothDir || tryCount == 2 || fixDir {
					if confirmed == 2 {
						*dir = 2
					}
					return confirmed
				}
			} else {
				offsetX = -offsetX
				*dir = -(*dir)
			}
		}
		if !(tryCount < 2 && !fixDir) {
			break
		}
	}
	return confirmed
}

// crossCheckPatternVertical validates and refines a candidate along the
// vertical. As with the horizontal walk, centery and moduleSize are outputs of
// a positive verdict only.
func crossCheckPatternVertical(image *core.Bitmap, moduleSizeMax int, centerx float64, centery, moduleSize *float64, slack int) bool {
	// Ports crossCheckPatternVertical in detector.c.
	const stateMiddle = 2
	var stateCount [5]int
	cx := int(centerx)
	cy := int(*centery)

	// The module-size bound of the horizontal walk, which holds here for the
	// same reason: the three middle state counts only ever grow, so once their
	// third passes moduleSizeMax no continuation brings the candidate back
	// into range. This walk strides a row per step, so the steps it drops are
	// the expensive kind.
	insideLimit := image.Height + 1
	if lim := float64(moduleSizeMax) * 3; lim >= 0 && lim < float64(image.Height) {
		insideLimit = int(lim) + 2
	}
	inside := 1

	var i, stateIndex int
	stateCount[1]++
	for i = 1; i <= cy && stateIndex <= stateMiddle; i++ {
		if image.Pix[(cy-i)*image.Width+cx] == image.Pix[(cy-(i-1))*image.Width+cx] {
			stateCount[stateMiddle-stateIndex]++
			if stateMiddle-stateIndex > 0 {
				if inside++; inside >= insideLimit {
					return false
				}
			}
		} else if stateIndex > 0 && stateCount[stateMiddle-stateIndex] < slack {
			stateCount[stateMiddle-(stateIndex-1)] += stateCount[stateMiddle-stateIndex]
			stateCount[stateMiddle-stateIndex] = 0
			stateIndex--
			stateCount[stateMiddle-stateIndex]++
			if inside = stateCount[1] + stateCount[2] + stateCount[3]; inside >= insideLimit {
				return false
			}
		} else {
			stateIndex++
			if stateIndex > stateMiddle {
				break
			}
			stateCount[stateMiddle-stateIndex]++
			if stateMiddle-stateIndex > 0 {
				if inside++; inside >= insideLimit {
					return false
				}
			}
		}
	}
	if stateIndex < stateMiddle {
		return false
	}
	stateIndex = 0
	for i = 1; cy+i < image.Height && stateIndex <= stateMiddle; i++ {
		if image.Pix[(cy+i)*image.Width+cx] == image.Pix[(cy+(i-1))*image.Width+cx] {
			stateCount[stateMiddle+stateIndex]++
			if stateMiddle+stateIndex < 4 {
				if inside++; inside >= insideLimit {
					return false
				}
			}
		} else if stateIndex > 0 && stateCount[stateMiddle+stateIndex] < slack {
			stateCount[stateMiddle+(stateIndex-1)] += stateCount[stateMiddle+stateIndex]
			stateCount[stateMiddle+stateIndex] = 0
			stateIndex--
			stateCount[stateMiddle+stateIndex]++
			if inside = stateCount[1] + stateCount[2] + stateCount[3]; inside >= insideLimit {
				return false
			}
		} else {
			stateIndex++
			if stateIndex > stateMiddle {
				break
			}
			stateCount[stateMiddle+stateIndex]++
			if stateMiddle+stateIndex < 4 {
				if inside++; inside >= insideLimit {
					return false
				}
			}
		}
	}
	if stateIndex < stateMiddle {
		return false
	}
	ms, ret := checkPatternCross(stateCount)
	*moduleSize = ms
	if ret && *moduleSize <= float64(moduleSizeMax) {
		*centery = float64(cy+i-stateCount[4]-stateCount[3]) - float64(stateCount[2])/2.0
		return true
	}
	return false
}

// crossCheckPatternHorizontal validates and refines a candidate along the
// horizontal. centerx and moduleSize are outputs of a positive verdict only: a
// rejected candidate leaves both as the caller passed them, because the walk
// stops as soon as rejection is certain rather than deriving values nothing
// reads.
func crossCheckPatternHorizontal(image *core.Bitmap, moduleSizeMax float64, centerx *float64, centery float64, moduleSize *float64, slack int) bool {
	// Ports crossCheckPatternHorizontal in detector.c.
	const stateMiddle = 2
	var stateCount [5]int
	startx := int(*centerx)
	rowOffset := int(centery) * image.Width

	rowEnd := min(rowOffset+image.Width, len(image.Pix))

	// An accepted candidate's module size is one third of the three middle
	// state counts, so it is out of range as soon as that third passes
	// moduleSizeMax. Those three counts never shrink: a slack merge only moves
	// a count inward, into a state that is also one of the three. Their sum is
	// therefore monotone, and once it crosses the bound no continuation of the
	// walk can bring it back. Stopping there is what keeps a run of background
	// from being measured to its far end before the ratio test rejects it.
	insideLimit := image.Width + 1
	if lim := moduleSizeMax * 3; lim >= 0 && lim < float64(image.Width) {
		// Two above the truncated bound, so reaching the limit exceeds it by
		// more than any rounding in the module-size division can recover.
		insideLimit = int(lim) + 2
	}
	inside := 1

	var i, stateIndex int
	stateCount[stateMiddle]++
	// Each step compares a pixel with the one the previous step already read,
	// so carrying it forward halves the loads on the hottest walk in detection.
	// Inside a run the step only accumulates a length, so whole runs are
	// measured at once instead of branching per pixel. A run feeding one of
	// the three middle states is measured no further than its remaining
	// budget, since anything past that rejects the candidate outright.
	prev := image.Pix[rowOffset+startx]
	for i = 1; i <= startx && stateIndex <= stateMiddle; i++ {
		cur := image.Pix[rowOffset+(startx-i)]
		if cur == prev {
			state := stateMiddle - stateIndex
			tail := image.Pix[rowOffset : rowOffset+startx-i]
			if state > 0 {
				if budget := insideLimit - inside; len(tail) >= budget {
					tail = tail[len(tail)-budget+1:]
				}
			}
			run := 1 + trailingEqual(tail, prev)
			stateCount[state] += run
			if state > 0 {
				if inside += run; inside >= insideLimit {
					return false
				}
			}
			i += run - 1
			continue
		}
		if stateIndex > 0 && stateCount[stateMiddle-stateIndex] < slack {
			stateCount[stateMiddle-(stateIndex-1)] += stateCount[stateMiddle-stateIndex]
			stateCount[stateMiddle-stateIndex] = 0
			stateIndex--
			stateCount[stateMiddle-stateIndex]++
			if inside = stateCount[1] + stateCount[2] + stateCount[3]; inside >= insideLimit {
				return false
			}
		} else {
			stateIndex++
			if stateIndex > stateMiddle {
				break
			}
			stateCount[stateMiddle-stateIndex]++
			if stateMiddle-stateIndex > 0 {
				if inside++; inside >= insideLimit {
					return false
				}
			}
		}
		prev = cur
	}
	if stateIndex < stateMiddle {
		return false
	}
	stateIndex = 0
	prev = image.Pix[rowOffset+startx]
	for i = 1; startx+i < image.Width && stateIndex <= stateMiddle; i++ {
		cur := image.Pix[rowOffset+(startx+i)]
		if cur == prev {
			state := stateMiddle + stateIndex
			head := image.Pix[min(rowOffset+startx+i+1, rowEnd):rowEnd]
			if state < 4 {
				if budget := insideLimit - inside; len(head) >= budget {
					head = head[:budget-1]
				}
			}
			run := 1 + leadingEqual(head, prev)
			stateCount[state] += run
			if state < 4 {
				if inside += run; inside >= insideLimit {
					return false
				}
			}
			i += run - 1
			continue
		}
		if stateIndex > 0 && stateCount[stateMiddle+stateIndex] < slack {
			stateCount[stateMiddle+(stateIndex-1)] += stateCount[stateMiddle+stateIndex]
			stateCount[stateMiddle+stateIndex] = 0
			stateIndex--
			stateCount[stateMiddle+stateIndex]++
			if inside = stateCount[1] + stateCount[2] + stateCount[3]; inside >= insideLimit {
				return false
			}
		} else {
			stateIndex++
			if stateIndex > stateMiddle {
				break
			}
			stateCount[stateMiddle+stateIndex]++
			if stateMiddle+stateIndex < 4 {
				if inside++; inside >= insideLimit {
					return false
				}
			}
		}
		prev = cur
	}
	if stateIndex < stateMiddle {
		return false
	}
	ms, ret := checkPatternCross(stateCount)
	*moduleSize = ms
	if ret && *moduleSize <= moduleSizeMax {
		*centerx = float64(startx+i-stateCount[4]-stateCount[3]) - float64(stateCount[2])/2.0
		return true
	}
	return false
}

// crossCheckColor verifies the finder-pattern core has the expected color along
// a direction (0:horizontal, 1:vertical, 2:diagonal).
func crossCheckColor(image *core.Bitmap, color, moduleSize, moduleNumber, centerx, centery, dir, tol int) bool {
	// Ports crossCheckColor in detector.c.
	// A candidate whose refined center left the image has no evidence to
	// verify: the walks below index row-major from that center and would read
	// a neighbouring row, or past the buffer on the last one.
	if centerx < 0 || centerx >= image.Width || centery < 0 || centery >= image.Height {
		return false
	}
	switch dir {
	case 0:
		length := moduleSize * (moduleNumber - 1)
		startx := max(centerx-length/2, 0)
		unmatch := 0
		for j := startx; j < startx+length && j < image.Width; j++ {
			if int(image.Pixel(j, centery)) != color {
				unmatch++
			} else if unmatch <= tol {
				unmatch = 0
			}
			if unmatch > tol {
				return false
			}
		}
		return true
	case 1:
		length := moduleSize * (moduleNumber - 1)
		starty := max(centery-length/2, 0)
		unmatch := 0
		for i := starty; i < starty+length && i < image.Height; i++ {
			if int(image.Pixel(centerx, i)) != color {
				unmatch++
			} else if unmatch <= tol {
				unmatch = 0
			}
			if unmatch > tol {
				return false
			}
		}
		return true
	case 2:
		offset := int(float64(moduleSize) * (float64(moduleNumber) / (2.0 * 1.41421)))
		length := offset * 2
		unmatch := 0
		startx := max(centerx-offset, 0)
		starty := max(centery-offset, 0)
		// Both walks stop at the right edge as well as the bottom one: without
		// the column bound the row-major index runs into the next row and, on
		// the last row, past the buffer entirely.
		for i := 0; i < length && starty+i < image.Height && startx+i < image.Width; i++ {
			if int(image.Pixel(startx+i, starty+i)) != color {
				unmatch++
			} else if unmatch <= tol {
				unmatch = 0
			}
			if unmatch > tol {
				break
			}
		}
		if unmatch < tol {
			return true
		}
		unmatch = 0
		startx = max(centerx-offset, 0)
		starty = min(centery+offset, image.Height-1)
		for i := 0; i < length && starty-i >= 0 && startx+i < image.Width; i++ {
			if int(image.Pix[image.Width*(starty-i)+(startx+i)]) != color {
				unmatch++
			} else if unmatch <= tol {
				unmatch = 0
			}
			if unmatch > tol {
				return false
			}
		}
		return true
	}
	return false
}

// crossCheckPatternCh validates a candidate in a single channel across vertical,
// horizontal and diagonal directions.
func crossCheckPatternCh(ch *core.Bitmap, typ, hv int, moduleSizeMax float64, moduleSize, centerx, centery *float64, dir, dcc *int, slack int) bool {
	// Ports crossCheckPatternCh in detector.c.
	var msV, msH, msD float64
	if hv == 0 {
		vcc := false
		if crossCheckPatternVertical(ch, int(moduleSizeMax), *centerx, centery, &msV, slack) {
			vcc = true
			if !crossCheckPatternHorizontal(ch, moduleSizeMax, centerx, *centery, &msH, slack) {
				return false
			}
		}
		*dcc = crossCheckPatternDiagonal(ch, typ, moduleSizeMax, centerx, centery, &msD, dir, !vcc, slack)
		switch {
		case vcc && *dcc > 0:
			*moduleSize = (msV + msH + msD) / 3.0
			return true
		case *dcc == 2:
			if !crossCheckPatternHorizontal(ch, moduleSizeMax, centerx, *centery, &msH, slack) {
				return false
			}
			*moduleSize = (msH + msD*2.0) / 3.0
			return true
		}
	} else {
		hcc := false
		if crossCheckPatternHorizontal(ch, moduleSizeMax, centerx, *centery, &msH, slack) {
			hcc = true
			if !crossCheckPatternVertical(ch, int(moduleSizeMax), *centerx, centery, &msV, slack) {
				return false
			}
		}
		*dcc = crossCheckPatternDiagonal(ch, typ, moduleSizeMax, centerx, centery, &msD, dir, !hcc, slack)
		switch {
		case hcc && *dcc > 0:
			*moduleSize = (msV + msH + msD) / 3.0
			return true
		case *dcc == 2:
			if !crossCheckPatternVertical(ch, int(moduleSizeMax), *centerx, centery, &msV, slack) {
				return false
			}
			*moduleSize = (msV + msD*2.0) / 3.0
			return true
		}
	}
	return false
}

// crossCheckPattern validates a finder-pattern candidate across the relevant
// color channels and refines its center, module size and direction. hv is 0 for
// a horizontal candidate, 1 for vertical.
func crossCheckPattern(ch [3]*core.Bitmap, fp *FinderPattern, hv int, slack int) bool {
	// Ports crossCheckPattern in detector.c.
	moduleSizeMax := fp.ModuleSize * 2.0

	var msG float64
	cxG, cyG := fp.Center.X, fp.Center.Y
	dirG, dccG := 0, 0
	if !crossCheckPatternCh(ch[1], fp.Typ, hv, moduleSizeMax, &msG, &cxG, &cyG, &dirG, &dccG, slack) {
		return false
	}

	if fp.Typ == fp1 || fp.Typ == fp2 {
		var msR float64
		cxR, cyR := fp.Center.X, fp.Center.Y
		dirR, dccR := 0, 0
		if !crossCheckPatternCh(ch[0], fp.Typ, hv, moduleSizeMax, &msR, &cxR, &cyR, &dirR, &dccR, slack) {
			return false
		}
		if !checkModuleSize2(msR, msG) {
			return false
		}
		fp.ModuleSize = (msR + msG) / 2.0
		fp.Center.X = (cxR + cxG) / 2.0
		fp.Center.Y = (cyR + cyG) / 2.0
		coreBlue := int(palette.Default[spec.FP2CoreColor*3+2])
		for d := range 3 {
			if !crossCheckColor(ch[2], coreBlue, int(fp.ModuleSize), 5, int(fp.Center.X), int(fp.Center.Y), d, slack) {
				return false
			}
		}
		switch {
		case dccR == 2 || dccG == 2:
			fp.direction = 2
		case dirR+dirG > 0:
			fp.direction = 1
		default:
			fp.direction = -1
		}
	}

	if fp.Typ == fp0 || fp.Typ == fp3 {
		var msB float64
		cxB, cyB := fp.Center.X, fp.Center.Y
		dirB, dccB := 0, 0
		if !crossCheckPatternCh(ch[2], fp.Typ, hv, moduleSizeMax, &msB, &cxB, &cyB, &dirB, &dccB, slack) {
			return false
		}
		if !checkModuleSize2(msG, msB) {
			return false
		}
		fp.ModuleSize = (msG + msB) / 2.0
		fp.Center.X = (cxG + cxB) / 2.0
		fp.Center.Y = (cyG + cyB) / 2.0
		coreRed := int(palette.Default[spec.FP3CoreColor*3+0])
		for d := range 3 {
			if !crossCheckColor(ch[0], coreRed, int(fp.ModuleSize), 5, int(fp.Center.X), int(fp.Center.Y), d, slack) {
				return false
			}
		}
		switch {
		case dccG == 2 || dccB == 2:
			fp.direction = 2
		case dirG+dirB > 0:
			fp.direction = 1
		default:
			fp.direction = -1
		}
	}
	return true
}

// saveFinderPattern merges a candidate into the list, averaging with an existing
// nearby pattern of the same type or appending it.
func saveFinderPattern(fp *FinderPattern, fps []FinderPattern, counter *int, fpTypeCount []int) {
	// Ports saveFinderPattern in detector.c.
	for i := 0; i < *counter; i++ {
		if fps[i].FoundCount > 0 &&
			math.Abs(fp.Center.X-fps[i].Center.X) <= fp.ModuleSize && math.Abs(fp.Center.Y-fps[i].Center.Y) <= fp.ModuleSize &&
			(math.Abs(fp.ModuleSize-fps[i].ModuleSize) <= fps[i].ModuleSize || math.Abs(fp.ModuleSize-fps[i].ModuleSize) <= 1.0) &&
			fp.Typ == fps[i].Typ {
			fc := float64(fps[i].FoundCount)
			fps[i].Center.X = (fc*fps[i].Center.X + fp.Center.X) / (fc + 1)
			fps[i].Center.Y = (fc*fps[i].Center.Y + fp.Center.Y) / (fc + 1)
			fps[i].ModuleSize = (fc*fps[i].ModuleSize + fp.ModuleSize) / (fc + 1)
			fps[i].FoundCount++
			fps[i].direction += fp.direction
			return
		}
	}
	fps[*counter] = *fp
	*counter++
	fpTypeCount[fp.Typ]++
}

// removeBadPatterns zeroes patterns whose module size deviates too far from the
// mean, recovering the closest one if all were removed.
func removeBadPatterns(fps []FinderPattern, fpCount int, mean, threshold float64) {
	// Ports removeBadPatterns in detector.c.
	removeCount := 0
	backup := make([]int, fpCount)
	for i := range fpCount {
		if fps[i].FoundCount < 2 || math.Abs(fps[i].ModuleSize-mean) > threshold {
			backup[i] = fps[i].FoundCount
			fps[i].FoundCount = 0
			removeCount++
		}
	}
	if removeCount == fpCount {
		minDiff := (threshold + mean) * 100
		minIndex := 0
		for i := range fpCount {
			if diff := math.Abs(fps[i].ModuleSize - mean); diff < minDiff {
				minDiff = diff
				minIndex = i
			}
		}
		fps[minIndex].FoundCount = backup[minIndex]
	}
}

// bestPattern returns the most-frequently-detected pattern (ties broken by
// closeness to the mean module size) and clears it from the list.
func bestPattern(fps []FinderPattern, fpCount int) FinderPattern {
	// Ports getBestPattern in detector.c.
	counter := 0
	total := 0.0
	for i := range fpCount {
		if fps[i].FoundCount > 0 {
			counter++
			total += fps[i].ModuleSize
		}
	}
	mean := total / float64(counter)

	maxFound := 0
	minDiff := 100.0
	maxIndex := 0
	for i := range fpCount {
		if fps[i].FoundCount == 0 {
			continue
		}
		if fps[i].FoundCount > maxFound {
			maxFound = fps[i].FoundCount
			maxIndex = i
			minDiff = math.Abs(fps[i].ModuleSize - mean)
		} else if fps[i].FoundCount == maxFound && math.Abs(fps[i].ModuleSize-mean) < minDiff {
			maxIndex = i
			minDiff = math.Abs(fps[i].ModuleSize - mean)
		}
	}
	fp := fps[maxIndex]
	fps[maxIndex].FoundCount = 0
	return fp
}

// selectBestPatternsFor reduces the candidate list to the single best pattern of
// each of the four types in fps[0..3], returning how many types are missing and
// recording the pre-prune group sizes and the post-prune selection in this scan
// direction's stats. fpTypeCount is unused here, kept to mirror the C signature.
// pre, when non-nil, receives each type's best pattern as it stood before the
// outvoted-type prune. What that prune removes cannot be recovered afterwards,
// and "a true corner was deleted for being rarer than a background blob" is
// indistinguishable from "the corner was never found" without it.
// minFinderCrossings is how many scan lines must re-find a pattern before the
// selection will group it: a module spans at least three pixels, so a genuine
// finder is crossed at least three times. Anything under it is a single stray
// crossing, which is also why nothing under it may replace an interpolated
// corner.
const minFinderCrossings = 3

func (d *PrimaryDetector) selectBestPatternsFor(
	fps []FinderPattern,
	fpCount int,
	fpTypeCount []int,
	contextualTypes [4]bool,
	st *FinderFamilyScanStats,
	pre *[4]FinderPattern,
) int {
	// Ports selectBestPatterns in detector.c.
	var groups [4][]FinderPattern
	for i := range fpCount {
		if fps[i].FoundCount < minFinderCrossings {
			continue
		}
		if t := fps[i].Typ; t >= 0 && t < 4 {
			groups[t] = append(groups[t], fps[i])
		}
	}
	for t := range 4 {
		st.Preprune[t] = len(groups[t])
	}
	for t := range 4 {
		switch len(groups[t]) {
		case 0:
			fps[t] = FinderPattern{}
		case 1:
			fps[t] = groups[t][0]
		default:
			fps[t] = bestPattern(groups[t], len(groups[t]))
		}
	}

	maxFound := 0
	for i := range 4 {
		st.Preselect[i] = fps[i].FoundCount
		if fps[i].FoundCount > maxFound {
			maxFound = fps[i].FoundCount
		}
	}
	if pre != nil {
		copy(pre[:], fps[:4])
	}
	// The outvoted-type prune treats a rarely-confirmed type as noise. The
	// print-level passes skip it: colorant-plane misregistration degrades the
	// corners asymmetrically, and a candidate there has already survived the
	// widened multi-channel cross-checks.
	//
	// Where all four types were found, it prunes weakest first and stops while
	// the selection is still recoverable. One absent type is interpolated from
	// the other three; a second is fatal, so a prune that removes two has
	// discarded the direction rather than cleaned it up. That is not
	// hypothetical: on an oblique capture the one direction holding all four
	// types lost its true corner this way, and the direction published instead
	// had no such corner at all.
	//
	// Where a type was already absent, that leniency inverts: keeping the weak
	// types only lets an incomplete direction manufacture its fourth corner from
	// background blobs, pass the coarse consistency gate, and stop the sweep
	// before a direction that can actually read the symbol runs. Measured on an
	// oblique capture, direction 0 selects about 0/4/3/11 and publishes a quad
	// built on two blobs plus a construction. So there the outvote prune runs to
	// completion and the direction fails cleanly, which is the correct outcome.
	// A source-qualified contextual group is different: it proves the absent
	// type was repeatedly crossed in this or an earlier scan direction, although
	// its standalone geometry chain failed. The current direction remains
	// one-corner recoverable with that pooled evidence, so pruning a second
	// detected type would discard the triple the contextual completion needs.
	//
	// FoundCount justifies the order and nothing stronger. It counts how many
	// scan lines re-found a pattern, which scales with the pattern's extent
	// along the scan and with how the sweep crosses it, so it ranks candidates
	// but does not measure confidence, and it is not comparable between types.
	if !d.printPass {
		missing := 0
		outvoted := make([]int, 0, 4)
		for i := range 4 {
			switch {
			case fps[i].FoundCount == 0:
				missing++
			case float64(fps[i].FoundCount) < 0.5*float64(maxFound):
				outvoted = append(outvoted, i)
			}
		}
		slices.SortStableFunc(outvoted, func(a, b int) int {
			return fps[a].FoundCount - fps[b].FoundCount
		})
		recoverable := missing == 0
		if missing == 1 {
			for typ := range 4 {
				if fps[typ].FoundCount == 0 && contextualTypes[typ] {
					recoverable = true
					break
				}
			}
		}
		for _, i := range outvoted {
			if recoverable && missing >= 1 {
				break
			}
			fps[i] = FinderPattern{}
			missing++
		}
	}

	missing := 0
	for i := range 4 {
		if fps[i].FoundCount == 0 {
			missing++
		} else {
			st.Selected[i] = fps[i].FoundCount
		}
	}
	st.Missing = missing
	return missing
}
