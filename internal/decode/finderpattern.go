package decode

import (
	"math"

	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/spec"
)

// Finder-pattern types.
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
	Center     PointF
	FoundCount int
	direction  int
}

// crossCheckPatternDiagonal validates a finder-pattern candidate along a
// diagonal and refines its center, returning the number of confirmed diagonals.
func crossCheckPatternDiagonal(image *Bitmap, typ int, moduleSizeMax float64, centerx, centery, moduleSize *float64, dir *int, bothDir bool) int {
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
			} else if stateIndex > 0 && stateCount[stateMiddle-stateIndex] < 3 {
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
				} else if stateIndex > 0 && stateCount[stateMiddle+stateIndex] < 3 {
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

// crossCheckPatternVertical validates and refines a candidate along the vertical.
func crossCheckPatternVertical(image *Bitmap, moduleSizeMax int, centerx float64, centery, moduleSize *float64) bool {
	// Ports crossCheckPatternVertical in detector.c.
	const stateMiddle = 2
	var stateCount [5]int
	cx := int(centerx)
	cy := int(*centery)

	var i, stateIndex int
	stateCount[1]++
	for i = 1; i <= cy && stateIndex <= stateMiddle; i++ {
		if image.Pix[(cy-i)*image.Width+cx] == image.Pix[(cy-(i-1))*image.Width+cx] {
			stateCount[stateMiddle-stateIndex]++
		} else if stateIndex > 0 && stateCount[stateMiddle-stateIndex] < 3 {
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
		return false
	}
	stateIndex = 0
	for i = 1; cy+i < image.Height && stateIndex <= stateMiddle; i++ {
		if image.Pix[(cy+i)*image.Width+cx] == image.Pix[(cy+(i-1))*image.Width+cx] {
			stateCount[stateMiddle+stateIndex]++
		} else if stateIndex > 0 && stateCount[stateMiddle+stateIndex] < 3 {
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
// horizontal.
func crossCheckPatternHorizontal(image *Bitmap, moduleSizeMax float64, centerx *float64, centery float64, moduleSize *float64) bool {
	// Ports crossCheckPatternHorizontal in detector.c.
	const stateMiddle = 2
	var stateCount [5]int
	startx := int(*centerx)
	rowOffset := int(centery) * image.Width

	var i, stateIndex int
	stateCount[stateMiddle]++
	for i = 1; i <= startx && stateIndex <= stateMiddle; i++ {
		if image.Pix[rowOffset+(startx-i)] == image.Pix[rowOffset+(startx-(i-1))] {
			stateCount[stateMiddle-stateIndex]++
		} else if stateIndex > 0 && stateCount[stateMiddle-stateIndex] < 3 {
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
		return false
	}
	stateIndex = 0
	for i = 1; startx+i < image.Width && stateIndex <= stateMiddle; i++ {
		if image.Pix[rowOffset+(startx+i)] == image.Pix[rowOffset+(startx+(i-1))] {
			stateCount[stateMiddle+stateIndex]++
		} else if stateIndex > 0 && stateCount[stateMiddle+stateIndex] < 3 {
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
func crossCheckColor(image *Bitmap, color, moduleSize, moduleNumber, centerx, centery, dir int) bool {
	// Ports crossCheckColor in detector.c.
	const tolerance = 3
	switch dir {
	case 0:
		length := moduleSize * (moduleNumber - 1)
		startx := max(centerx-length/2, 0)
		unmatch := 0
		for j := startx; j < startx+length && j < image.Width; j++ {
			if int(image.Pix[centery*image.Width+j]) != color {
				unmatch++
			} else if unmatch <= tolerance {
				unmatch = 0
			}
			if unmatch > tolerance {
				return false
			}
		}
		return true
	case 1:
		length := moduleSize * (moduleNumber - 1)
		starty := max(centery-length/2, 0)
		unmatch := 0
		for i := starty; i < starty+length && i < image.Height; i++ {
			if int(image.Pix[image.Width*i+centerx]) != color {
				unmatch++
			} else if unmatch <= tolerance {
				unmatch = 0
			}
			if unmatch > tolerance {
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
		for i := 0; i < length && starty+i < image.Height; i++ {
			if int(image.Pix[image.Width*(starty+i)+(startx+i)]) != color {
				unmatch++
			} else if unmatch <= tolerance {
				unmatch = 0
			}
			if unmatch > tolerance {
				break
			}
		}
		if unmatch < tolerance {
			return true
		}
		unmatch = 0
		startx = max(centerx-offset, 0)
		starty = min(centery+offset, image.Height-1)
		for i := 0; i < length && starty-i >= 0; i++ {
			if int(image.Pix[image.Width*(starty-i)+(startx+i)]) != color {
				unmatch++
			} else if unmatch <= tolerance {
				unmatch = 0
			}
			if unmatch > tolerance {
				return false
			}
		}
		return true
	}
	return false
}

// crossCheckPatternCh validates a candidate in a single channel across vertical,
// horizontal and diagonal directions.
func crossCheckPatternCh(ch *Bitmap, typ, hv int, moduleSizeMax float64, moduleSize, centerx, centery *float64, dir, dcc *int) bool {
	// Ports crossCheckPatternCh in detector.c.
	var msV, msH, msD float64
	if hv == 0 {
		vcc := false
		if crossCheckPatternVertical(ch, int(moduleSizeMax), *centerx, centery, &msV) {
			vcc = true
			if !crossCheckPatternHorizontal(ch, moduleSizeMax, centerx, *centery, &msH) {
				return false
			}
		}
		*dcc = crossCheckPatternDiagonal(ch, typ, moduleSizeMax, centerx, centery, &msD, dir, !vcc)
		switch {
		case vcc && *dcc > 0:
			*moduleSize = (msV + msH + msD) / 3.0
			return true
		case *dcc == 2:
			if !crossCheckPatternHorizontal(ch, moduleSizeMax, centerx, *centery, &msH) {
				return false
			}
			*moduleSize = (msH + msD*2.0) / 3.0
			return true
		}
	} else {
		hcc := false
		if crossCheckPatternHorizontal(ch, moduleSizeMax, centerx, *centery, &msH) {
			hcc = true
			if !crossCheckPatternVertical(ch, int(moduleSizeMax), *centerx, centery, &msV) {
				return false
			}
		}
		*dcc = crossCheckPatternDiagonal(ch, typ, moduleSizeMax, centerx, centery, &msD, dir, !hcc)
		switch {
		case hcc && *dcc > 0:
			*moduleSize = (msV + msH + msD) / 3.0
			return true
		case *dcc == 2:
			if !crossCheckPatternVertical(ch, int(moduleSizeMax), *centerx, centery, &msV) {
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
func crossCheckPattern(ch [3]*Bitmap, fp *FinderPattern, hv int) bool {
	// Ports crossCheckPattern in detector.c.
	moduleSizeMax := fp.ModuleSize * 2.0

	var msG float64
	cxG, cyG := fp.Center.X, fp.Center.Y
	dirG, dccG := 0, 0
	if !crossCheckPatternCh(ch[1], fp.Typ, hv, moduleSizeMax, &msG, &cxG, &cyG, &dirG, &dccG) {
		return false
	}

	if fp.Typ == fp1 || fp.Typ == fp2 {
		var msR float64
		cxR, cyR := fp.Center.X, fp.Center.Y
		dirR, dccR := 0, 0
		if !crossCheckPatternCh(ch[0], fp.Typ, hv, moduleSizeMax, &msR, &cxR, &cyR, &dirR, &dccR) {
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
			if !crossCheckColor(ch[2], coreBlue, int(fp.ModuleSize), 5, int(fp.Center.X), int(fp.Center.Y), d) {
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
		if !crossCheckPatternCh(ch[2], fp.Typ, hv, moduleSizeMax, &msB, &cxB, &cyB, &dirB, &dccB) {
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
			if !crossCheckColor(ch[0], coreRed, int(fp.ModuleSize), 5, int(fp.Center.X), int(fp.Center.Y), d) {
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

// getBestPattern returns the most-frequently-detected pattern (ties broken by
// closeness to the mean module size) and clears it from the list.
func getBestPattern(fps []FinderPattern, fpCount int) FinderPattern {
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

// selectBestPatterns reduces the candidate list to the single best pattern of
// each of the four types in fps[0..3], returning how many types are missing
// records the pre-prune group sizes and the post-prune selection in the current
// pass's d.stats. fpTypeCount is unused here, kept to mirror the C signature.
func (d *PrimaryDetector) selectBestPatterns(fps []FinderPattern, fpCount int, fpTypeCount []int) int {
	// Ports selectBestPatterns in detector.c.
	var groups [4][]FinderPattern
	for i := range fpCount {
		if fps[i].FoundCount < 3 { // a module must be at least 3 pixels
			continue
		}
		if t := fps[i].Typ; t >= 0 && t < 4 {
			groups[t] = append(groups[t], fps[i])
		}
	}
	st := d.pass()
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
			fps[t] = getBestPattern(groups[t], len(groups[t]))
		}
	}

	maxFound := 0
	for i := range 4 {
		if fps[i].FoundCount > maxFound {
			maxFound = fps[i].FoundCount
		}
	}
	for i := range 4 {
		if fps[i].FoundCount > 0 && float64(fps[i].FoundCount) < 0.5*float64(maxFound) {
			fps[i] = FinderPattern{}
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
