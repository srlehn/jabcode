package detect

import (
	"image"
	"math"

	"github.com/srlehn/jabcode/internal/core"
)

// CalculateModuleNumber estimates the number of modules between two patterns,
// correcting for the scanline angle.
func CalculateModuleNumber(fp1, fp2 FinderPattern) int {
	// Ports calculateModuleNumber in detector.c.
	dist := math.Hypot(fp1.Center.X-fp2.Center.X, fp1.Center.Y-fp2.Center.Y)
	cosTheta := math.Max(math.Abs(fp2.Center.X-fp1.Center.X), math.Abs(fp2.Center.Y-fp1.Center.Y)) / dist
	mean := (fp1.ModuleSize + fp2.ModuleSize) * cosTheta / 2.0
	return int(dist/mean + 0.5)
}

// SideSize rounds a raw module count to the nearest valid side size and
// returns a reliability flag. flag: 1 reliable, 0 guessed, -1 invalid.
func SideSize(size int) (int, int) {
	// Ports getSideSize in detector.c.
	flag := 1
	switch size & 0x03 {
	case 0:
		size++
	case 2:
		size--
	case 3:
		size += 2 // error bigger than 1; guess the next version
		flag = 0
	}
	if size < 21 || size > 145 {
		return -1, -1
	}
	return size, flag
}

// chooseSideSize picks between two side-size estimates by reliability.
func chooseSideSize(size1, flag1, size2, flag2 int) int {
	// Ports chooseSideSize in detector.c.
	switch {
	case flag1 == -1 && flag2 == -1:
		return -1
	case flag1 == flag2:
		return max(size1, size2)
	case flag1 > flag2:
		return size1
	default:
		return size2
	}
}

// CalculateSideSize derives the symbol's side size in modules from the four
// finder-pattern positions. When a bitmap is given, the modules along each
// edge are counted by the local-sampling walk, which stays accurate on large
// and rectangular symbols; nil restricts it to the finder-distance estimate.
// The layout is FP0 FP1 / FP3 FP2.
func CalculateSideSize(bm *core.Bitmap, fps []FinderPattern) image.Point {
	// Ports calculateSideSize in detector.c.
	x := chooseAxisSize(edgeEstimateOf(bm, fps[0], fps[1]), edgeEstimateOf(bm, fps[3], fps[2]))
	y := chooseAxisSize(edgeEstimateOf(bm, fps[0], fps[3]), edgeEstimateOf(bm, fps[1], fps[2]))
	return image.Pt(x, y)
}

// edgeEstimate is one finder-to-finder edge's module-count evidence: the
// walk-preferred rounded side size with its reliability flag (the ported
// preference), the distance method's rounding, and the quality bits
// chooseAxisSize ranks disagreeing edges by.
type edgeEstimate struct {
	size, flag         int
	distSize, distFlag int
	rounding           bool    // walk and distance raw counts differ by at most one module
	agree              bool    // walk and distance estimates round to the same size
	msRatio            float64 // endpoint module-size ratio >= 1; near 1 = consistent endpoints
}

// edgeEstimateOf measures one edge with both module-count methods, keeping
// the ported preference: the local-sampling walk corrects its error at every
// module and wins whenever it counts (the distance estimate's bias grows on
// large symbols and small modules); the distance estimate is the fallback.
// Both roundings are kept so chooseAxisSize can weigh them.
func edgeEstimateOf(bm *core.Bitmap, fp1, fp2 FinderPattern) edgeEstimate {
	w := LocalModuleCount(bm, fp1, fp2)
	d := CalculateModuleNumber(fp1, fp2)
	ws, wf := SideSize(w + 7)
	ds, df := SideSize(d + 7)
	e := edgeEstimate{
		size: ws, flag: wf,
		distSize: ds, distFlag: df,
		rounding: d-1 <= w && w <= d+1,
		agree:    ws > 0 && ws == ds,
	}
	if w <= 0 {
		e.size, e.flag = ds, df
	}
	lo, hi := fp1.ModuleSize, fp2.ModuleSize
	if lo > hi {
		lo, hi = hi, lo
	}
	if lo > 0 {
		e.msRatio = hi / lo
	} else {
		e.msRatio = math.Inf(1)
	}
	return e
}

// msRatioSlack is how far apart an edge's two endpoint module sizes may be
// before the edge counts as endpoint-inconsistent. Genuine perspective
// foreshortening on accepted captures stays below ~1.2; a mislocated finder
// measures near half its partner (1.8+), and poisons both edges it touches.
const msRatioSlack = 1.4

// chooseAxisSize resolves one axis from its two opposite-edge estimates.
// Opposite edges of a symbol cross the same number of modules, so agreement
// settles the axis. When the walks only disagree by rounding, a corroborated
// distance consensus overrides them: both edges' distance estimates rounding
// reliably to the SAME size, against walks whose own roundings are guesses
// one module away, means the walks drifted in step (a low-contrast
// high-colour grid does this) while the distances stayed clean - a single
// clean distance rounding proves nothing, which is why the consensus needs
// both edges. On remaining disagreement the more self-consistent edge wins:
// a reliable rounding beats a guess (the ported rule), agreeing count
// methods beat disagreeing ones, and consistent endpoint module sizes beat
// an edge touching a mismeasured finder. The ported max() stays as the
// final tie-break, so behaviour without quality evidence is unchanged.
func chooseAxisSize(a, b edgeEstimate) int {
	switch {
	case a.flag == -1 && b.flag == -1:
		return -1
	case a.flag == -1:
		return b.size
	case b.flag == -1:
		return a.size
	}
	if a.distSize == b.distSize && a.distFlag == 1 && b.distFlag == 1 &&
		a.flag < 1 && b.flag < 1 && a.rounding && b.rounding {
		return a.distSize
	}
	switch {
	case a.size == b.size:
		return a.size
	case a.flag != b.flag:
		if a.flag > b.flag {
			return a.size
		}
		return b.size
	case a.agree != b.agree:
		if a.agree {
			return a.size
		}
		return b.size
	}
	aOK, bOK := a.msRatio <= msRatioSlack, b.msRatio <= msRatioSlack
	if aOK != bOK {
		if aOK {
			return a.size
		}
		return b.size
	}
	return max(a.size, b.size)
}

// averagePixelValue computes the average RGB value in a neighborhood around
// each detected finder pattern, then averages those — used as adaptive black
// thresholds for a second binarization pass (averagePixelValue).
func averagePixelValue(bm *core.Bitmap, fps []FinderPattern) [3]float32 {
	var rAvg, gAvg, bAvg [4]float64
	bpp := bm.Channels
	bytesPerRow := bm.Width * bpp

	for i := range 4 {
		if fps[i].FoundCount <= 0 {
			continue
		}
		radius := fps[i].ModuleSize * 4
		startX := max(int(fps[i].Center.X-radius), 0)
		startY := max(int(fps[i].Center.Y-radius), 0)
		endX := min(int(fps[i].Center.X+radius), bm.Width-1)
		endY := min(int(fps[i].Center.Y+radius), bm.Height-1)
		for y := startY; y < endY; y++ {
			for x := startX; x < endX; x++ {
				offset := y*bytesPerRow + x*bpp
				rAvg[i] += float64(bm.Pix[offset+0])
				gAvg[i] += float64(bm.Pix[offset+1])
				bAvg[i] += float64(bm.Pix[offset+2])
			}
		}
		area := float64((endX - startX) * (endY - startY))
		rAvg[i] /= area
		gAvg[i] /= area
		bAvg[i] /= area
	}

	var sum [3]float64
	var count [3]int
	for i := range 4 {
		if rAvg[i] > 0 {
			sum[0] += rAvg[i]
			count[0]++
		}
		if gAvg[i] > 0 {
			sum[1] += gAvg[i]
			count[1]++
		}
		if bAvg[i] > 0 {
			sum[2] += bAvg[i]
			count[2]++
		}
	}
	var avg [3]float32
	for c := range 3 {
		if count[c] > 0 {
			avg[c] = float32(sum[c] / float64(count[c]))
		}
	}
	return avg
}
