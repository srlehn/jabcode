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
	topX, f1 := SideSize(edgeModuleNumber(bm, fps[0], fps[1]) + 7)
	botX, f2 := SideSize(edgeModuleNumber(bm, fps[3], fps[2]) + 7)
	x := chooseSideSize(topX, f1, botX, f2)

	leftY, f3 := SideSize(edgeModuleNumber(bm, fps[0], fps[3]) + 7)
	rightY, f4 := SideSize(edgeModuleNumber(bm, fps[1], fps[2]) + 7)
	y := chooseSideSize(leftY, f3, rightY, f4)

	return image.Pt(x, y)
}

// edgeModuleNumber counts the modules between two finder patterns, preferring
// the local-sampling walk and falling back to the distance estimate when the
// walk yields nothing.
func edgeModuleNumber(bm *core.Bitmap, fp1, fp2 FinderPattern) int {
	if n := LocalModuleCount(bm, fp1, fp2); n > 0 {
		return n
	}
	return CalculateModuleNumber(fp1, fp2)
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
