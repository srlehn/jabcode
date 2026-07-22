package detect

import (
	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/spec"
)

// seekMissingFinderPattern searches a local area around the estimated position
// of a single missing finder pattern and, if found, replaces the estimate with
// the detected pattern.
func seekMissingFinderPattern(bm *core.Bitmap, fps []FinderPattern, missIndex int) {
	// Ports seekMissingFinderPattern in detector.c.
	radius := fps[missIndex].ModuleSize * 5
	startX := max(int(fps[missIndex].Center.X-radius), 0)
	startY := max(int(fps[missIndex].Center.Y-radius), 0)
	endX := min(int(fps[missIndex].Center.X+radius), bm.Width-1)
	endY := min(int(fps[missIndex].Center.Y+radius), bm.Height-1)
	areaW := endX - startX
	areaH := endY - startY
	if areaW <= 0 || areaH <= 0 {
		return
	}

	var rgb [3]*core.Bitmap
	for i := range rgb {
		rgb[i] = core.NewBitmap(areaW, areaH, 1)
	}

	bpp := bm.Channels
	bytesPerRow := bm.Width * bpp
	var sum [3]int64
	for i := startY; i < endY; i++ {
		for j := startX; j < endX; j++ {
			off := i*bytesPerRow + j*bpp
			sum[0] += int64(bm.Pix[off+0])
			sum[1] += int64(bm.Pix[off+1])
			sum[2] += int64(bm.Pix[off+2])
		}
	}
	area := int64(areaW) * int64(areaH)

	// Quantize the search area to black, cyan and yellow.
	for i, y := startY, 0; i < endY; i, y = i+1, y+1 {
		for j, x := startX, 0; j < endX; j, x = j+1, x+1 {
			off := i*bytesPerRow + j*bpp
			r, g, b := bm.Pix[off+0], bm.Pix[off+1], bm.Pix[off+2]
			idx := y*areaW + x
			switch {
			case int64(r)*area < sum[0] && int64(g)*area < sum[1] && int64(b)*area < sum[2]: // black
			case r < b: // cyan
				rgb[1].Pix[idx] = 255
				rgb[2].Pix[idx] = 255
			default: // yellow
				rgb[0].Pix[idx] = 255
				rgb[1].Pix[idx] = 255
			}
		}
	}

	var expR, expG, expB int
	switch missIndex {
	case fp2:
		expR, expG, expB = 255, 255, 0
	case fp3:
		expR, expG, expB = 0, 255, 255
	} // fp0/fp1 expect 0,0,0

	fpsMiss := make([]FinderPattern, maxFinderPatterns)
	total := 0
	fpTypeCount := make([]int, 4)
	done := false

	for i := 0; i < areaH && !done; i++ {
		rowR := rgb[0].Pix[i*areaW : (i+1)*areaW]
		rowG := rgb[1].Pix[i*areaW : (i+1)*areaW]
		rowB := rgb[2].Pix[i*areaW : (i+1)*areaW]
		startx, endx, skip := 0, areaW, 0
		for first := true; first || (startx < areaW && endx < areaW); {
			first = false
			startx += skip
			endx = areaW
			ps := seekPatternHorizontal(rowG, startx, endx)
			startx, endx = ps.start, ps.end
			if !ps.ok {
				continue
			}
			skip = ps.skip
			centerxG, moduleSizeG := ps.Center, ps.ModuleSize
			if core.BoolColor(rowG[int(centerxG)] > 0) != expG {
				continue
			}
			centerxR, centerxB := centerxG, centerxG
			var moduleSizeR, moduleSizeB float64
			found := false
			var fp FinderPattern

			switch missIndex {
			case fp0, fp3:
				if crossCheckPatternHorizontal(rgb[2], moduleSizeG*2, &centerxB, float64(i), &moduleSizeB, 3) {
					if core.BoolColor(rowB[int(centerxB)] > 0) != expB {
						continue
					}
					moduleSizeR = moduleSizeG
					if crossCheckColor(rgb[0], int(palette.Default[spec.FP3CoreColor*3+0]), int(moduleSizeR), 5, int(centerxR), i, 0, 3) {
						found = true
					}
				}
				if found {
					if !checkModuleSize2(moduleSizeG, moduleSizeB) {
						continue
					}
					fp.Center.X = (centerxG + centerxB) / 2.0
					fp.ModuleSize = (moduleSizeG + moduleSizeB) / 2.0
				}
			case fp1, fp2:
				if crossCheckPatternHorizontal(rgb[0], moduleSizeG*2, &centerxR, float64(i), &moduleSizeR, 3) {
					if core.BoolColor(rowR[int(centerxR)] > 0) != expR {
						continue
					}
					moduleSizeB = moduleSizeG
					if crossCheckColor(rgb[2], int(palette.Default[spec.FP2CoreColor*3+2]), int(moduleSizeB), 5, int(centerxB), i, 0, 3) {
						found = true
					}
				}
				if found {
					if !checkModuleSize2(moduleSizeR, moduleSizeG) {
						continue
					}
					fp.Center.X = (centerxR + centerxG) / 2.0
					fp.ModuleSize = (moduleSizeR + moduleSizeG) / 2.0
				}
			}

			if found {
				fp.Center.Y = float64(i)
				fp.FoundCount = 1
				fp.Typ = missIndex
				if crossCheckPattern(rgb, &fp, 0, 3) {
					saveFinderPattern(&fp, fpsMiss, &total, fpTypeCount)
					if total >= maxFinderPatterns-1 {
						done = true
						break
					}
				}
			}
		}
	}

	if total > 0 {
		maxFound, maxIdx := 0, 0
		for i := 0; i < total; i++ {
			if fpsMiss[i].FoundCount > maxFound {
				maxFound = fpsMiss[i].FoundCount
				maxIdx = i
			}
		}
		fps[missIndex] = fpsMiss[maxIdx]
		fps[missIndex].Center.X += float64(startX)
		fps[missIndex].Center.Y += float64(startY)
	}
}
