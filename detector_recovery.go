package jabcode

import "github.com/srlehn/jabcode/internal/palette"

// seekMissingFinderPattern searches a local area around the estimated position
// of a single missing finder pattern and, if found, replaces the estimate with
// the detected pattern (seekMissingFinderPattern in detector.c).
func seekMissingFinderPattern(bm *bitmap, fps []finderPattern, missIndex int) {
	radius := fps[missIndex].moduleSize * 5
	startX := max(int(fps[missIndex].center.x-radius), 0)
	startY := max(int(fps[missIndex].center.y-radius), 0)
	endX := min(int(fps[missIndex].center.x+radius), bm.width-1)
	endY := min(int(fps[missIndex].center.y+radius), bm.height-1)
	areaW := endX - startX
	areaH := endY - startY
	if areaW <= 0 || areaH <= 0 {
		return
	}

	var rgb [3]*bitmap
	for i := range rgb {
		rgb[i] = newBitmap(areaW, areaH, 1)
	}

	bpp := bm.channels
	bytesPerRow := bm.width * bpp
	var sum [3]float64
	for i := startY; i < endY; i++ {
		for j := startX; j < endX; j++ {
			off := i*bytesPerRow + j*bpp
			sum[0] += float64(bm.pix[off+0])
			sum[1] += float64(bm.pix[off+1])
			sum[2] += float64(bm.pix[off+2])
		}
	}
	area := float64(areaW * areaH)
	avg := [3]float64{sum[0] / area, sum[1] / area, sum[2] / area}

	// Quantize the search area to black, cyan and yellow.
	for i, y := startY, 0; i < endY; i, y = i+1, y+1 {
		for j, x := startX, 0; j < endX; j, x = j+1, x+1 {
			off := i*bytesPerRow + j*bpp
			r, g, b := bm.pix[off+0], bm.pix[off+1], bm.pix[off+2]
			idx := y*areaW + x
			switch {
			case float64(r) < avg[0] && float64(g) < avg[1] && float64(b) < avg[2]: // black
			case r < b: // cyan
				rgb[1].pix[idx] = 255
				rgb[2].pix[idx] = 255
			default: // yellow
				rgb[0].pix[idx] = 255
				rgb[1].pix[idx] = 255
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

	fpsMiss := make([]finderPattern, maxFinderPatterns)
	total := 0
	fpTypeCount := make([]int, 4)
	done := false

	for i := 0; i < areaH && !done; i++ {
		rowR := rgb[0].pix[i*areaW : (i+1)*areaW]
		rowG := rgb[1].pix[i*areaW : (i+1)*areaW]
		rowB := rgb[2].pix[i*areaW : (i+1)*areaW]
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
			centerxG, moduleSizeG := ps.center, ps.moduleSize
			if boolColor(rowG[int(centerxG)] > 0) != expG {
				continue
			}
			centerxR, centerxB := centerxG, centerxG
			var moduleSizeR, moduleSizeB float64
			found := false
			var fp finderPattern

			switch missIndex {
			case fp0, fp3:
				if crossCheckPatternHorizontal(rgb[2], moduleSizeG*2, &centerxB, float64(i), &moduleSizeB) {
					if boolColor(rowB[int(centerxB)] > 0) != expB {
						continue
					}
					moduleSizeR = moduleSizeG
					if crossCheckColor(rgb[0], int(palette.Default[fp3CoreColor*3+0]), int(moduleSizeR), 5, int(centerxR), i, 0) {
						found = true
					}
				}
				if found {
					if !checkModuleSize2(moduleSizeG, moduleSizeB) {
						continue
					}
					fp.center.x = (centerxG + centerxB) / 2.0
					fp.moduleSize = (moduleSizeG + moduleSizeB) / 2.0
				}
			case fp1, fp2:
				if crossCheckPatternHorizontal(rgb[0], moduleSizeG*2, &centerxR, float64(i), &moduleSizeR) {
					if boolColor(rowR[int(centerxR)] > 0) != expR {
						continue
					}
					moduleSizeB = moduleSizeG
					if crossCheckColor(rgb[2], int(palette.Default[fp2CoreColor*3+2]), int(moduleSizeB), 5, int(centerxB), i, 0) {
						found = true
					}
				}
				if found {
					if !checkModuleSize2(moduleSizeR, moduleSizeG) {
						continue
					}
					fp.center.x = (centerxR + centerxG) / 2.0
					fp.moduleSize = (moduleSizeR + moduleSizeG) / 2.0
				}
			}

			if found {
				fp.center.y = float64(i)
				fp.foundCount = 1
				fp.typ = missIndex
				if crossCheckPattern(rgb, &fp, 0) {
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
			if fpsMiss[i].foundCount > maxFound {
				maxFound = fpsMiss[i].foundCount
				maxIdx = i
			}
		}
		fps[missIndex] = fpsMiss[maxIdx]
		fps[missIndex].center.x += float64(startX)
		fps[missIndex].center.y += float64(startY)
	}
}
