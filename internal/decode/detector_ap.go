package decode

import (
	"image"
	"math"
	"sort"

	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/tables"
)

// Alignment-pattern types. AP0..AP3 share core color index 3 (cyan); APX uses
// index 6 (yellow).
const (
	ap0 = 0
	ap1 = 1
	ap2 = 2
	ap3 = 3
	apx = 4
)

// apCoreColorIndex returns the default-palette color index of an alignment
// pattern's core.
func apCoreColorIndex(apType int) int {
	if apType == apx {
		return 6
	}
	return 3
}

// saveAlignmentPattern merges an alignment pattern into the list, returning the
// index if it combined with an existing one, or -1 if appended.
func saveAlignmentPattern(ap *finderPattern, aps []finderPattern, counter *int) int {
	// Ports saveAlignmentPattern in detector.c.
	for i := 0; i < *counter; i++ {
		if aps[i].foundCount > 0 &&
			math.Abs(ap.center.x-aps[i].center.x) <= ap.moduleSize && math.Abs(ap.center.y-aps[i].center.y) <= ap.moduleSize &&
			(math.Abs(ap.moduleSize-aps[i].moduleSize) <= aps[i].moduleSize || math.Abs(ap.moduleSize-aps[i].moduleSize) <= 1.0) &&
			ap.typ == aps[i].typ {
			fc := float64(aps[i].foundCount)
			aps[i].center.x = (fc*aps[i].center.x + ap.center.x) / (fc + 1)
			aps[i].center.y = (fc*aps[i].center.y + ap.center.y) / (fc + 1)
			aps[i].moduleSize = (fc*aps[i].moduleSize + ap.moduleSize) / (fc + 1)
			aps[i].foundCount++
			return i
		}
	}
	aps[*counter] = *ap
	*counter++
	return -1
}

// crossCheckPatternDiagonalAP validates an alignment pattern along a diagonal,
// returning the refined center y or -1.
func crossCheckPatternDiagonalAP(image *bitmap, apType, moduleSizeMax int, center pointF, dir *int) float64 {
	// Ports crossCheckPatternDiagonalAP in detector.c.
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
	case apType == ap0 || apType == ap1:
		offsetX, offsetY, *dir = -1, -1, 1
	default:
		offsetX, offsetY, *dir = 1, -1, -1
	}

	tryCount := 0
	for {
		flag := false
		tryCount++
		var i, si int
		var sc [3]int
		startx := int(center.x)
		starty := int(center.y)

		sc[1]++
		for i = 1; i <= starty && i <= startx && si <= 1; i++ {
			if image.pix[(starty+i*offsetY)*image.width+(startx+i*offsetX)] == image.pix[(starty+(i-1)*offsetY)*image.width+(startx+(i-1)*offsetX)] {
				sc[1-si]++
			} else if si > 0 && sc[1-si] < 3 {
				sc[1-(si-1)] += sc[1-si]
				sc[1-si] = 0
				si--
				sc[1-si]++
			} else {
				si++
				if si > 1 {
					break
				}
				sc[1-si]++
			}
		}
		if si < 1 {
			if tryCount == 1 {
				flag, offsetX, *dir = true, -offsetX, -(*dir)
			} else {
				return -1
			}
		}
		if !flag {
			si = 0
			for i = 1; starty+i < image.height && startx+i < image.width && si <= 1; i++ {
				if image.pix[(starty-i*offsetY)*image.width+(startx-i*offsetX)] == image.pix[(starty-(i-1)*offsetY)*image.width+(startx-(i-1)*offsetX)] {
					sc[1+si]++
				} else if si > 0 && sc[1+si] < 3 {
					sc[1+(si-1)] += sc[1+si]
					sc[1+si] = 0
					si--
					sc[1+si]++
				} else {
					si++
					if si > 1 {
						break
					}
					sc[1+si]++
				}
			}
			if si < 1 {
				if tryCount == 1 {
					flag, offsetX, *dir = true, -offsetX, -(*dir)
				} else {
					return -1
				}
			}
		}
		if !flag {
			if sc[1] < moduleSizeMax && float64(sc[0]) > 0.5*float64(sc[1]) && float64(sc[2]) > 0.5*float64(sc[1]) {
				return float64(starty+i-sc[2]) - float64(sc[1])/2.0
			}
			flag, offsetX, *dir = true, -offsetX, -(*dir)
		}
		if !(flag && tryCount < 2 && !fixDir) {
			break
		}
	}
	return -1
}

// crossCheckPatternVerticalAP validates an alignment pattern along the vertical,
// returning the refined center y or -1.
func crossCheckPatternVerticalAP(image *bitmap, center pointF, moduleSizeMax int, moduleSize *float64) float64 {
	// Ports crossCheckPatternVerticalAP in detector.c.
	var sc [3]int
	cx, cy := int(center.x), int(center.y)
	var i, si int
	sc[1]++
	for i = 1; i <= cy && si <= 1; i++ {
		if image.pix[(cy-i)*image.width+cx] == image.pix[(cy-(i-1))*image.width+cx] {
			sc[1-si]++
		} else if si > 0 && sc[1-si] < 3 {
			sc[1-(si-1)] += sc[1-si]
			sc[1-si] = 0
			si--
			sc[1-si]++
		} else {
			si++
			if si > 1 {
				break
			}
			sc[1-si]++
		}
	}
	if si < 1 {
		return -1
	}
	si = 0
	for i = 1; cy+i < image.height && si <= 1; i++ {
		if image.pix[(cy+i)*image.width+cx] == image.pix[(cy+(i-1))*image.width+cx] {
			sc[1+si]++
		} else if si > 0 && sc[1+si] < 3 {
			sc[1+(si-1)] += sc[1+si]
			sc[1+si] = 0
			si--
			sc[1+si]++
		} else {
			si++
			if si > 1 {
				break
			}
			sc[1+si]++
		}
	}
	if si < 1 {
		return -1
	}
	if sc[1] < moduleSizeMax && float64(sc[0]) > 0.5*float64(sc[1]) && float64(sc[2]) > 0.5*float64(sc[1]) {
		*moduleSize = float64(sc[1])
		return float64(cy+i-sc[2]) - float64(sc[1])/2.0
	}
	return -1
}

// crossCheckPatternHorizontalAP validates an alignment pattern along a row,
// returning the refined center x or -1.
func crossCheckPatternHorizontalAP(row []byte, channel, startx, endx, centerx, apType int, moduleSizeMax float64, moduleSize *float64) float64 {
	// Ports crossCheckPatternHorizontalAP in detector.c.
	coreColor := int(palette.Default[apCoreColorIndex(apType)*3+channel])
	if int(row[centerx]) != coreColor {
		return -1
	}
	var sc [3]int
	var i, si int
	sc[1]++
	for i = 1; centerx-i >= startx && si <= 1; i++ {
		if row[centerx-i] == row[centerx-(i-1)] {
			sc[1-si]++
		} else if si > 0 && sc[1-si] < 3 {
			sc[1-(si-1)] += sc[1-si]
			sc[1-si] = 0
			si--
			sc[1-si]++
		} else {
			si++
			if si > 1 {
				break
			}
			sc[1-si]++
		}
	}
	if si < 1 {
		return -1
	}
	si = 0
	for i = 1; centerx+i <= endx && si <= 1; i++ {
		if row[centerx+i] == row[centerx+(i-1)] {
			sc[1+si]++
		} else if si > 0 && sc[1+si] < 3 {
			sc[1+(si-1)] += sc[1+si]
			sc[1+si] = 0
			si--
			sc[1+si]++
		} else {
			si++
			if si > 1 {
				break
			}
			sc[1+si]++
		}
	}
	if si < 1 {
		return -1
	}
	if float64(sc[1]) < moduleSizeMax && float64(sc[0]) > 0.5*float64(sc[1]) && float64(sc[2]) > 0.5*float64(sc[1]) {
		*moduleSize = float64(sc[1])
		return float64(centerx+i-sc[2]) - float64(sc[1])/2.0
	}
	return -1
}

// crossCheckPatternAP validates an alignment-pattern candidate across channels
// and directions, refining its center, module size and direction.
func crossCheckPatternAP(ch [3]*bitmap, y, minx, maxx, curX, apType int, maxModuleSize float64, centerx, centery, moduleSize *float64, dir *int) bool {
	// Ports crossCheckPatternAP in detector.c.
	rowR := ch[0].pix[y*ch[0].width : (y+1)*ch[0].width]
	rowB := ch[2].pix[y*ch[2].width : (y+1)*ch[2].width]
	var lcx, lcy, lmsH, lmsV [3]float64

	lcx[0] = crossCheckPatternHorizontalAP(rowR, 0, minx, maxx, curX, apType, maxModuleSize, &lmsH[0])
	if lcx[0] < 0 {
		return false
	}
	lcx[2] = crossCheckPatternHorizontalAP(rowB, 2, minx, maxx, int(lcx[0]), apType, maxModuleSize, &lmsH[2])
	if lcx[2] < 0 {
		return false
	}
	center := pointF{(lcx[0] + lcx[2]) / 2.0, float64(y)}
	*moduleSize = (lmsH[0] + lmsH[2]) / 2.0
	greenCore := int(palette.Default[apCoreColorIndex(apType)*3+1])
	if !crossCheckColor(ch[1], greenCore, int(*moduleSize), 3, int(center.x), int(center.y), 0) {
		return false
	}

	lcy[0] = crossCheckPatternVerticalAP(ch[0], center, int(maxModuleSize), &lmsV[0])
	if lcy[0] < 0 {
		return false
	}
	rowR = ch[0].pix[int(lcy[0])*ch[0].width : (int(lcy[0])+1)*ch[0].width]
	lcx[0] = crossCheckPatternHorizontalAP(rowR, 0, minx, maxx, int(center.x), apType, maxModuleSize, &lmsH[0])
	if lcx[0] < 0 {
		return false
	}

	lcy[2] = crossCheckPatternVerticalAP(ch[2], center, int(maxModuleSize), &lmsV[2])
	if lcy[2] < 0 {
		return false
	}
	rowB = ch[2].pix[int(lcy[2])*ch[2].width : (int(lcy[2])+1)*ch[2].width]
	lcx[2] = crossCheckPatternHorizontalAP(rowB, 2, minx, maxx, int(center.x), apType, maxModuleSize, &lmsH[2])
	if lcx[2] < 0 {
		return false
	}

	*moduleSize = (lmsH[0] + lmsH[2] + lmsV[0] + lmsV[2]) / 4.0
	*centerx = (lcx[0] + lcx[2]) / 2.0
	*centery = (lcy[0] + lcy[2]) / 2.0
	center.x, center.y = *centerx, *centery
	if !crossCheckColor(ch[1], greenCore, int(*moduleSize), 3, int(center.x), int(center.y), 1) {
		return false
	}

	var ldir [3]int
	if crossCheckPatternDiagonalAP(ch[0], apType, int(*moduleSize*2), center, &ldir[0]) < 0 {
		return false
	}
	if crossCheckPatternDiagonalAP(ch[2], apType, int(*moduleSize*2), center, &ldir[2]) < 0 {
		return false
	}
	if !crossCheckColor(ch[1], greenCore, int(*moduleSize), 3, int(center.x), int(center.y), 2) {
		return false
	}
	if ldir[0]+ldir[2] > 0 {
		*dir = 1
	} else {
		*dir = -1
	}
	return true
}

// findAlignmentPattern searches for an alignment pattern of the given type near
// (x, y).
func findAlignmentPattern(ch [3]*bitmap, x, y, moduleSize float64, apType int) finderPattern {
	// Ports findAlignmentPattern in detector.c.
	coreColorR := byte(palette.Default[apCoreColorIndex(apType)*3])
	radius := int(4 * moduleSize)
	radiusMax := 4 * radius
	for ; radius < radiusMax; radius <<= 1 {
		aps := make([]finderPattern, maxFinderPatterns)
		startx := max(0, int(x)-radius)
		starty := max(0, int(y)-radius)
		endx := min(ch[0].width-1, int(x)+radius)
		endy := min(ch[0].height-1, int(y)+radius)
		if float64(endx-startx) < 3*moduleSize || float64(endy-starty) < 3*moduleSize {
			continue
		}
		counter := 0
		for k := starty; k < endy; k++ {
			kk := k - starty
			var i int
			if kk&0x01 == 0 {
				i = int(y) + (kk+1)/2
			} else {
				i = int(y) - (kk+1)/2
			}
			if i < starty || i > endy {
				continue
			}
			rowR := ch[0].pix[i*ch[0].width : (i+1)*ch[0].width]

			var apModuleSize, centerx, centery float64
			var apDir int
			apFound := false
			dir := -1
			// Seed the outward scan inside the clamped window. For an in-image centre
			// this is int(x) unchanged; a bad geometry can drive x off-image (negative
			// or past the width), and the inner loops read rowR[leftTmpX] before testing
			// the bound, so an unclamped seed would index out of range.
			cx := min(max(int(x), startx), endx)
			leftTmpX, rightTmpX := cx, cx
			for (leftTmpX > startx || rightTmpX < endx) && !apFound {
				if dir < 0 {
					for rowR[leftTmpX] != coreColorR && leftTmpX > startx {
						leftTmpX--
					}
					if leftTmpX <= startx {
						dir = -dir
						continue
					}
					apFound = crossCheckPatternAP(ch, i, startx, endx, leftTmpX, apType, moduleSize*2, &centerx, &centery, &apModuleSize, &apDir)
					for rowR[leftTmpX] == coreColorR && leftTmpX > startx {
						leftTmpX--
					}
					dir = -dir
				} else {
					for rowR[rightTmpX] == coreColorR && rightTmpX < endx {
						rightTmpX++
					}
					for rowR[rightTmpX] != coreColorR && rightTmpX < endx {
						rightTmpX++
					}
					if rightTmpX >= endx {
						dir = -dir
						continue
					}
					apFound = crossCheckPatternAP(ch, i, startx, endx, rightTmpX, apType, moduleSize*2, &centerx, &centery, &apModuleSize, &apDir)
					for rowR[rightTmpX] == coreColorR && rightTmpX < endx {
						rightTmpX++
					}
					dir = -dir
				}
			}
			if !apFound {
				continue
			}
			ap := finderPattern{typ: apType, foundCount: 1, moduleSize: apModuleSize, center: pointF{centerx, centery}, direction: apDir}
			if index := saveAlignmentPattern(&ap, aps, &counter); index >= 0 {
				return aps[index]
			}
		}
	}
	return finderPattern{typ: -1}
}

// getFirstAPPos rounds a raw module count to the nearest valid first-AP position.
func getFirstAPPos(pos int) int {
	// Ports getFirstAPPos in detector.c.
	switch pos % 3 {
	case 0:
		pos--
	case 1:
		pos++
	}
	if pos < 14 || pos > 26 {
		pos = -1
	}
	return pos
}

// detectFirstAP detects the first alignment pattern between two finder patterns,
// returning its position.
func detectFirstAP(ch [3]*bitmap, sideVersion int, fp1, fp2 finderPattern) int {
	// Ports detectFirstAP in detector.c.
	alpha := math.Atan2(fp2.center.y-fp1.center.y, fp2.center.x-fp1.center.x)
	nextVersion := sideVersion
	dir := 1
	up, down := 0, 0
	for {
		distance := fp1.moduleSize * float64(tables.APPos[nextVersion-1][1]-tables.APPos[nextVersion-1][0])
		cx := fp1.center.x + distance*math.Cos(alpha)
		cy := fp1.center.y + distance*math.Sin(alpha)
		ap := findAlignmentPattern(ch, cx, cy, fp1.moduleSize, apx)
		if ap.foundCount > 0 {
			if pos := getFirstAPPos(4 + calculateModuleNumber(fp1, ap)); pos > 0 {
				return pos
			}
		}
		dir = -dir
		if dir == -1 {
			up++
			nextVersion = up*dir + sideVersion
			if nextVersion < 6 || nextVersion > 32 {
				dir, up, down = -dir, up-1, down+1
				nextVersion = down*dir + sideVersion
			}
		} else {
			down++
			nextVersion = down*dir + sideVersion
			if nextVersion < 6 || nextVersion > 32 {
				dir, down, up = -dir, down-1, up+1
				nextVersion = up*dir + sideVersion
			}
		}
		if up+down >= 5 {
			break
		}
	}
	return jabFailure
}

// confirmSideVersion confirms a side version from the first AP position.
func confirmSideVersion(sideVersion, firstAPPos int) int {
	// Ports confirmSideVersion in detector.c.
	if firstAPPos <= 0 {
		return jabFailure
	}
	v := sideVersion
	k, sign := 1, -1
	flag := false
	for {
		if firstAPPos == tables.APPos[v-1][1] {
			flag = true
			break
		}
		v = sideVersion + sign*k
		if sign > 0 {
			k++
		}
		sign = -sign
		if v < 6 || v > 32 {
			break
		}
	}
	if flag {
		return v
	}
	return jabFailure
}

// confirmSymbolSize confirms the symbol's side sizes using alignment patterns.
func confirmSymbolSize(ch [3]*bitmap, fps []finderPattern, symbol *decodedSymbol) bool {
	// Ports confirmSymbolSize in detector.c.
	pos := detectFirstAP(ch, symbol.meta.sideVersion.X, fps[0], fps[1])
	vx := confirmSideVersion(symbol.meta.sideVersion.X, pos)
	if vx == 0 {
		pos = detectFirstAP(ch, symbol.meta.sideVersion.X, fps[3], fps[2])
		vx = confirmSideVersion(symbol.meta.sideVersion.X, pos)
		if vx == 0 {
			return false
		}
	}
	symbol.meta.sideVersion.X = vx
	symbol.sideSize.X = spec.VersionToSize(vx)

	pos = detectFirstAP(ch, symbol.meta.sideVersion.Y, fps[0], fps[3])
	vy := confirmSideVersion(symbol.meta.sideVersion.Y, pos)
	if vy == 0 {
		pos = detectFirstAP(ch, symbol.meta.sideVersion.Y, fps[1], fps[2])
		vy = confirmSideVersion(symbol.meta.sideVersion.Y, pos)
		if vy == 0 {
			return false
		}
	}
	symbol.meta.sideVersion.Y = vy
	symbol.sideSize.Y = spec.VersionToSize(vy)
	return true
}

// sampleSymbolByAlignmentPattern detects all alignment patterns, splits the
// symbol into blocks bounded by four found patterns, and samples each block with
// its own perspective transform.
func sampleSymbolByAlignmentPattern(bm *bitmap, ch [3]*bitmap, symbol *decodedSymbol, fps []finderPattern) *bitmap {
	// Ports sampleSymbolByAlignmentPattern in detector.c.
	if symbol.meta.sideVersion.X < 6 && symbol.meta.sideVersion.Y < 6 {
		return nil
	}
	if symbol.meta.defaultMode {
		if !confirmSymbolSize(ch, fps, symbol) {
			return nil
		}
	}

	vxi := symbol.meta.sideVersion.X - 1
	vyi := symbol.meta.sideVersion.Y - 1
	nApX := tables.APNum[vxi]
	nApY := tables.APNum[vyi]

	aps := make([]finderPattern, nApX*nApY)
	for i := range nApY {
		for j := range nApX {
			index := i*nApX + j
			switch {
			case i == 0 && j == 0:
				aps[index] = fps[0]
			case i == 0 && j == nApX-1:
				aps[index] = fps[1]
			case i == nApY-1 && j == nApX-1:
				aps[index] = fps[2]
			case i == nApY-1 && j == 0:
				aps[index] = fps[3]
			default:
				switch {
				case i == 0:
					alpha := math.Atan2(fps[1].center.y-aps[j-1].center.y, fps[1].center.x-aps[j-1].center.x)
					distance := aps[j-1].moduleSize * float64(tables.APPos[vxi][j]-tables.APPos[vxi][j-1])
					aps[index].center.x = aps[j-1].center.x + distance*math.Cos(alpha)
					aps[index].center.y = aps[j-1].center.y + distance*math.Sin(alpha)
					aps[index].moduleSize = aps[j-1].moduleSize
				case j == 0:
					base := (i - 1) * nApX
					alpha := math.Atan2(fps[3].center.y-aps[base].center.y, fps[3].center.x-aps[base].center.x)
					distance := aps[base].moduleSize * float64(tables.APPos[vyi][i]-tables.APPos[vyi][i-1])
					aps[index].center.x = aps[base].center.x + distance*math.Cos(alpha)
					aps[index].center.y = aps[base].center.y + distance*math.Sin(alpha)
					aps[index].moduleSize = aps[base].moduleSize
				default:
					iAp0 := (i-1)*nApX + (j - 1)
					iAp1 := (i-1)*nApX + j
					iAp3 := i*nApX + (j - 1)
					avg01 := (aps[iAp0].moduleSize + aps[iAp1].moduleSize) / 2.0
					avg13 := (aps[iAp1].moduleSize + aps[iAp3].moduleSize) / 2.0
					aps[index].center.x = (aps[iAp1].center.x-aps[iAp0].center.x)/avg01*avg13 + aps[iAp3].center.x
					aps[index].center.y = (aps[iAp1].center.y-aps[iAp0].center.y)/avg01*avg13 + aps[iAp3].center.y
					aps[index].moduleSize = avg13
				}
				aps[index].foundCount = 0
				tmp := aps[index]
				aps[index] = findAlignmentPattern(ch, aps[index].center.x, aps[index].center.y, aps[index].moduleSize, apx)
				if aps[index].foundCount == 0 {
					aps[index] = tmp
				}
			}
		}
	}

	// Determine the minimal sampling rectangle (four found APs) for each cell.
	type rect struct{ tl, br image.Point }
	var rects []rect
	for i := 0; i < nApY-1; i++ {
		for j := 0; j < nApX-1; j++ {
			var tl, br image.Point
			flag := true
			for delta := 0; delta <= (nApX-2)+(nApY-2) && flag; delta++ {
				for dy := 0; dy <= min(delta, nApY-2) && flag; dy++ {
					dx := min(delta-dy, nApX-2)
					for dy1 := 0; dy1 <= dy && flag; dy1++ {
						dy2 := dy - dy1
						for dx1 := 0; dx1 <= dx && flag; dx1++ {
							dx2 := dx - dx1
							tl = image.Pt(max(j-dx1, 0), max(i-dy1, 0))
							br = image.Pt(min(j+1+dx2, nApX-1), min(i+1+dy2, nApY-1))
							if aps[tl.Y*nApX+tl.X].foundCount > 0 && aps[tl.Y*nApX+br.X].foundCount > 0 &&
								aps[br.Y*nApX+tl.X].foundCount > 0 && aps[br.Y*nApX+br.X].foundCount > 0 {
								flag = false
							}
						}
					}
				}
			}
			dup := false
			for _, r := range rects {
				if r.tl == tl && r.br == br {
					dup = true
					break
				}
			}
			if !dup {
				rects = append(rects, rect{tl, br})
			}
		}
	}
	sort.SliceStable(rects, func(a, b int) bool {
		sa := (rects[a].br.X - rects[a].tl.X) * (rects[a].br.Y - rects[a].tl.Y)
		sb := (rects[b].br.X - rects[b].tl.X) * (rects[b].br.Y - rects[b].tl.Y)
		return sa > sb
	})

	width, height := symbol.sideSize.X, symbol.sideSize.Y
	matrix := newBitmap(width, height, bm.channels)

	for _, r := range rects {
		blkX := tables.APPos[vxi][r.br.X] - tables.APPos[vxi][r.tl.X] + 1
		blkY := tables.APPos[vyi][r.br.Y] - tables.APPos[vyi][r.tl.Y] + 1
		p0 := pointF{0.5, 0.5}
		p1 := pointF{float64(blkX) - 0.5, 0.5}
		p2 := pointF{float64(blkX) - 0.5, float64(blkY) - 0.5}
		p3 := pointF{0.5, float64(blkY) - 0.5}
		if r.tl.Y == 0 {
			blkY += spec.DistanceToBorder - 1
			p0.y, p1.y = 3.5, 3.5
			p2.y, p3.y = float64(blkY)-0.5, float64(blkY)-0.5
		}
		if r.br.Y == nApY-1 {
			blkY += spec.DistanceToBorder - 1
			p2.y, p3.y = float64(blkY)-3.5, float64(blkY)-3.5
		}
		if r.tl.X == 0 {
			blkX += spec.DistanceToBorder - 1
			p0.x, p3.x = 3.5, 3.5
			p1.x, p2.x = float64(blkX)-0.5, float64(blkX)-0.5
		}
		if r.br.X == nApX-1 {
			blkX += spec.DistanceToBorder - 1
			p1.x, p2.x = float64(blkX)-3.5, float64(blkX)-3.5
		}
		src := [4]pointF{p0, p1, p2, p3}
		dst := [4]pointF{
			aps[r.tl.Y*nApX+r.tl.X].center,
			aps[r.tl.Y*nApX+r.br.X].center,
			aps[r.br.Y*nApX+r.br.X].center,
			aps[r.br.Y*nApX+r.tl.X].center,
		}
		block := sampleSymbol(bm, quadToQuad(src, dst), image.Pt(blkX, blkY))
		if block == nil {
			return nil
		}
		startX := tables.APPos[vxi][r.tl.X] - 1
		startY := tables.APPos[vyi][r.tl.Y] - 1
		if r.tl.X == 0 {
			startX = 0
		}
		if r.tl.Y == 0 {
			startY = 0
		}
		for y, my := 0, startY; y < blkY && my < height; y, my = y+1, my+1 {
			for x, mx := 0, startX; x < blkX && mx < width; x, mx = x+1, mx+1 {
				mo := (my*width + mx) * matrix.channels
				bo := (y*blkX + x) * block.channels
				copy(matrix.pix[mo:mo+matrix.channels], block.pix[bo:bo+block.channels])
			}
		}
	}
	return matrix
}
