//go:build jabcode_legacy

package detect

import (
	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/palette"
)

func preV2CAPCoreColorIndex(apType int) int {
	if apType == apx {
		return 7
	}
	return 0
}

func crossCheckPatternHorizontalPreV2CAP(row []byte, channel, startx, endx, centerx, apType int, moduleSizeMax float64, moduleSize *float64) float64 {
	coreColor := int(palette.Default[preV2CAPCoreColorIndex(apType)*3+channel])
	if int(row[centerx]) != coreColor {
		return -1
	}
	var state [3]int
	var i, stateIndex int
	state[1]++
	for i = 1; centerx-i >= startx && stateIndex <= 1; i++ {
		if row[centerx-i] == row[centerx-(i-1)] {
			state[1-stateIndex]++
		} else if stateIndex > 0 && state[1-stateIndex] < 3 {
			state[1-(stateIndex-1)] += state[1-stateIndex]
			state[1-stateIndex] = 0
			stateIndex--
			state[1-stateIndex]++
		} else {
			stateIndex++
			if stateIndex > 1 {
				break
			}
			state[1-stateIndex]++
		}
	}
	if stateIndex < 1 {
		return -1
	}
	stateIndex = 0
	for i = 1; centerx+i <= endx && stateIndex <= 1; i++ {
		if row[centerx+i] == row[centerx+(i-1)] {
			state[1+stateIndex]++
		} else if stateIndex > 0 && state[1+stateIndex] < 3 {
			state[1+(stateIndex-1)] += state[1+stateIndex]
			state[1+stateIndex] = 0
			stateIndex--
			state[1+stateIndex]++
		} else {
			stateIndex++
			if stateIndex > 1 {
				break
			}
			state[1+stateIndex]++
		}
	}
	if stateIndex < 1 {
		return -1
	}
	if float64(state[1]) < moduleSizeMax && float64(state[0]) > 0.5*float64(state[1]) && float64(state[2]) > 0.5*float64(state[1]) {
		*moduleSize = float64(state[1])
		return float64(centerx+i-state[2]) - float64(state[1])/2
	}
	return -1
}

func crossCheckPatternPreV2CAP(ch [3]*core.Bitmap, y, minx, maxx, curX, apType int, maxModuleSize float64, centerx, centery, moduleSize *float64, dir *int) bool {
	var rows [3][]byte
	for channel := range rows {
		rows[channel] = ch[channel].Pix[y*ch[channel].Width : (y+1)*ch[channel].Width]
	}
	var localX, localY, horizontalSize, verticalSize [3]float64

	localX[0] = crossCheckPatternHorizontalPreV2CAP(rows[0], 0, minx, maxx, curX, apType, maxModuleSize, &horizontalSize[0])
	if localX[0] < 0 {
		return false
	}
	for channel := 1; channel < 3; channel++ {
		localX[channel] = crossCheckPatternHorizontalPreV2CAP(rows[channel], channel, minx, maxx, int(localX[0]), apType, maxModuleSize, &horizontalSize[channel])
		if localX[channel] < 0 {
			return false
		}
	}
	center := core.Pt((localX[0]+localX[1]+localX[2])/3, float64(y))
	for channel := range 3 {
		localY[channel] = crossCheckPatternVerticalAP(ch[channel], center, int(maxModuleSize), &verticalSize[channel])
		if localY[channel] < 0 {
			return false
		}
		row := ch[channel].Pix[int(localY[channel])*ch[channel].Width : (int(localY[channel])+1)*ch[channel].Width]
		localX[channel] = crossCheckPatternHorizontalPreV2CAP(row, channel, minx, maxx, int(center.X), apType, maxModuleSize, &horizontalSize[channel])
		if localX[channel] < 0 {
			return false
		}
	}

	*moduleSize = (horizontalSize[0] + horizontalSize[1] + horizontalSize[2] + verticalSize[0] + verticalSize[1] + verticalSize[2]) / 6
	*centerx = (localX[0] + localX[1] + localX[2]) / 3
	*centery = (localY[0] + localY[1] + localY[2]) / 3
	center.X, center.Y = *centerx, *centery

	var localDir [3]int
	for channel := range 3 {
		if crossCheckPatternDiagonalAP(ch[channel], apType, int(*moduleSize*2), center, &localDir[channel]) < 0 {
			return false
		}
	}
	if localDir[0]+localDir[1]+localDir[2] > 0 {
		*dir = 1
	} else {
		*dir = -1
	}
	return true
}

func findPreV2CAlignmentPattern(ch [3]*core.Bitmap, x, y, moduleSize float64, apType int) FinderPattern {
	coreColorR := byte(palette.Default[preV2CAPCoreColorIndex(apType)*3])
	radius := int(4 * moduleSize)
	radiusMax := 4 * radius
	for ; radius < radiusMax; radius <<= 1 {
		aps := make([]FinderPattern, maxFinderPatterns)
		startx := max(0, int(x)-radius)
		starty := max(0, int(y)-radius)
		endx := min(ch[0].Width-1, int(x)+radius)
		endy := min(ch[0].Height-1, int(y)+radius)
		if float64(endx-startx) < 3*moduleSize || float64(endy-starty) < 3*moduleSize {
			continue
		}
		counter := 0
		for k := starty; k < endy; k++ {
			kk := k - starty
			i := int(y) + (kk+1)/2
			if kk&1 != 0 {
				i = int(y) - (kk+1)/2
			}
			if i < starty || i > endy {
				continue
			}
			rowR := ch[0].Pix[i*ch[0].Width : (i+1)*ch[0].Width]
			var apModuleSize, centerx, centery float64
			var apDir int
			apFound := false
			dir := -1
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
					apFound = crossCheckPatternPreV2CAP(ch, i, startx, endx, leftTmpX, apType, moduleSize*2, &centerx, &centery, &apModuleSize, &apDir)
					for rowR[leftTmpX] == coreColorR && leftTmpX > startx {
						leftTmpX--
					}
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
					apFound = crossCheckPatternPreV2CAP(ch, i, startx, endx, rightTmpX, apType, moduleSize*2, &centerx, &centery, &apModuleSize, &apDir)
					for rowR[rightTmpX] == coreColorR && rightTmpX < endx {
						rightTmpX++
					}
				}
				dir = -dir
			}
			if !apFound {
				continue
			}
			ap := FinderPattern{Typ: apType, FoundCount: 1, ModuleSize: apModuleSize, Center: core.Pt(centerx, centery), direction: apDir}
			if index := saveAlignmentPattern(&ap, aps, &counter); index >= 0 {
				return aps[index]
			}
		}
	}
	return FinderPattern{Typ: -1}
}

func findSecondaryAlignmentPattern(ch [3]*core.Bitmap, x, y, moduleSize float64, apType int, family secondaryPatternFamily) FinderPattern {
	if family == secondaryPatternPreV2C {
		return findPreV2CAlignmentPattern(ch, x, y, moduleSize, apType)
	}
	return findAlignmentPattern(ch, x, y, moduleSize, apType)
}

// DetectPreV2CSecondary finds and samples a JAB Code secondary symbol emitted
// by the pre-v2.0 C reference implementation, whose alignment patterns use
// monochrome cores.
func DetectPreV2CSecondary(bm *core.Bitmap, ch [3]*core.Bitmap, host, secondary *core.DecodedSymbol, dockedPosition int) *core.Bitmap {
	return detectSecondary(bm, ch, host, secondary, dockedPosition, secondaryPatternPreV2C)
}
