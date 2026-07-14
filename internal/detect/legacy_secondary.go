//go:build jabcode_legacy

package detect

import (
	"image"
	"math"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/spec"
)

func legacyAPCoreColorIndex(apType int) int {
	if apType == apx {
		return 7
	}
	return 0
}

func crossCheckPatternHorizontalLegacyAP(row []byte, channel, startx, endx, centerx, apType int, moduleSizeMax float64, moduleSize *float64) float64 {
	coreColor := int(palette.Default[legacyAPCoreColorIndex(apType)*3+channel])
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

func crossCheckPatternLegacyAP(ch [3]*core.Bitmap, y, minx, maxx, curX, apType int, maxModuleSize float64, centerx, centery, moduleSize *float64, dir *int) bool {
	var rows [3][]byte
	for channel := range rows {
		rows[channel] = ch[channel].Pix[y*ch[channel].Width : (y+1)*ch[channel].Width]
	}
	var localX, localY, horizontalSize, verticalSize [3]float64

	localX[0] = crossCheckPatternHorizontalLegacyAP(rows[0], 0, minx, maxx, curX, apType, maxModuleSize, &horizontalSize[0])
	if localX[0] < 0 {
		return false
	}
	for channel := 1; channel < 3; channel++ {
		localX[channel] = crossCheckPatternHorizontalLegacyAP(rows[channel], channel, minx, maxx, int(localX[0]), apType, maxModuleSize, &horizontalSize[channel])
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
		localX[channel] = crossCheckPatternHorizontalLegacyAP(row, channel, minx, maxx, int(center.X), apType, maxModuleSize, &horizontalSize[channel])
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

func findLegacyAlignmentPattern(ch [3]*core.Bitmap, x, y, moduleSize float64, apType int) FinderPattern {
	coreColorR := byte(palette.Default[legacyAPCoreColorIndex(apType)*3])
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
					apFound = crossCheckPatternLegacyAP(ch, i, startx, endx, leftTmpX, apType, moduleSize*2, &centerx, &centery, &apModuleSize, &apDir)
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
					apFound = crossCheckPatternLegacyAP(ch, i, startx, endx, rightTmpX, apType, moduleSize*2, &centerx, &centery, &apModuleSize, &apDir)
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

func findLegacySecondarySymbol(bm *core.Bitmap, ch [3]*core.Bitmap, host, secondary *core.DecodedSymbol, dockedPosition int) bool {
	var aps [4]FinderPattern
	secondary.SideSize = image.Pt(spec.VersionToSize(secondary.Meta.SideVersion.X), spec.VersionToSize(secondary.Meta.SideVersion.Y))
	hp := host.PatternPositions
	distx01, disty01 := hp[1].X-hp[0].X, hp[1].Y-hp[0].Y
	distx32, disty32 := hp[2].X-hp[3].X, hp[2].Y-hp[3].Y
	distx03, disty03 := hp[3].X-hp[0].X, hp[3].Y-hp[0].Y
	distx12, disty12 := hp[2].X-hp[1].X, hp[2].Y-hp[1].Y

	var alpha1, alpha2 float64
	sign := 1
	var dockedSideSize, undockedSideSize int
	var t1, t2, t3, t4, h1, h2 int
	switch dockedPosition {
	case 3:
		alpha1, alpha2, sign = math.Atan2(disty01, distx01), math.Atan2(disty32, distx32), 1
		dockedSideSize, undockedSideSize = secondary.SideSize.Y, secondary.SideSize.X
		t1, t2, t3, t4, h1, h2 = ap0, ap3, ap1, ap2, fp1, fp2
		secondary.HostPosition = 2
	case 2:
		alpha1, alpha2, sign = math.Atan2(disty32, distx32), math.Atan2(disty01, distx01), -1
		dockedSideSize, undockedSideSize = secondary.SideSize.Y, secondary.SideSize.X
		t1, t2, t3, t4, h1, h2 = ap2, ap1, ap3, ap0, fp3, fp0
		secondary.HostPosition = 3
	case 1:
		alpha1, alpha2, sign = math.Atan2(disty12, distx12), math.Atan2(disty03, distx03), 1
		dockedSideSize, undockedSideSize = secondary.SideSize.X, secondary.SideSize.Y
		t1, t2, t3, t4, h1, h2 = ap1, ap0, ap2, ap3, fp2, fp3
		secondary.HostPosition = 0
	case 0:
		alpha1, alpha2, sign = math.Atan2(disty03, distx03), math.Atan2(disty12, distx12), -1
		dockedSideSize, undockedSideSize = secondary.SideSize.X, secondary.SideSize.Y
		t1, t2, t3, t4, h1, h2 = ap3, ap2, ap0, ap1, fp0, fp1
		secondary.HostPosition = 1
	default:
		return false
	}
	signf := float64(sign)
	aps[t1].Center = core.Pt(
		hp[h1].X+signf*7*host.ModuleSize*math.Cos(alpha1),
		hp[h1].Y+signf*7*host.ModuleSize*math.Sin(alpha1),
	)
	aps[t1] = findLegacyAlignmentPattern(ch, aps[t1].Center.X, aps[t1].Center.Y, host.ModuleSize, t1)
	if aps[t1].FoundCount == 0 {
		return false
	}
	aps[t2].Center = core.Pt(
		hp[h2].X+signf*7*host.ModuleSize*math.Cos(alpha2),
		hp[h2].Y+signf*7*host.ModuleSize*math.Sin(alpha2),
	)
	aps[t2] = findLegacyAlignmentPattern(ch, aps[t2].Center.X, aps[t2].Center.Y, host.ModuleSize, t2)
	if aps[t2].FoundCount == 0 {
		return false
	}

	secondary.ModuleSize = math.Hypot(aps[t1].Center.X-aps[t2].Center.X, aps[t1].Center.Y-aps[t2].Center.Y) / float64(dockedSideSize-7)
	aps[t3].Center = core.Pt(
		aps[t1].Center.X+signf*float64(undockedSideSize-7)*secondary.ModuleSize*math.Cos(alpha1),
		aps[t1].Center.Y+signf*float64(undockedSideSize-7)*secondary.ModuleSize*math.Sin(alpha1),
	)
	aps[t3] = findLegacyAlignmentPattern(ch, aps[t3].Center.X, aps[t3].Center.Y, secondary.ModuleSize, t3)
	aps[t4].Center = core.Pt(
		aps[t2].Center.X+signf*float64(undockedSideSize-7)*secondary.ModuleSize*math.Cos(alpha2),
		aps[t2].Center.Y+signf*float64(undockedSideSize-7)*secondary.ModuleSize*math.Sin(alpha2),
	)
	aps[t4] = findLegacyAlignmentPattern(ch, aps[t4].Center.X, aps[t4].Center.Y, secondary.ModuleSize, t4)

	if aps[t3].FoundCount == 0 && aps[t4].FoundCount == 0 {
		return false
	}
	if aps[t3].FoundCount == 0 {
		avg24 := (aps[t2].ModuleSize + aps[t4].ModuleSize) / 2
		avg14 := (aps[t1].ModuleSize + aps[t4].ModuleSize) / 2
		aps[t3].Center.X = (aps[t4].Center.X-aps[t2].Center.X)/avg24*avg14 + aps[t1].Center.X
		aps[t3].Center.Y = (aps[t4].Center.Y-aps[t2].Center.Y)/avg24*avg14 + aps[t1].Center.Y
		aps[t3].Typ, aps[t3].FoundCount = t3, 1
		aps[t3].ModuleSize = (aps[t1].ModuleSize + aps[t2].ModuleSize + aps[t4].ModuleSize) / 3
		if aps[t3].Center.X < 0 || aps[t3].Center.Y < 0 || aps[t3].Center.X > float64(bm.Width-1) || aps[t3].Center.Y > float64(bm.Height-1) {
			return false
		}
	}
	if aps[t4].FoundCount == 0 {
		avg13 := (aps[t1].ModuleSize + aps[t3].ModuleSize) / 2
		avg23 := (aps[t2].ModuleSize + aps[t3].ModuleSize) / 2
		aps[t4].Center.X = (aps[t3].Center.X-aps[t1].Center.X)/avg13*avg23 + aps[t2].Center.X
		aps[t4].Center.Y = (aps[t3].Center.Y-aps[t1].Center.Y)/avg13*avg23 + aps[t2].Center.Y
		aps[t4].Typ, aps[t4].FoundCount = t4, 1
		aps[t4].ModuleSize = (aps[t1].ModuleSize + aps[t1].ModuleSize + aps[t3].ModuleSize) / 3
		if aps[t4].Center.X < 0 || aps[t4].Center.Y < 0 || aps[t4].Center.X > float64(bm.Width-1) || aps[t4].Center.Y > float64(bm.Height-1) {
			return false
		}
	}

	secondary.PatternPositions[t1] = aps[t1].Center
	secondary.PatternPositions[t2] = aps[t2].Center
	secondary.PatternPositions[t3] = aps[t3].Center
	secondary.PatternPositions[t4] = aps[t4].Center
	secondary.ModuleSize = (aps[t1].ModuleSize + aps[t2].ModuleSize + aps[t3].ModuleSize + aps[t4].ModuleSize) / 4
	return true
}

// DetectLegacySecondary finds and samples a JAB Code secondary symbol emitted
// by the pre-v2.0 C reference implementation, whose alignment patterns use
// monochrome cores.
func DetectLegacySecondary(bm *core.Bitmap, ch [3]*core.Bitmap, host, secondary *core.DecodedSymbol, dockedPosition int) *core.Bitmap {
	if !findLegacySecondarySymbol(bm, ch, host, secondary, dockedPosition) {
		return nil
	}
	pt := core.PerspectiveTransform(
		secondary.PatternPositions[0], secondary.PatternPositions[1],
		secondary.PatternPositions[2], secondary.PatternPositions[3], secondary.SideSize,
	)
	return SampleSymbol(bm, pt, secondary.SideSize)
}
