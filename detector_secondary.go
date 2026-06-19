package jabcode

import (
	"image"
	"math"
)

// findSecondarySymbol locates a secondary symbol docked to the given side of a
// host symbol by detecting its four corner alignment patterns
// (findSlaveSymbol in detector.c).
func findSecondarySymbol(bm *bitmap, ch [3]*bitmap, host, secondary *decodedSymbol, dockedPosition int) bool {
	var aps [4]finderPattern

	secondary.sideSize = image.Pt(version2size(secondary.meta.sideVersion.X), version2size(secondary.meta.sideVersion.Y))

	hp := host.patternPositions
	distx01, disty01 := hp[1].x-hp[0].x, hp[1].y-hp[0].y
	distx32, disty32 := hp[2].x-hp[3].x, hp[2].y-hp[3].y
	distx03, disty03 := hp[3].x-hp[0].x, hp[3].y-hp[0].y
	distx12, disty12 := hp[2].x-hp[1].x, hp[2].y-hp[1].y

	var alpha1, alpha2 float64
	sign := 1
	var dockedSideSize, undockedSideSize int
	var t1, t2, t3, t4, h1, h2 int

	switch dockedPosition {
	case 3:
		alpha1, alpha2, sign = math.Atan2(disty01, distx01), math.Atan2(disty32, distx32), 1
		dockedSideSize, undockedSideSize = secondary.sideSize.Y, secondary.sideSize.X
		t1, t2, t3, t4, h1, h2 = ap0, ap3, ap1, ap2, fp1, fp2
		secondary.hostPosition = 2
	case 2:
		alpha1, alpha2, sign = math.Atan2(disty32, distx32), math.Atan2(disty01, distx01), -1
		dockedSideSize, undockedSideSize = secondary.sideSize.Y, secondary.sideSize.X
		t1, t2, t3, t4, h1, h2 = ap2, ap1, ap3, ap0, fp3, fp0
		secondary.hostPosition = 3
	case 1:
		alpha1, alpha2, sign = math.Atan2(disty12, distx12), math.Atan2(disty03, distx03), 1
		dockedSideSize, undockedSideSize = secondary.sideSize.X, secondary.sideSize.Y
		t1, t2, t3, t4, h1, h2 = ap1, ap0, ap2, ap3, fp2, fp3
		secondary.hostPosition = 0
	case 0:
		alpha1, alpha2, sign = math.Atan2(disty03, distx03), math.Atan2(disty12, distx12), -1
		dockedSideSize, undockedSideSize = secondary.sideSize.X, secondary.sideSize.Y
		t1, t2, t3, t4, h1, h2 = ap3, ap2, ap0, ap1, fp0, fp1
		secondary.hostPosition = 1
	}
	signf := float64(sign)

	aps[t1].center.x = hp[h1].x + signf*7*host.moduleSize*math.Cos(alpha1)
	aps[t1].center.y = hp[h1].y + signf*7*host.moduleSize*math.Sin(alpha1)
	aps[t1] = findAlignmentPattern(ch, aps[t1].center.x, aps[t1].center.y, host.moduleSize, t1)
	if aps[t1].foundCount == 0 {
		return false
	}
	aps[t2].center.x = hp[h2].x + signf*7*host.moduleSize*math.Cos(alpha2)
	aps[t2].center.y = hp[h2].y + signf*7*host.moduleSize*math.Sin(alpha2)
	aps[t2] = findAlignmentPattern(ch, aps[t2].center.x, aps[t2].center.y, host.moduleSize, t2)
	if aps[t2].foundCount == 0 {
		return false
	}

	secondary.moduleSize = math.Hypot(aps[t1].center.x-aps[t2].center.x, aps[t1].center.y-aps[t2].center.y) / float64(dockedSideSize-7)

	aps[t3].center.x = aps[t1].center.x + signf*float64(undockedSideSize-7)*secondary.moduleSize*math.Cos(alpha1)
	aps[t3].center.y = aps[t1].center.y + signf*float64(undockedSideSize-7)*secondary.moduleSize*math.Sin(alpha1)
	aps[t3] = findAlignmentPattern(ch, aps[t3].center.x, aps[t3].center.y, secondary.moduleSize, t3)
	aps[t4].center.x = aps[t2].center.x + signf*float64(undockedSideSize-7)*secondary.moduleSize*math.Cos(alpha2)
	aps[t4].center.y = aps[t2].center.y + signf*float64(undockedSideSize-7)*secondary.moduleSize*math.Sin(alpha2)
	aps[t4] = findAlignmentPattern(ch, aps[t4].center.x, aps[t4].center.y, secondary.moduleSize, t4)

	if aps[t3].foundCount == 0 && aps[t4].foundCount == 0 {
		return false
	}
	if aps[t3].foundCount == 0 {
		avg24 := (aps[t2].moduleSize + aps[t4].moduleSize) / 2.0
		avg14 := (aps[t1].moduleSize + aps[t4].moduleSize) / 2.0
		aps[t3].center.x = (aps[t4].center.x-aps[t2].center.x)/avg24*avg14 + aps[t1].center.x
		aps[t3].center.y = (aps[t4].center.y-aps[t2].center.y)/avg24*avg14 + aps[t1].center.y
		aps[t3].typ, aps[t3].foundCount = t3, 1
		aps[t3].moduleSize = (aps[t1].moduleSize + aps[t2].moduleSize + aps[t4].moduleSize) / 3.0
		if aps[t3].center.x > float64(bm.width-1) || aps[t3].center.y > float64(bm.height-1) {
			return false
		}
	}
	if aps[t4].foundCount == 0 {
		avg13 := (aps[t1].moduleSize + aps[t3].moduleSize) / 2.0
		avg23 := (aps[t2].moduleSize + aps[t3].moduleSize) / 2.0
		aps[t4].center.x = (aps[t3].center.x-aps[t1].center.x)/avg13*avg23 + aps[t2].center.x
		aps[t4].center.y = (aps[t3].center.y-aps[t1].center.y)/avg13*avg23 + aps[t2].center.y
		aps[t4].typ, aps[t4].foundCount = t4, 1
		// Note: the reference averages ap1 twice here; kept identical.
		aps[t4].moduleSize = (aps[t1].moduleSize + aps[t1].moduleSize + aps[t3].moduleSize) / 3.0
		if aps[t4].center.x > float64(bm.width-1) || aps[t4].center.y > float64(bm.height-1) {
			return false
		}
	}

	secondary.patternPositions[t1] = aps[t1].center
	secondary.patternPositions[t2] = aps[t2].center
	secondary.patternPositions[t3] = aps[t3].center
	secondary.patternPositions[t4] = aps[t4].center
	secondary.moduleSize = (aps[t1].moduleSize + aps[t2].moduleSize + aps[t3].moduleSize + aps[t4].moduleSize) / 4.0
	return true
}

// detectSecondary finds and samples a secondary symbol docked at the given
// position of a host symbol (detectSlave in detector.c).
func detectSecondary(bm *bitmap, ch [3]*bitmap, host, secondary *decodedSymbol, dockedPosition int) *bitmap {
	if dockedPosition < 0 || dockedPosition > 3 {
		return nil
	}
	if !findSecondarySymbol(bm, ch, host, secondary, dockedPosition) {
		return nil
	}
	pt := getPerspectiveTransform(secondary.patternPositions[0], secondary.patternPositions[1],
		secondary.patternPositions[2], secondary.patternPositions[3], secondary.sideSize)
	return sampleSymbol(bm, pt, secondary.sideSize)
}

// decodeDockedSecondaries detects and decodes every secondary symbol docked to a
// host symbol (decodeDockedSlaves in detector.c).
func decodeDockedSecondaries(bm *bitmap, ch [3]*bitmap, symbols []decodedSymbol, hostIndex int, total *int) bool {
	dp := symbols[hostIndex].meta.dockedPosition
	docked := [4]int{dp & 0x08, dp & 0x04, dp & 0x02, dp & 0x01}
	for j := range 4 {
		if docked[j] > 0 && *total < maxSymbolNumber {
			symbols[*total].index = *total
			symbols[*total].hostIndex = hostIndex
			symbols[*total].meta = symbols[hostIndex].secondaryMeta[j]
			matrix := detectSecondary(bm, ch, &symbols[hostIndex], &symbols[*total], j)
			if matrix == nil {
				return false
			}
			if decodeSecondary(matrix, &symbols[*total]) > 0 {
				*total++
			} else {
				return false
			}
		}
	}
	return true
}
