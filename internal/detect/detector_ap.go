package detect

import (
	"image"
	"math"
	"sort"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/tables"
)

// Alignment-pattern types. AP0..AP3 share core color index 3 (cyan); APX uses
// index 6 (yellow).
//
// A JAB alignment pattern is NOT the concentric QR alignment pattern. Per
// ISO/IEC 23634:2022 4.3.8 and Figure 7 it has the same construction as a
// finder, one layer shallower: two equal 2x2 square references joined at a
// single overlapping core module, laid out along a diagonal. Around the core
// only the two same-sign quadrants are pattern; the other two are data. So it
// carries the finder's order-2 point symmetry rather than a quarter-turn
// symmetry, and its four spec types fall into two diagonal classes that a
// quarter turn exchanges rather than fixes (X0 against X1, U against L).
//
// Note APX covers both X0 and X1: the encoder alternates them across the
// interior grid, so a single type here spans both diagonal classes and the
// cross-check has to resolve the class per hit rather than per type.
//
// Being smaller makes it more rotation-tolerant than a finder, not less. A scan
// line through the core drifts off the core row by |tan(phi)| at the outermost
// layer, which sits at 1 module rather than the finder's 2, so the drift stays
// under half a module out to 26.6 degrees and holds the core row through that
// module's far edge to 18.4 degrees, against 14.0 and 11.3 for a finder.
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
func saveAlignmentPattern(ap *FinderPattern, aps []FinderPattern, counter *int) int {
	// Ports saveAlignmentPattern in detector.c.
	for i := 0; i < *counter; i++ {
		if aps[i].FoundCount > 0 &&
			math.Abs(ap.Center.X-aps[i].Center.X) <= ap.ModuleSize && math.Abs(ap.Center.Y-aps[i].Center.Y) <= ap.ModuleSize &&
			(math.Abs(ap.ModuleSize-aps[i].ModuleSize) <= aps[i].ModuleSize || math.Abs(ap.ModuleSize-aps[i].ModuleSize) <= 1.0) &&
			ap.Typ == aps[i].Typ {
			fc := float64(aps[i].FoundCount)
			aps[i].Center.X = (fc*aps[i].Center.X + ap.Center.X) / (fc + 1)
			aps[i].Center.Y = (fc*aps[i].Center.Y + ap.Center.Y) / (fc + 1)
			aps[i].ModuleSize = (fc*aps[i].ModuleSize + ap.ModuleSize) / (fc + 1)
			aps[i].FoundCount++
			return i
		}
	}
	aps[*counter] = *ap
	*counter++
	return -1
}

// crossCheckPatternDiagonalAP validates an alignment pattern along a diagonal,
// returning the refined center y or -1.
func crossCheckPatternDiagonalAP(image *core.Bitmap, apType, moduleSizeMax int, center core.PointF, dir *int) float64 {
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
		startx := int(center.X)
		starty := int(center.Y)

		sc[1]++
		for i = 1; i <= starty && i <= startx && si <= 1; i++ {
			if image.Pixel(startx+i*offsetX, starty+i*offsetY) == image.Pixel(startx+(i-1)*offsetX, starty+(i-1)*offsetY) {
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
			for i = 1; starty+i < image.Height && startx+i < image.Width && si <= 1; i++ {
				if image.Pixel(startx-i*offsetX, starty-i*offsetY) == image.Pixel(startx-(i-1)*offsetX, starty-(i-1)*offsetY) {
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
func crossCheckPatternVerticalAP(image *core.Bitmap, center core.PointF, moduleSizeMax int, moduleSize *float64) float64 {
	// Ports crossCheckPatternVerticalAP in detector.c.
	var sc [3]int
	cx, cy := int(center.X), int(center.Y)
	var i, si int
	sc[1]++
	for i = 1; i <= cy && si <= 1; i++ {
		if image.Pixel(cx, cy-i) == image.Pixel(cx, cy-(i-1)) {
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
	for i = 1; cy+i < image.Height && si <= 1; i++ {
		if image.Pixel(cx, cy+i) == image.Pixel(cx, cy+(i-1)) {
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
func crossCheckPatternHorizontalAP(pixel func(int) byte, channel, startx, endx, centerx, apType int, moduleSizeMax float64, moduleSize *float64) float64 {
	// Ports crossCheckPatternHorizontalAP in detector.c.
	coreColor := int(palette.Default[apCoreColorIndex(apType)*3+channel])
	if int(pixel(centerx)) != coreColor {
		return -1
	}
	var sc [3]int
	var i, si int
	sc[1]++
	// Each step re-reads the pixel the previous step fetched, and here a read
	// is an indirect call through the accessor rather than a byte load, so
	// carrying it forward halves the calls.
	prev := pixel(centerx)
	for i = 1; centerx-i >= startx && si <= 1; i++ {
		cur := pixel(centerx - i)
		if cur == prev {
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
		prev = cur
	}
	if si < 1 {
		return -1
	}
	si = 0
	prev = pixel(centerx)
	for i = 1; centerx+i <= endx && si <= 1; i++ {
		cur := pixel(centerx + i)
		if cur == prev {
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
		prev = cur
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
func crossCheckPatternAP(ch [3]*core.Bitmap, y, minx, maxx, curX, apType int, maxModuleSize float64, centerx, centery, moduleSize *float64, dir *int) bool {
	// Ports crossCheckPatternAP in detector.c.
	rowR := func(x int) byte { return ch[0].Pixel(x, y) }
	rowB := func(x int) byte { return ch[2].Pixel(x, y) }
	var lcx, lcy, lmsH, lmsV [3]float64

	lcx[0] = crossCheckPatternHorizontalAP(rowR, 0, minx, maxx, curX, apType, maxModuleSize, &lmsH[0])
	if lcx[0] < 0 {
		return false
	}
	lcx[2] = crossCheckPatternHorizontalAP(rowB, 2, minx, maxx, int(lcx[0]), apType, maxModuleSize, &lmsH[2])
	if lcx[2] < 0 {
		return false
	}
	center := core.Pt((lcx[0]+lcx[2])/2.0, float64(y))
	*moduleSize = (lmsH[0] + lmsH[2]) / 2.0
	greenCore := int(palette.Default[apCoreColorIndex(apType)*3+1])
	if !crossCheckColor(ch[1], greenCore, int(*moduleSize), 3, int(center.X), int(center.Y), 0, 3) {
		return false
	}

	lcy[0] = crossCheckPatternVerticalAP(ch[0], center, int(maxModuleSize), &lmsV[0])
	if lcy[0] < 0 {
		return false
	}
	rowR = func(x int) byte { return ch[0].Pixel(x, int(lcy[0])) }
	lcx[0] = crossCheckPatternHorizontalAP(rowR, 0, minx, maxx, int(center.X), apType, maxModuleSize, &lmsH[0])
	if lcx[0] < 0 {
		return false
	}

	lcy[2] = crossCheckPatternVerticalAP(ch[2], center, int(maxModuleSize), &lmsV[2])
	if lcy[2] < 0 {
		return false
	}
	rowB = func(x int) byte { return ch[2].Pixel(x, int(lcy[2])) }
	lcx[2] = crossCheckPatternHorizontalAP(rowB, 2, minx, maxx, int(center.X), apType, maxModuleSize, &lmsH[2])
	if lcx[2] < 0 {
		return false
	}

	*moduleSize = (lmsH[0] + lmsH[2] + lmsV[0] + lmsV[2]) / 4.0
	*centerx = (lcx[0] + lcx[2]) / 2.0
	*centery = (lcy[0] + lcy[2]) / 2.0
	center.X, center.Y = *centerx, *centery
	if !crossCheckColor(ch[1], greenCore, int(*moduleSize), 3, int(center.X), int(center.Y), 1, 3) {
		return false
	}

	var ldir [3]int
	if crossCheckPatternDiagonalAP(ch[0], apType, int(*moduleSize*2), center, &ldir[0]) < 0 {
		return false
	}
	if crossCheckPatternDiagonalAP(ch[2], apType, int(*moduleSize*2), center, &ldir[2]) < 0 {
		return false
	}
	if !crossCheckColor(ch[1], greenCore, int(*moduleSize), 3, int(center.X), int(center.Y), 2, 3) {
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
// (x, y), with the symbol's module axes at that position supplied by the
// caller. Every caller knows them: they are the directions it extrapolated the
// prediction along, so a rotated or projectively distorted symbol costs nothing
// extra to search.
func findAlignmentPattern(ch [3]*core.Bitmap, x, y, moduleSize float64, apType int, b apBasis) FinderPattern {
	return findAlignmentPatternBasis(ch, x, y, moduleSize, apType, b)
}

// findAlignmentPatternRows searches along image rows only. Retained as the
// comparison oracle for the directional locator at an upright basis, where the
// two must agree.
func findAlignmentPatternRows(ch [3]*core.Bitmap, x, y, moduleSize float64, apType int) FinderPattern {
	// Ports findAlignmentPattern in detector.c.
	coreColorR := byte(palette.Default[apCoreColorIndex(apType)*3])
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
			var i int
			if kk&0x01 == 0 {
				i = int(y) + (kk+1)/2
			} else {
				i = int(y) - (kk+1)/2
			}
			if i < starty || i > endy {
				continue
			}
			rowR := func(x int) byte { return ch[0].Pixel(x, i) }

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
					for rowR(leftTmpX) != coreColorR && leftTmpX > startx {
						leftTmpX--
					}
					if leftTmpX <= startx {
						dir = -dir
						continue
					}
					apFound = crossCheckPatternAP(ch, i, startx, endx, leftTmpX, apType, moduleSize*2, &centerx, &centery, &apModuleSize, &apDir)
					for rowR(leftTmpX) == coreColorR && leftTmpX > startx {
						leftTmpX--
					}
					dir = -dir
				} else {
					for rowR(rightTmpX) == coreColorR && rightTmpX < endx {
						rightTmpX++
					}
					for rowR(rightTmpX) != coreColorR && rightTmpX < endx {
						rightTmpX++
					}
					if rightTmpX >= endx {
						dir = -dir
						continue
					}
					apFound = crossCheckPatternAP(ch, i, startx, endx, rightTmpX, apType, moduleSize*2, &centerx, &centery, &apModuleSize, &apDir)
					for rowR(rightTmpX) == coreColorR && rightTmpX < endx {
						rightTmpX++
					}
					dir = -dir
				}
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

// lerpEdge interpolates between two opposite edges of the finder quad, a0 to a1
// and b0 to b1, at fractional position t across the symbol.
func lerpEdge(a0, a1, b0, b1 core.PointF, t float64) (x, y float64) {
	ax, ay := a1.X-a0.X, a1.Y-a0.Y
	bx, by := b1.X-b0.X, b1.Y-b0.Y
	return ax + (bx-ax)*t, ay + (by-ay)*t
}

// firstAPPos rounds a raw module count to the nearest valid first-AP position.
func firstAPPos(pos int) int {
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

// apWalkVersions lists the candidate side versions the confirmation visits, in
// the order it visits them: the sampled version first, then alternating outward
// so a correct sample is confirmed immediately and a near miss is reached before
// a far one.
//
// The order is load-bearing and is why this is separated from the search rather
// than folded into it. The walk stops at the first candidate that both finds a
// pattern and passes the position check, so a search that reported its best
// match instead of the caller's first acceptance would answer a different
// question.
func apWalkVersions(sideVersion int) []int {
	// Ports the version walk in detectFirstAP in detector.c.
	versions := make([]int, 0, 5)
	nextVersion := sideVersion
	dir := 1
	up, down := 0, 0
	for {
		versions = append(versions, nextVersion)
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
			return versions
		}
	}
}

// apWalkCenter is where a candidate version puts the first alignment pattern:
// along the edge from fp1 towards fp2, at the module distance that version's
// first alignment column sits from the finder's own.
func apWalkCenter(fp1 FinderPattern, alpha float64, sideVersion int) core.PointF {
	distance := fp1.ModuleSize * float64(tables.APPos[sideVersion-1][1]-tables.APPos[sideVersion-1][0])
	return core.Pt(
		fp1.Center.X+distance*math.Cos(alpha),
		fp1.Center.Y+distance*math.Sin(alpha),
	)
}

// detectFirstAP detects the first alignment pattern between two finder patterns,
// returning its position. b carries the module axes along the searched edge and
// across it, so the search follows the symbol rather than the image.
func detectFirstAP(ch [3]*core.Bitmap, sideVersion int, fp1, fp2 FinderPattern, b apBasis, locate alignmentPositionLocator) int {
	// Ports detectFirstAP in detector.c.
	alpha := math.Atan2(fp2.Center.Y-fp1.Center.Y, fp2.Center.X-fp1.Center.X)
	versions := apWalkVersions(sideVersion)
	if locate != nil {
		if pos, ok := detectFirstAPOnDevice(locate, versions, fp1, alpha, b); ok {
			return pos
		}
	}
	for _, version := range versions {
		centre := apWalkCenter(fp1, alpha, version)
		ap := findAlignmentPattern(ch, centre.X, centre.Y, fp1.ModuleSize, apx, b)
		if ap.FoundCount > 0 {
			if pos := firstAPPos(4 + CalculateModuleNumber(fp1, ap)); pos > 0 {
				return pos
			}
		}
	}
	return core.Failure
}

// detectFirstAPOnDevice answers every candidate version in one batch and then
// applies the walk's own order to the results. It reports false when the device
// declined, leaving the caller to walk the masks itself.
func detectFirstAPOnDevice(
	locate alignmentPositionLocator,
	versions []int,
	fp1 FinderPattern,
	alpha float64,
	b apBasis,
) (int, bool) {
	ux, uy := b.u.unit()
	vx, vy := b.v.unit()
	candidates := make([]alignmentCandidate, len(versions))
	for at, version := range versions {
		candidates[at] = alignmentCandidate{
			Center:     apWalkCenter(fp1, alpha, version),
			ModuleSize: fp1.ModuleSize,
			// The ceiling the host walk hands its cross-check, so both arms
			// reject the same over-long core runs.
			ModuleMax: fp1.ModuleSize * 2,
			U:         core.Pt(ux, uy),
			V:         core.Pt(vx, vy),
		}
	}
	found, err := locate(apx, candidates)
	if err != nil || len(found) != len(candidates) {
		return 0, false
	}
	for _, ap := range found {
		if ap.FoundCount == 0 {
			continue
		}
		if pos := firstAPPos(4 + CalculateModuleNumber(fp1, ap)); pos > 0 {
			return pos, true
		}
	}
	return core.Failure, true
}

// confirmSideVersion confirms a side version from the first AP position.
func confirmSideVersion(sideVersion, firstAPPos int) int {
	// Ports confirmSideVersion in detector.c.
	if firstAPPos <= 0 {
		return core.Failure
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
	return core.Failure
}

// confirmSymbolSize confirms the symbol's side sizes using alignment patterns.
func confirmSymbolSize(ch [3]*core.Bitmap, fps []FinderPattern, symbol *core.DecodedSymbol, locate alignmentPositionLocator) bool {
	// Ports confirmSymbolSize in detector.c.
	// The quad's own edges are the module axes: fp0 to fp1 runs along x, fp0 to
	// fp3 along y. They are neither perpendicular nor equally scaled in image
	// space once the capture has perspective, which is why both are measured
	// and each is paired with the module count it spans. A finder-centre edge
	// spans the side less the seven modules the two finders occupy.
	spanX := float64(symbol.SideSize.X - 7)
	spanY := float64(symbol.SideSize.Y - 7)
	acrossX, acrossY := fps[3].Center.X-fps[0].Center.X, fps[3].Center.Y-fps[0].Center.Y
	bTop, okTop := newAPBasis(fps[1].Center.X-fps[0].Center.X, fps[1].Center.Y-fps[0].Center.Y, spanX, acrossX, acrossY, spanY)
	bBottom, okBottom := newAPBasis(fps[2].Center.X-fps[3].Center.X, fps[2].Center.Y-fps[3].Center.Y, spanX, acrossX, acrossY, spanY)
	if !okTop || !okBottom {
		return false
	}
	pos := detectFirstAP(ch, symbol.Meta.SideVersion.X, fps[0], fps[1], bTop, locate)
	vx := confirmSideVersion(symbol.Meta.SideVersion.X, pos)
	if vx == 0 {
		pos = detectFirstAP(ch, symbol.Meta.SideVersion.X, fps[3], fps[2], bBottom, locate)
		vx = confirmSideVersion(symbol.Meta.SideVersion.X, pos)
		if vx == 0 {
			return false
		}
	}
	symbol.Meta.SideVersion.X = vx
	symbol.SideSize.X = spec.VersionToSize(vx)

	// The X side is confirmed by now, so the along axis spans a known count.
	spanX = float64(symbol.SideSize.X - 7)
	alongX, alongY := fps[1].Center.X-fps[0].Center.X, fps[1].Center.Y-fps[0].Center.Y
	bLeft, okLeft := newAPBasis(acrossX, acrossY, spanY, alongX, alongY, spanX)
	bRight, okRight := newAPBasis(fps[2].Center.X-fps[1].Center.X, fps[2].Center.Y-fps[1].Center.Y, spanY, alongX, alongY, spanX)
	if !okLeft || !okRight {
		return false
	}
	pos = detectFirstAP(ch, symbol.Meta.SideVersion.Y, fps[0], fps[3], bLeft, locate)
	vy := confirmSideVersion(symbol.Meta.SideVersion.Y, pos)
	if vy == 0 {
		pos = detectFirstAP(ch, symbol.Meta.SideVersion.Y, fps[1], fps[2], bRight, locate)
		vy = confirmSideVersion(symbol.Meta.SideVersion.Y, pos)
		if vy == 0 {
			return false
		}
	}
	symbol.Meta.SideVersion.Y = vy
	symbol.SideSize.Y = spec.VersionToSize(vy)
	return true
}

// AlignmentTrace records the expected and resolved alignment-pattern grid and
// the sampling rectangles selected from it.
type AlignmentTrace struct {
	Attempted  bool
	ReuseCount int
	Reason     string
	Grid       image.Point
	Expected   []FinderPattern
	Patterns   []FinderPattern
	Rectangles []AlignmentRectangle
	Matrix     *core.Bitmap
}

// AlignmentRectangle identifies one AP-grid rectangle used to sample a block.
type AlignmentRectangle struct {
	TopLeft     image.Point
	BottomRight image.Point
}

// AlignmentBlock is one alignment-grid rectangle's sampling request: the
// perspective that carries the block's own module coordinates into the image,
// the block's module extent, and where its top-left module lands in the
// assembled symbol grid.
type AlignmentBlock struct {
	Transform core.Perspective
	Size      image.Point
	Origin    image.Point
}

// BlockSampler assembles one symbol's module grid from its alignment blocks,
// returning nil when a block maps outside the image. The alignment resample
// takes one rather than an image because pattern detection reads the binarized
// channels while only the block sampling needs source colour, and on a device
// route that colour never leaves the device.
//
// It takes the whole block set at once rather than a block at a time so that a
// device can scatter each block into the grid it already holds. Handing blocks
// back one by one made the assembled matrix a host artefact, which the payload
// chain then had to decline because it had never held that grid.
type BlockSampler func(side image.Point, blocks []AlignmentBlock) *core.Bitmap

// SampleAlignmentBlocks assembles the module grid on the host, sampling each
// block through its own perspective. Blocks are written in the order given
// because they overlap: the selection sorts the widest rectangle first, so a
// later, tighter block is the one whose modules should survive.
func SampleAlignmentBlocks(bm *core.Bitmap, side image.Point, blocks []AlignmentBlock) *core.Bitmap {
	matrix := core.NewBitmap(side.X, side.Y, alignmentBlockChannels)
	for _, b := range blocks {
		block := SampleSymbol(bm, b.Transform, b.Size)
		if block == nil {
			return nil
		}
		for y, my := 0, b.Origin.Y; y < b.Size.Y && my < side.Y; y, my = y+1, my+1 {
			for x, mx := 0, b.Origin.X; x < b.Size.X && mx < side.X; x, mx = x+1, mx+1 {
				mo := (my*side.X + mx) * matrix.Channels
				bo := (y*b.Size.X + x) * block.Channels
				copy(matrix.Pix[mo:mo+matrix.Channels], block.Pix[bo:bo+block.Channels])
			}
		}
	}
	return matrix
}

// alignmentBlockChannels is the RGBA width every block sampler returns, which is
// the balanced image's own.
const alignmentBlockChannels = 4

// SampleSymbolByAlignmentPattern detects all alignment patterns, splits the
// symbol into blocks bounded by four found patterns, and samples each block with
// its own perspective transform.
func SampleSymbolByAlignmentPattern(sample BlockSampler, ch [3]*core.Bitmap, symbol *core.DecodedSymbol, fps []FinderPattern) *core.Bitmap {
	return sampleSymbolByAlignmentPattern(sample, alignmentSearches{}, ch, symbol, fps, nil)
}

// SampleSymbolByAlignmentPatternTraced is SampleSymbolByAlignmentPattern with
// detailed observation of the same sampling run.
func SampleSymbolByAlignmentPatternTraced(sample BlockSampler, ch [3]*core.Bitmap, symbol *core.DecodedSymbol, fps []FinderPattern, trace *AlignmentTrace) *core.Bitmap {
	return sampleSymbolByAlignmentPattern(sample, alignmentSearches{}, ch, symbol, fps, trace)
}

// alignmentGrid is one symbol's alignment-pattern search request: the grid
// shape, the finder quad that seeds its corners and axes, and the tables that
// place each cell in module coordinates.
type alignmentGrid struct {
	nApX, nApY int
	sideX      int
	sideY      int
	apType     int
	corners    [4]FinderPattern
	posX       []int
	posY       []int
}

// alignmentLocator locates a whole alignment grid at once. The device
// implementation searches the masks it already holds; a nil locator means the
// caller has no device and the host walks the grid cell by cell instead.
type alignmentLocator func(grid alignmentGrid) ([]FinderPattern, error)

// alignmentCandidate is one explicit alignment-pattern search: where the caller
// predicts the pattern, the module size to size the window by, the two module
// axes as image vectors of one module each, and the run ceiling above which a
// core run is a field of colour rather than a pattern core.
type alignmentCandidate struct {
	Center     core.PointF
	ModuleSize float64
	ModuleMax  float64
	U, V       core.PointF
}

// alignmentPositionLocator answers a batch of explicit predictions, returning
// one entry per candidate in the order given. The side-version walk needs it
// because its candidates are image positions rather than grid cells, and it
// needs them answered together because asking per candidate would be a
// submission per version.
type alignmentPositionLocator func(apType int, candidates []alignmentCandidate) ([]FinderPattern, error)

// alignmentSearches are the device searches the alignment path uses when the
// masks are resident. Both are nil on a host route, and then the host reads
// mask pixels instead.
type alignmentSearches struct {
	grid      alignmentLocator
	positions alignmentPositionLocator
}

func sampleSymbolByAlignmentPattern(sample BlockSampler, locate alignmentSearches, ch [3]*core.Bitmap, symbol *core.DecodedSymbol, fps []FinderPattern, trace *AlignmentTrace) *core.Bitmap {
	// Ports sampleSymbolByAlignmentPattern in detector.c.
	if trace != nil {
		*trace = AlignmentTrace{Attempted: true}
	}
	if sample == nil {
		if trace != nil {
			trace.Reason = "no block sampler"
		}
		return nil
	}
	if symbol.Meta.SideVersion.X < 6 && symbol.Meta.SideVersion.Y < 6 {
		if trace != nil {
			trace.Reason = "side version has no alignment grid"
		}
		return nil
	}
	if symbol.Meta.DefaultMode {
		if !confirmSymbolSize(ch, fps, symbol, locate.positions) {
			if trace != nil {
				trace.Reason = "default-mode side confirmation failed"
			}
			return nil
		}
	}

	vxi := symbol.Meta.SideVersion.X - 1
	vyi := symbol.Meta.SideVersion.Y - 1
	nApX := tables.APNum[vxi]
	nApY := tables.APNum[vyi]
	if trace != nil {
		trace.Grid = image.Pt(nApX, nApY)
	}

	aps := make([]FinderPattern, nApX*nApY)
	expected := make([]FinderPattern, len(aps))
	// A device locator answers for the whole grid without the masks ever
	// reaching the host, which is the only reason they would be downloaded on a
	// route that sampled and decoded entirely on the device. It reports the
	// same per-cell measurements the host walk would, so everything downstream
	// is unchanged.
	if locate.grid != nil {
		grid := alignmentGrid{
			nApX: nApX, nApY: nApY,
			sideX: symbol.SideSize.X, sideY: symbol.SideSize.Y,
			apType: apx,
			corners: [4]FinderPattern{
				fps[0], fps[1], fps[2], fps[3],
			},
			posX: tables.APPos[vxi][:nApX],
			posY: tables.APPos[vyi][:nApY],
		}
		located, err := locate.grid(grid)
		if err == nil && len(located) == len(aps) {
			copy(aps, located)
			copy(expected, located)
			if trace != nil {
				trace.Expected = append([]FinderPattern(nil), expected...)
				trace.Patterns = append([]FinderPattern(nil), aps...)
			}
			return sampleAlignmentRects(sample, aps, expected, symbol, trace, vxi, vyi, nApX, nApY)
		}
	}
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
				// The two module axes at this cell, each interpolated between
				// the quad's two opposite edges at the cell's own position.
				//
				// This is local and long-baseline at the same time, which is
				// what the estimate has to be. Deriving an axis from the two
				// nearest placed neighbours is local but spans only a few
				// modules, so a fraction of a pixel of centre error becomes a
				// visible angle error, and at three pixels per module that
				// tilt moves the sampling grid enough to flip whole rows.
				// Taking one global quad edge instead is stable but wrong under
				// perspective, where the axes genuinely converge across the
				// symbol. Interpolating between opposite edges is exact in both
				// regimes: on an upright symbol both edges are parallel so every
				// interpolation is upright and no noise is amplified, and under
				// perspective the interpolation is the local axis.
				var ux, uy, uSpan, vx, vy, vSpan float64
				switch {
				case i == 0:
					alpha := math.Atan2(fps[1].Center.Y-aps[j-1].Center.Y, fps[1].Center.X-aps[j-1].Center.X)
					distance := aps[j-1].ModuleSize * float64(tables.APPos[vxi][j]-tables.APPos[vxi][j-1])
					aps[index].Center.X = aps[j-1].Center.X + distance*math.Cos(alpha)
					aps[index].Center.Y = aps[j-1].Center.Y + distance*math.Sin(alpha)
					aps[index].ModuleSize = aps[j-1].ModuleSize
				case j == 0:
					base := (i - 1) * nApX
					alpha := math.Atan2(fps[3].Center.Y-aps[base].Center.Y, fps[3].Center.X-aps[base].Center.X)
					distance := aps[base].ModuleSize * float64(tables.APPos[vyi][i]-tables.APPos[vyi][i-1])
					aps[index].Center.X = aps[base].Center.X + distance*math.Cos(alpha)
					aps[index].Center.Y = aps[base].Center.Y + distance*math.Sin(alpha)
					aps[index].ModuleSize = aps[base].ModuleSize
				default:
					iAp0 := (i-1)*nApX + (j - 1)
					iAp1 := (i-1)*nApX + j
					iAp3 := i*nApX + (j - 1)
					avg01 := (aps[iAp0].ModuleSize + aps[iAp1].ModuleSize) / 2.0
					avg13 := (aps[iAp1].ModuleSize + aps[iAp3].ModuleSize) / 2.0
					aps[index].Center.X = (aps[iAp1].Center.X-aps[iAp0].Center.X)/avg01*avg13 + aps[iAp3].Center.X
					aps[index].Center.Y = (aps[iAp1].Center.Y-aps[iAp0].Center.Y)/avg01*avg13 + aps[iAp3].Center.Y
					aps[index].ModuleSize = avg13
				}
				tx := float64(tables.APPos[vxi][j]) / float64(symbol.SideSize.X)
				ty := float64(tables.APPos[vyi][i]) / float64(symbol.SideSize.Y)
				ux, uy = lerpEdge(fps[0].Center, fps[1].Center, fps[3].Center, fps[2].Center, ty)
				uSpan = float64(symbol.SideSize.X - 7)
				vx, vy = lerpEdge(fps[0].Center, fps[3].Center, fps[1].Center, fps[2].Center, tx)
				vSpan = float64(symbol.SideSize.Y - 7)
				aps[index].FoundCount = 0
				tmp := aps[index]
				expected[index] = tmp
				if cellBasis, ok := newAPBasis(ux, uy, uSpan, vx, vy, vSpan); ok {
					aps[index] = findAlignmentPattern(ch, aps[index].Center.X, aps[index].Center.Y, aps[index].ModuleSize, apx, cellBasis)
				}
				if aps[index].FoundCount == 0 {
					aps[index] = tmp
				}
			}
			if expected[index].ModuleSize == 0 {
				expected[index] = aps[index]
			}
		}
	}
	return sampleAlignmentRects(sample, aps, expected, symbol, trace, vxi, vyi, nApX, nApY)
}

// sampleAlignmentRects turns a located alignment grid into the sampled module
// matrix. It is the half of the alignment path that does not care how the
// patterns were found, so the host walk and the device search share it rather
// than each carrying a copy of the rectangle selection.
func sampleAlignmentRects(
	sample BlockSampler,
	aps, expected []FinderPattern,
	symbol *core.DecodedSymbol,
	trace *AlignmentTrace,
	vxi, vyi, nApX, nApY int,
) *core.Bitmap {
	if trace != nil {
		trace.Expected = append([]FinderPattern(nil), expected...)
		trace.Patterns = append([]FinderPattern(nil), aps...)
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
							if aps[tl.Y*nApX+tl.X].FoundCount > 0 && aps[tl.Y*nApX+br.X].FoundCount > 0 &&
								aps[br.Y*nApX+tl.X].FoundCount > 0 && aps[br.Y*nApX+br.X].FoundCount > 0 {
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
	if trace != nil {
		trace.Rectangles = make([]AlignmentRectangle, len(rects))
		for i, r := range rects {
			trace.Rectangles[i] = AlignmentRectangle{TopLeft: r.tl, BottomRight: r.br}
		}
	}

	width, height := symbol.SideSize.X, symbol.SideSize.Y
	blocks := make([]AlignmentBlock, 0, len(rects))

	for _, r := range rects {
		blkX := tables.APPos[vxi][r.br.X] - tables.APPos[vxi][r.tl.X] + 1
		blkY := tables.APPos[vyi][r.br.Y] - tables.APPos[vyi][r.tl.Y] + 1
		p0 := core.Pt(0.5, 0.5)
		p1 := core.Pt(float64(blkX)-0.5, 0.5)
		p2 := core.Pt(float64(blkX)-0.5, float64(blkY)-0.5)
		p3 := core.Pt(0.5, float64(blkY)-0.5)
		if r.tl.Y == 0 {
			blkY += spec.DistanceToBorder - 1
			p0.Y, p1.Y = 3.5, 3.5
			p2.Y, p3.Y = float64(blkY)-0.5, float64(blkY)-0.5
		}
		if r.br.Y == nApY-1 {
			blkY += spec.DistanceToBorder - 1
			p2.Y, p3.Y = float64(blkY)-3.5, float64(blkY)-3.5
		}
		if r.tl.X == 0 {
			blkX += spec.DistanceToBorder - 1
			p0.X, p3.X = 3.5, 3.5
			p1.X, p2.X = float64(blkX)-0.5, float64(blkX)-0.5
		}
		if r.br.X == nApX-1 {
			blkX += spec.DistanceToBorder - 1
			p1.X, p2.X = float64(blkX)-3.5, float64(blkX)-3.5
		}
		src := [4]core.PointF{p0, p1, p2, p3}
		dst := [4]core.PointF{
			aps[r.tl.Y*nApX+r.tl.X].Center,
			aps[r.tl.Y*nApX+r.br.X].Center,
			aps[r.br.Y*nApX+r.br.X].Center,
			aps[r.br.Y*nApX+r.tl.X].Center,
		}
		startX := tables.APPos[vxi][r.tl.X] - 1
		startY := tables.APPos[vyi][r.tl.Y] - 1
		if r.tl.X == 0 {
			startX = 0
		}
		if r.tl.Y == 0 {
			startY = 0
		}
		blocks = append(blocks, AlignmentBlock{
			Transform: core.QuadToQuad(src, dst),
			Size:      image.Pt(blkX, blkY),
			Origin:    image.Pt(startX, startY),
		})
	}

	matrix := sample(image.Pt(width, height), blocks)
	if matrix == nil {
		if trace != nil {
			trace.Reason = "alignment block sampling failed"
		}
		return nil
	}
	if trace != nil {
		trace.Matrix = matrix
	}
	return matrix
}
