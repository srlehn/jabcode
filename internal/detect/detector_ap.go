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
			if image.Pix[(starty+i*offsetY)*image.Width+(startx+i*offsetX)] == image.Pix[(starty+(i-1)*offsetY)*image.Width+(startx+(i-1)*offsetX)] {
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
				if image.Pix[(starty-i*offsetY)*image.Width+(startx-i*offsetX)] == image.Pix[(starty-(i-1)*offsetY)*image.Width+(startx-(i-1)*offsetX)] {
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
		if image.Pix[(cy-i)*image.Width+cx] == image.Pix[(cy-(i-1))*image.Width+cx] {
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
		if image.Pix[(cy+i)*image.Width+cx] == image.Pix[(cy+(i-1))*image.Width+cx] {
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
func crossCheckPatternAP(ch [3]*core.Bitmap, y, minx, maxx, curX, apType int, maxModuleSize float64, centerx, centery, moduleSize *float64, dir *int) bool {
	// Ports crossCheckPatternAP in detector.c.
	rowR := ch[0].Pix[y*ch[0].Width : (y+1)*ch[0].Width]
	rowB := ch[2].Pix[y*ch[2].Width : (y+1)*ch[2].Width]
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
	rowR = ch[0].Pix[int(lcy[0])*ch[0].Width : (int(lcy[0])+1)*ch[0].Width]
	lcx[0] = crossCheckPatternHorizontalAP(rowR, 0, minx, maxx, int(center.X), apType, maxModuleSize, &lmsH[0])
	if lcx[0] < 0 {
		return false
	}

	lcy[2] = crossCheckPatternVerticalAP(ch[2], center, int(maxModuleSize), &lmsV[2])
	if lcy[2] < 0 {
		return false
	}
	rowB = ch[2].Pix[int(lcy[2])*ch[2].Width : (int(lcy[2])+1)*ch[2].Width]
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
// (x, y).
func findAlignmentPattern(ch [3]*core.Bitmap, x, y, moduleSize float64, apType int) FinderPattern {
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
			rowR := ch[0].Pix[i*ch[0].Width : (i+1)*ch[0].Width]

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
			ap := FinderPattern{Typ: apType, FoundCount: 1, ModuleSize: apModuleSize, Center: core.Pt(centerx, centery), direction: apDir}
			if index := saveAlignmentPattern(&ap, aps, &counter); index >= 0 {
				return aps[index]
			}
		}
	}
	return FinderPattern{Typ: -1}
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

// detectFirstAP detects the first alignment pattern between two finder patterns,
// returning its position.
func detectFirstAP(ch [3]*core.Bitmap, sideVersion int, fp1, fp2 FinderPattern) int {
	// Ports detectFirstAP in detector.c.
	alpha := math.Atan2(fp2.Center.Y-fp1.Center.Y, fp2.Center.X-fp1.Center.X)
	nextVersion := sideVersion
	dir := 1
	up, down := 0, 0
	for {
		distance := fp1.ModuleSize * float64(tables.APPos[nextVersion-1][1]-tables.APPos[nextVersion-1][0])
		cx := fp1.Center.X + distance*math.Cos(alpha)
		cy := fp1.Center.Y + distance*math.Sin(alpha)
		ap := findAlignmentPattern(ch, cx, cy, fp1.ModuleSize, apx)
		if ap.FoundCount > 0 {
			if pos := firstAPPos(4 + CalculateModuleNumber(fp1, ap)); pos > 0 {
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
	return core.Failure
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
func confirmSymbolSize(ch [3]*core.Bitmap, fps []FinderPattern, symbol *core.DecodedSymbol) bool {
	// Ports confirmSymbolSize in detector.c.
	pos := detectFirstAP(ch, symbol.Meta.SideVersion.X, fps[0], fps[1])
	vx := confirmSideVersion(symbol.Meta.SideVersion.X, pos)
	if vx == 0 {
		pos = detectFirstAP(ch, symbol.Meta.SideVersion.X, fps[3], fps[2])
		vx = confirmSideVersion(symbol.Meta.SideVersion.X, pos)
		if vx == 0 {
			return false
		}
	}
	symbol.Meta.SideVersion.X = vx
	symbol.SideSize.X = spec.VersionToSize(vx)

	pos = detectFirstAP(ch, symbol.Meta.SideVersion.Y, fps[0], fps[3])
	vy := confirmSideVersion(symbol.Meta.SideVersion.Y, pos)
	if vy == 0 {
		pos = detectFirstAP(ch, symbol.Meta.SideVersion.Y, fps[1], fps[2])
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

// SampleSymbolByAlignmentPattern detects all alignment patterns, splits the
// symbol into blocks bounded by four found patterns, and samples each block with
// its own perspective transform.
func SampleSymbolByAlignmentPattern(bm *core.Bitmap, ch [3]*core.Bitmap, symbol *core.DecodedSymbol, fps []FinderPattern) *core.Bitmap {
	return sampleSymbolByAlignmentPattern(bm, ch, symbol, fps, nil)
}

// SampleSymbolByAlignmentPatternTraced is SampleSymbolByAlignmentPattern with
// detailed observation of the same sampling run.
func SampleSymbolByAlignmentPatternTraced(bm *core.Bitmap, ch [3]*core.Bitmap, symbol *core.DecodedSymbol, fps []FinderPattern, trace *AlignmentTrace) *core.Bitmap {
	return sampleSymbolByAlignmentPattern(bm, ch, symbol, fps, trace)
}

func sampleSymbolByAlignmentPattern(bm *core.Bitmap, ch [3]*core.Bitmap, symbol *core.DecodedSymbol, fps []FinderPattern, trace *AlignmentTrace) *core.Bitmap {
	// Ports sampleSymbolByAlignmentPattern in detector.c.
	if trace != nil {
		*trace = AlignmentTrace{Attempted: true}
	}
	if symbol.Meta.SideVersion.X < 6 && symbol.Meta.SideVersion.Y < 6 {
		if trace != nil {
			trace.Reason = "side version has no alignment grid"
		}
		return nil
	}
	if symbol.Meta.DefaultMode {
		if !confirmSymbolSize(ch, fps, symbol) {
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
				aps[index].FoundCount = 0
				tmp := aps[index]
				expected[index] = tmp
				aps[index] = findAlignmentPattern(ch, aps[index].Center.X, aps[index].Center.Y, aps[index].ModuleSize, apx)
				if aps[index].FoundCount == 0 {
					aps[index] = tmp
				}
			}
			if expected[index].ModuleSize == 0 {
				expected[index] = aps[index]
			}
		}
	}
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
	matrix := core.NewBitmap(width, height, bm.Channels)

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
		block := SampleSymbol(bm, core.QuadToQuad(src, dst), image.Pt(blkX, blkY))
		if block == nil {
			if trace != nil {
				trace.Reason = "alignment block sampling failed"
			}
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
				mo := (my*width + mx) * matrix.Channels
				bo := (y*blkX + x) * block.Channels
				copy(matrix.Pix[mo:mo+matrix.Channels], block.Pix[bo:bo+block.Channels])
			}
		}
	}
	if trace != nil {
		trace.Matrix = matrix
	}
	return matrix
}
