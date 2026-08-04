package detect

import (
	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/palette"
)

// This file is the Go blueprint of the finder chain kernels: every mirrorChain*
// function computes the same per-hit sequence as its WGSL twin in
// shaders/finder_chain_prelude.wgsl and the family fragments, reading packed
// mask words and using the sf* softfloat mirrors for every float64 the CPU
// chain computes (the kernels fold duplicated call sites into loops to
// bound pipeline compile time; the arithmetic is identical). The
// equivalence tests prove the mirrored chain bit-identical to the CPU
// per-hit chain, so the algorithm is pinned without a device; the device
// parity tests then pin the WGSL transcription itself.

// chainMasks is the packed three-channel binary mask layout the GPU pack
// stage emits: eight pixels per word, three bits per pixel (R, G, B).
type chainMasks struct {
	words []uint32
	w, h  int
}

// bit returns the binary mask bit of a pixel index. Out-of-range indexes read
// as zero; the CPU chain never survives to read one on decodable inputs.
func (m chainMasks) bit(pixel, channel int32) uint32 {
	if pixel < 0 || int(pixel) >= m.w*m.h {
		return 0
	}
	word := m.words[pixel/8]
	return (word >> ((uint32(pixel) % 8) * 3)) >> uint32(channel) & 1
}

// packChainMasks packs CPU binarized channels into the GPU mask word layout.
func packChainMasks(ch [3]*core.Bitmap) chainMasks {
	w, h := ch[0].Width, ch[0].Height
	words := make([]uint32, (w*h+7)/8)
	for pixel := 0; pixel < w*h; pixel++ {
		var mask uint32
		for channel := range 3 {
			if ch[channel].Pix[pixel] > 0 {
				mask |= 1 << channel
			}
		}
		words[pixel/8] |= mask << ((uint32(pixel) % 8) * 3)
	}
	return chainMasks{words: words, w: w, h: h}
}

// chainOutcome is one raw hit's mirrored device-chain outcome record; the
// production flag constants live in finder_row_hits.go.
type chainOutcome struct {
	flags uint32
	typ   int32
	dir   int32
	cx    float32
	cy    float32
	ms    float32
}

// chainAbs is the kernels' abs over one f32; math.Abs would round-trip
// through float64 and hide a difference the shader would keep.
func chainAbs(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// mirrorChainSlack mirrors PrimaryDetector.ccSlack.
func mirrorChainSlack(moduleSize float32, printPass bool) int32 {
	if printPass {
		s := int32(moduleSize*0.5 + 0.5)
		if s < 3 {
			return 3
		}
		return s
	}
	return 3
}

// mirrorCheckPatternCross mirrors checkPatternCross.
func mirrorCheckPatternCross(sc [5]int32) (float32, bool) {
	inside := int32(0)
	for i := 1; i < 4; i++ {
		if sc[i] == 0 {
			return 0, false
		}
		inside += sc[i]
	}
	layer := (float32(inside) / float32(3))
	tol := (layer * 0.5)
	halfTol := (tol * 0.5)
	ok := (chainAbs((layer - float32(sc[1]))) < tol) &&
		(chainAbs((layer - float32(sc[2]))) < tol) &&
		(chainAbs((layer - float32(sc[3]))) < tol) &&
		(halfTol < float32(sc[0])) &&
		(halfTol < float32(sc[4])) &&
		(chainAbs(float32(sc[1]-sc[3])) < tol)
	return layer, ok
}

// mirrorCheckModuleSize2 mirrors checkModuleSize2.
func mirrorCheckModuleSize2(s1, s2 float32) bool {
	mean := ((s1 + s2) * 0.5)
	tol := ((mean * 2) / float32(5))
	return (chainAbs((mean - s1)) < tol) && (chainAbs((mean - s2)) < tol)
}

// mirrorCheckModuleSize3 mirrors checkModuleSize3.
func mirrorCheckModuleSize3(r, g, b float32) bool {
	mean := (((r + g) + b) / float32(3))
	tol := ((mean * 2) / float32(5))
	return (chainAbs((mean - r)) < tol) &&
		(chainAbs((mean - g)) < tol) &&
		(chainAbs((mean - b)) < tol)
}

// mirrorCrossCheckPatternVertical mirrors crossCheckPatternVertical. It returns
// ok, the refined centery and the module size.
func mirrorCrossCheckPatternVertical(
	m chainMasks, channel int32,
	moduleSizeMax int32, centerx, centery float32, slack int32,
) (bool, float32, float32) {
	const stateMiddle = int32(2)
	var sc [5]int32
	w := int32(m.w)
	h := int32(m.h)
	cx := int32(centerx)
	cy := int32(centery)

	var i, stateIndex int32
	sc[stateMiddle]++
	for i = 1; i <= cy && stateIndex <= stateMiddle; i++ {
		if m.bit((cy-i)*w+cx, channel) == m.bit((cy-(i-1))*w+cx, channel) {
			sc[stateMiddle-stateIndex]++
		} else if stateIndex > 0 && sc[stateMiddle-stateIndex] < slack {
			sc[stateMiddle-(stateIndex-1)] += sc[stateMiddle-stateIndex]
			sc[stateMiddle-stateIndex] = 0
			stateIndex--
			sc[stateMiddle-stateIndex]++
		} else {
			stateIndex++
			if stateIndex > stateMiddle {
				break
			}
			sc[stateMiddle-stateIndex]++
		}
	}
	if stateIndex < stateMiddle {
		return false, centery, 0
	}
	stateIndex = 0
	for i = 1; cy+i < h && stateIndex <= stateMiddle; i++ {
		if m.bit((cy+i)*w+cx, channel) == m.bit((cy+(i-1))*w+cx, channel) {
			sc[stateMiddle+stateIndex]++
		} else if stateIndex > 0 && sc[stateMiddle+stateIndex] < slack {
			sc[stateMiddle+(stateIndex-1)] += sc[stateMiddle+stateIndex]
			sc[stateMiddle+stateIndex] = 0
			stateIndex--
			sc[stateMiddle+stateIndex]++
		} else {
			stateIndex++
			if stateIndex > stateMiddle {
				break
			}
			sc[stateMiddle+stateIndex]++
		}
	}
	if stateIndex < stateMiddle {
		return false, centery, 0
	}
	ms, ret := mirrorCheckPatternCross(sc)
	if ret && (ms <= float32(moduleSizeMax)) {
		newCentery := (float32(cy+i-sc[4]-sc[3]) - (float32(sc[2]) * 0.5))
		return true, newCentery, ms
	}
	return false, centery, ms
}

// mirrorCrossCheckPatternHorizontal mirrors crossCheckPatternHorizontal. It
// returns ok, the refined centerx and the module size.
func mirrorCrossCheckPatternHorizontal(
	m chainMasks, channel int32,
	moduleSizeMax float32, centerx, centery float32, slack int32,
) (bool, float32, float32) {
	const stateMiddle = int32(2)
	var sc [5]int32
	w := int32(m.w)
	startx := int32(centerx)
	rowOffset := int32(centery) * w

	var i, stateIndex int32
	sc[stateMiddle]++
	for i = 1; i <= startx && stateIndex <= stateMiddle; i++ {
		if m.bit(rowOffset+(startx-i), channel) == m.bit(rowOffset+(startx-(i-1)), channel) {
			sc[stateMiddle-stateIndex]++
		} else if stateIndex > 0 && sc[stateMiddle-stateIndex] < slack {
			sc[stateMiddle-(stateIndex-1)] += sc[stateMiddle-stateIndex]
			sc[stateMiddle-stateIndex] = 0
			stateIndex--
			sc[stateMiddle-stateIndex]++
		} else {
			stateIndex++
			if stateIndex > stateMiddle {
				break
			}
			sc[stateMiddle-stateIndex]++
		}
	}
	if stateIndex < stateMiddle {
		return false, centerx, 0
	}
	stateIndex = 0
	for i = 1; startx+i < w && stateIndex <= stateMiddle; i++ {
		if m.bit(rowOffset+(startx+i), channel) == m.bit(rowOffset+(startx+(i-1)), channel) {
			sc[stateMiddle+stateIndex]++
		} else if stateIndex > 0 && sc[stateMiddle+stateIndex] < slack {
			sc[stateMiddle+(stateIndex-1)] += sc[stateMiddle+stateIndex]
			sc[stateMiddle+stateIndex] = 0
			stateIndex--
			sc[stateMiddle+stateIndex]++
		} else {
			stateIndex++
			if stateIndex > stateMiddle {
				break
			}
			sc[stateMiddle+stateIndex]++
		}
	}
	if stateIndex < stateMiddle {
		return false, centerx, 0
	}
	ms, ret := mirrorCheckPatternCross(sc)
	if ret && (ms <= moduleSizeMax) {
		newCenterx := (float32(startx+i-sc[4]-sc[3]) - (float32(sc[2]) * 0.5))
		return true, newCenterx, ms
	}
	return false, centerx, ms
}

// mirrorCrossCheckPatternDiagonal mirrors crossCheckPatternDiagonal, including
// its retry flips and the module-size write of a failed second try. It
// returns the confirmed count and the refined center, module size and
// direction.
func mirrorCrossCheckPatternDiagonal(
	m chainMasks, channel int32,
	typ int32, moduleSizeMax float32, centerx, centery, moduleSize float32,
	dir int32, bothDir bool, slack int32,
) (int32, float32, float32, float32, int32) {
	const stateMiddle = int32(2)
	w := int32(m.w)
	h := int32(m.h)
	var offsetX, offsetY int32
	fixDir := false
	switch {
	case dir != 0:
		if dir > 0 {
			offsetX, offsetY, dir = -1, -1, 1
		} else {
			offsetX, offsetY, dir = 1, -1, -1
		}
		fixDir = true
	case typ == fp0 || typ == fp1:
		offsetX, offsetY, dir = -1, -1, 1
	default:
		offsetX, offsetY, dir = 1, -1, -1
	}

	confirmed := int32(0)
	tryCount := int32(0)
	var tmpModuleSize float32
	for {
		flag := false
		tryCount++
		var i, stateIndex int32
		var sc [5]int32
		startx := int32(centerx)
		starty := int32(centery)

		sc[stateMiddle]++
		for j := int32(1); starty+j*offsetY >= 0 && starty+j*offsetY < h &&
			startx+j*offsetX >= 0 && startx+j*offsetX < w && stateIndex <= stateMiddle; j++ {
			if m.bit((starty+j*offsetY)*w+(startx+j*offsetX), channel) ==
				m.bit((starty+(j-1)*offsetY)*w+(startx+(j-1)*offsetX), channel) {
				sc[stateMiddle-stateIndex]++
			} else if stateIndex > 0 && sc[stateMiddle-stateIndex] < slack {
				sc[stateMiddle-(stateIndex-1)] += sc[stateMiddle-stateIndex]
				sc[stateMiddle-stateIndex] = 0
				stateIndex--
				sc[stateMiddle-stateIndex]++
			} else {
				stateIndex++
				if stateIndex > stateMiddle {
					break
				}
				sc[stateMiddle-stateIndex]++
			}
		}
		if stateIndex < stateMiddle {
			if tryCount == 1 {
				flag = true
				offsetX = -offsetX
				dir = -dir
			} else {
				return confirmed, centerx, centery, moduleSize, dir
			}
		}

		if !flag {
			stateIndex = 0
			for i = 1; starty-i*offsetY >= 0 && starty-i*offsetY < h &&
				startx-i*offsetX >= 0 && startx-i*offsetX < w && stateIndex <= stateMiddle; i++ {
				if m.bit((starty-i*offsetY)*w+(startx-i*offsetX), channel) ==
					m.bit((starty-(i-1)*offsetY)*w+(startx-(i-1)*offsetX), channel) {
					sc[stateMiddle+stateIndex]++
				} else if stateIndex > 0 && sc[stateMiddle+stateIndex] < slack {
					sc[stateMiddle+(stateIndex-1)] += sc[stateMiddle+stateIndex]
					sc[stateMiddle+stateIndex] = 0
					stateIndex--
					sc[stateMiddle+stateIndex]++
				} else {
					stateIndex++
					if stateIndex > stateMiddle {
						break
					}
					sc[stateMiddle+stateIndex]++
				}
			}
			if stateIndex < stateMiddle {
				if tryCount == 1 {
					flag = true
					offsetX = -offsetX
					dir = -dir
				} else {
					return confirmed, centerx, centery, moduleSize, dir
				}
			}
		}

		if !flag {
			ms, ret := mirrorCheckPatternCross(sc)
			moduleSize = ms
			if ret && (moduleSize <= moduleSizeMax) {
				if 0 < tmpModuleSize {
					moduleSize = ((moduleSize + tmpModuleSize) * 0.5)
				} else {
					tmpModuleSize = moduleSize
				}
				// Mirrors the walk-direction split in crossCheckPatternDiagonal:
				// y always advances, x retreats wherever offsetX is +1.
				edge := i - sc[4] - sc[3]
				half := (float32(sc[2]) * 0.5)
				if offsetX < 0 {
					centerx = (float32(startx+edge) - half)
				} else {
					centerx = (float32(startx-edge+1) + half)
				}
				centery = (float32(starty+edge) - half)
				confirmed++
				if !bothDir || tryCount == 2 || fixDir {
					if confirmed == 2 {
						dir = 2
					}
					return confirmed, centerx, centery, moduleSize, dir
				}
			} else {
				offsetX = -offsetX
				dir = -dir
			}
		}
		if !(tryCount < 2 && !fixDir) {
			break
		}
	}
	return confirmed, centerx, centery, moduleSize, dir
}

// mirrorCrossCheckColor mirrors crossCheckColor with moduleNumber fixed at 5,
// the only value the chain uses. colorBit is the expected mask bit.
func mirrorCrossCheckColor(
	m chainMasks, channel int32,
	colorBit uint32, moduleSize, centerx, centery, dirMode, tol int32,
) bool {
	const moduleNumber = int32(5)
	w := int32(m.w)
	h := int32(m.h)
	if centerx < 0 || centerx >= w || centery < 0 || centery >= h {
		return false
	}
	switch dirMode {
	case 0:
		length := moduleSize * (moduleNumber - 1)
		startx := max(centerx-length/2, 0)
		unmatch := int32(0)
		for j := startx; j < startx+length && j < w; j++ {
			if m.bit(centery*w+j, channel) != colorBit {
				unmatch++
			} else if unmatch <= tol {
				unmatch = 0
			}
			if unmatch > tol {
				return false
			}
		}
		return true
	case 1:
		length := moduleSize * (moduleNumber - 1)
		starty := max(centery-length/2, 0)
		unmatch := int32(0)
		for i := starty; i < starty+length && i < h; i++ {
			if m.bit(w*i+centerx, channel) != colorBit {
				unmatch++
			} else if unmatch <= tol {
				unmatch = 0
			}
			if unmatch > tol {
				return false
			}
		}
		return true
	case 2:
		// int(float64(moduleSize) * (float64(moduleNumber) / (2.0 * 1.41421)))
		// with the constant division folded on the host side.
		offset := int32((float32(uint32(moduleSize)) * chainDiagonalLengthConst()))
		length := offset * 2
		unmatch := int32(0)
		startx := max(centerx-offset, 0)
		starty := max(centery-offset, 0)
		for i := int32(0); i < length && starty+i < h && startx+i < w; i++ {
			if m.bit(w*(starty+i)+(startx+i), channel) != colorBit {
				unmatch++
			} else if unmatch <= tol {
				unmatch = 0
			}
			if unmatch > tol {
				break
			}
		}
		if unmatch < tol {
			return true
		}
		unmatch = 0
		startx = max(centerx-offset, 0)
		starty = min(centery+offset, h-1)
		for i := int32(0); i < length && starty-i >= 0 && startx+i < w; i++ {
			if m.bit(w*(starty-i)+(startx+i), channel) != colorBit {
				unmatch++
			} else if unmatch <= tol {
				unmatch = 0
			}
			if unmatch > tol {
				return false
			}
		}
		return true
	}
	return false
}

// chainDiagonalLengthConst is float64(5) / (2.0 * 1.41421), the diagonal
// length factor of crossCheckColor, as raw bits.
func chainDiagonalLengthConst() float32 {
	return float32(5.0 / (2.0 * 1.41421))
}

// mirrorCrossCheckPatternCh mirrors crossCheckPatternCh for horizontal
// candidates (hv 0), the only orientation the device chain replays. It
// returns ok and the refined module size, center and direction results.
func mirrorCrossCheckPatternCh(
	m chainMasks, channel int32,
	typ int32, moduleSizeMax float32, centerx, centery float32, slack int32,
) (ok bool, moduleSize, cx, cy float32, dir, dcc int32) {
	cx, cy = centerx, centery
	var msV, msH, msD float32
	vcc := false
	if okV, newCy, ms := mirrorCrossCheckPatternVertical(m, channel, int32(moduleSizeMax), cx, cy, slack); okV {
		vcc = true
		cy = newCy
		msV = ms
		okH, newCx, ms := mirrorCrossCheckPatternHorizontal(m, channel, moduleSizeMax, cx, cy, slack)
		if !okH {
			return false, moduleSize, cx, cy, dir, dcc
		}
		cx = newCx
		msH = ms
	}
	axisx, axisy := cx, cy
	dcc, cx, cy, msD, dir = mirrorCrossCheckPatternDiagonal(m, channel, typ, moduleSizeMax, cx, cy, msD, dir, !vcc, slack)
	switch {
	case vcc && dcc > 0:
		moduleSize = ((msV + msH) / float32(2))
		return true, moduleSize, axisx, axisy, dir, dcc
	case dcc == 2:
		okH, newCx, ms := mirrorCrossCheckPatternHorizontal(m, channel, moduleSizeMax, cx, cy, slack)
		if !okH {
			return false, moduleSize, cx, cy, dir, dcc
		}
		cx = newCx
		msH = ms
		moduleSize = msH
		return true, moduleSize, cx, cy, dir, dcc
	}
	return false, moduleSize, cx, cy, dir, dcc
}

// chainPaletteBit is the binarized palette bit of one default-palette color
// index and channel, the classification authority the host passes to the
// kernel.
func chainPaletteBit(colorIndex, channel int32) uint32 {
	if palette.Default[colorIndex*3+channel] > 0 {
		return 1
	}
	return 0
}

// mirrorClassify mirrors FinderPattern.classify over mask bits.
func mirrorClassify(candidates []int32, typeR, typeG, typeB uint32) (int32, bool) {
	for _, t := range candidates {
		coreIdx := int32(fpCoreColorIndex(int(t)))
		if typeR == chainPaletteBit(coreIdx, 0) &&
			typeG == chainPaletteBit(coreIdx, 1) &&
			typeB == chainPaletteBit(coreIdx, 2) {
			return t, true
		}
	}
	return 0, false
}

// mirrorCrossCheckPattern mirrors crossCheckPattern for horizontal current-family
// candidates. It refines the pattern in place and reports survival.
func mirrorCrossCheckPattern(m chainMasks, typ int32, cx0, cy0, moduleSize0 float32, slack int32) (bool, float32, float32, float32, int32) {
	moduleSizeMax := (moduleSize0 * 2)

	okG, msG, cxG, cyG, dirG, dccG := mirrorCrossCheckPatternCh(m, 1, typ, moduleSizeMax, cx0, cy0, slack)
	if !okG {
		return false, cx0, cy0, moduleSize0, 0
	}

	if typ == fp1 || typ == fp2 {
		okR, msR, cxR, cyR, dirR, dccR := mirrorCrossCheckPatternCh(m, 0, typ, moduleSizeMax, cx0, cy0, slack)
		if !okR {
			return false, cx0, cy0, moduleSize0, 0
		}
		if !mirrorCheckModuleSize2(msR, msG) {
			return false, cx0, cy0, moduleSize0, 0
		}
		ms := ((msR + msG) * 0.5)
		cx := ((cxR + cxG) * 0.5)
		cy := ((cyR + cyG) * 0.5)
		coreBlue := chainPaletteBit(int32(fpCoreColorIndex(fp2)), 2)
		for d := int32(0); d < 3; d++ {
			if !mirrorCrossCheckColor(m, 2, coreBlue, int32(ms), int32(cx), int32(cy), d, slack) {
				return false, cx0, cy0, moduleSize0, 0
			}
		}
		direction := int32(-1)
		switch {
		case dccR == 2 || dccG == 2:
			direction = 2
		case dirR+dirG > 0:
			direction = 1
		}
		return true, cx, cy, ms, direction
	}

	okB, msB, cxB, cyB, dirB, dccB := mirrorCrossCheckPatternCh(m, 2, typ, moduleSizeMax, cx0, cy0, slack)
	if !okB {
		return false, cx0, cy0, moduleSize0, 0
	}
	if !mirrorCheckModuleSize2(msG, msB) {
		return false, cx0, cy0, moduleSize0, 0
	}
	ms := ((msG + msB) * 0.5)
	cx := ((cxG + cxB) * 0.5)
	cy := ((cyG + cyB) * 0.5)
	coreRed := chainPaletteBit(int32(fpCoreColorIndex(fp3)), 0)
	for d := int32(0); d < 3; d++ {
		if !mirrorCrossCheckColor(m, 0, coreRed, int32(ms), int32(cx), int32(cy), d, slack) {
			return false, cx0, cy0, moduleSize0, 0
		}
	}
	direction := int32(-1)
	switch {
	case dccG == 2 || dccB == 2:
		direction = 2
	case dirG+dirB > 0:
		direction = 1
	}
	return true, cx, cy, ms, direction
}

// mirrorChainCurrentHit mirrors processCurrentFamilyHit for one raw green-row
// hit, producing the outcome record the kernel writes.
func mirrorChainCurrentHit(m chainMasks, hit finderRowHit, printPass bool) chainOutcome {
	var out chainOutcome
	w := int32(m.w)
	y := int32(hit.y)
	centerG := (float32(int32(hit.endPos-hit.s4-hit.s3)) - (float32(int32(hit.s2)) * 0.5))
	moduleG := (float32(int32(hit.inside)) / float32(3))
	rowOffset := y * w

	typeG := m.bit(rowOffset+int32(centerG), 1)
	centerR, centerB := centerG, centerG
	var typeR, typeB uint32
	var moduleR, moduleB float32
	blueBranch, redBranch := false, false
	slack := mirrorChainSlack(moduleG, printPass)
	moduleGx2 := (moduleG * 2)

	if okB, newCenterB, ms := mirrorCrossCheckPatternHorizontal(m, 2, moduleGx2, centerB, float32(y), slack); okB {
		out.flags |= chainFlagBranchBlue
		centerB = newCenterB
		moduleB = ms
		typeB = m.bit(rowOffset+int32(centerB), 2)
		moduleR = moduleG
		coreRed := chainPaletteBit(int32(fpCoreColorIndex(fp3)), 0)
		if mirrorCrossCheckColor(m, 0, coreRed, int32(moduleR), int32(centerR), y, 0, slack) {
			typeR = 0
			blueBranch = true
		}
	} else if okR, newCenterR, ms := mirrorCrossCheckPatternHorizontal(m, 0, moduleGx2, centerR, float32(y), slack); okR {
		out.flags |= chainFlagBranchRed
		centerR = newCenterR
		moduleR = ms
		typeR = m.bit(rowOffset+int32(centerR), 0)
		moduleB = moduleG
		coreBlue := chainPaletteBit(int32(fpCoreColorIndex(fp2)), 2)
		if mirrorCrossCheckColor(m, 2, coreBlue, int32(moduleB), int32(centerB), y, 0, slack) {
			typeB = 0
			redBranch = true
			out.flags |= chainFlagRedColor
		}
	}

	if !(blueBranch || redBranch) {
		return out
	}
	var typ int32
	var cx, ms float32
	if blueBranch {
		if !mirrorCheckModuleSize2(moduleG, moduleB) {
			return out
		}
		cx = ((centerG + centerB) * 0.5)
		ms = ((moduleG + moduleB) * 0.5)
		t, ok := mirrorClassify([]int32{fp0, fp3}, typeR, typeG, typeB)
		if !ok {
			return out
		}
		typ = t
	} else {
		if !mirrorCheckModuleSize2(moduleR, moduleG) {
			return out
		}
		cx = ((centerR + centerG) * 0.5)
		ms = ((moduleR + moduleG) * 0.5)
		t, ok := mirrorClassify([]int32{fp1, fp2}, typeR, typeG, typeB)
		if !ok {
			return out
		}
		typ = t
		out.flags |= chainFlagRedClassified
	}
	survived, fcx, fcy, fms, dir := mirrorCrossCheckPattern(m, typ, cx, float32(y), ms, mirrorChainSlack(ms, printPass))
	if !survived {
		return out
	}
	out.flags |= chainFlagSurvivor
	out.typ = typ
	out.dir = dir
	out.cx = fcx
	out.cy = fcy
	out.ms = fms
	return out
}

// mirrorClassifyBSI mirrors FinderPattern.classifyBSIFamily over mask bits.
func mirrorClassifyBSI(typeR, typeG, typeB uint32) (int32, bool) {
	for typ, colorIndex := range bsiFamilyFinderCoreColors {
		if typeR == chainPaletteBit(int32(colorIndex), 0) &&
			typeG == chainPaletteBit(int32(colorIndex), 1) &&
			typeB == chainPaletteBit(int32(colorIndex), 2) {
			return int32(typ), true
		}
	}
	return 0, false
}

// mirrorCrossCheckPatternBSI mirrors crossCheckPatternBSIFamily for horizontal
// candidates (hv 0).
func mirrorCrossCheckPatternBSI(m chainMasks, typ int32, cx0, cy0, moduleSize0 float32, slack int32) (bool, float32, float32, float32, int32) {
	moduleSizeMax := (moduleSize0 * 2)
	var moduleSize [3]float32
	var centerX, centerY [3]float32
	var direction, diagonal [3]int32
	for c := int32(0); c < 3; c++ {
		ok, ms, cx, cy, dir, dcc := mirrorCrossCheckPatternCh(m, c, typ, moduleSizeMax, cx0, cy0, slack)
		if !ok {
			return false, cx0, cy0, moduleSize0, 0
		}
		moduleSize[c] = ms
		centerX[c] = cx
		centerY[c] = cy
		direction[c] = dir
		diagonal[c] = dcc
	}
	if !mirrorCheckModuleSize3(moduleSize[0], moduleSize[1], moduleSize[2]) {
		return false, cx0, cy0, moduleSize0, 0
	}
	ms := (((moduleSize[0] + moduleSize[1]) + moduleSize[2]) / float32(3))
	cx := (((centerX[0] + centerX[1]) + centerX[2]) / float32(3))
	cy := (((centerY[0] + centerY[1]) + centerY[2]) / float32(3))
	dir := int32(-1)
	if diagonal[0] == 2 || diagonal[1] == 2 || diagonal[2] == 2 {
		dir = 2
	} else if direction[0]+direction[1]+direction[2] > 0 {
		dir = 1
	}
	return true, cx, cy, ms, dir
}

// mirrorChainBSIHit mirrors processBSIFamilyHit for one raw red-row hit.
func mirrorChainBSIHit(m chainMasks, hit finderRowHit, printPass bool) chainOutcome {
	var out chainOutcome
	w := int32(m.w)
	y := int32(hit.y)
	center0 := (float32(int32(hit.endPos-hit.s4-hit.s3)) - (float32(int32(hit.s2)) * 0.5))
	module0 := (float32(int32(hit.inside)) / float32(3))
	rowOffset := y * w
	slack := mirrorChainSlack(module0, printPass)
	module0x2 := (module0 * 2)

	center := [3]float32{center0, center0, center0}
	moduleSize := [3]float32{module0}
	ok1, c1, ms1 := mirrorCrossCheckPatternHorizontal(m, 1, module0x2, center[1], float32(y), slack)
	if !ok1 {
		return out
	}
	center[1], moduleSize[1] = c1, ms1
	ok2, c2, ms2 := mirrorCrossCheckPatternHorizontal(m, 2, module0x2, center[2], float32(y), slack)
	if !ok2 {
		return out
	}
	center[2], moduleSize[2] = c2, ms2
	if !mirrorCheckModuleSize3(moduleSize[0], moduleSize[1], moduleSize[2]) {
		return out
	}

	cx := (((center[0] + center[1]) + center[2]) / float32(3))
	ms := (((moduleSize[0] + moduleSize[1]) + moduleSize[2]) / float32(3))
	typ, ok := mirrorClassifyBSI(
		m.bit(rowOffset+int32(center[0]), 0),
		m.bit(rowOffset+int32(center[1]), 1),
		m.bit(rowOffset+int32(center[2]), 2),
	)
	if !ok {
		return out
	}
	survived, fcx, fcy, fms, dir := mirrorCrossCheckPatternBSI(m, typ, cx, float32(y), ms, mirrorChainSlack(ms, printPass))
	if !survived {
		return out
	}
	out.flags |= chainFlagSurvivor
	out.typ = typ
	out.dir = dir
	out.cx = fcx
	out.cy = fcy
	out.ms = fms
	return out
}
