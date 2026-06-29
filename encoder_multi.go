package jabcode

import (
	"errors"
	"fmt"
	"image"
	"slices"

	"github.com/srlehn/jabcode/internal/ecc"
)

// codeParamsMulti holds the geometry of a multi-symbol code (jab_code).
type codeParamsMulti struct {
	dimension           int
	codeSize            image.Point
	minX, minY          int
	rows, cols          int
	rowHeight, colWidth []int
}

// generateMulti runs the encoding pipeline for a multi-symbol code
// (generateJABCode in encoder.c, multi-symbol path).
func (e *Encoder) generateMulti(data []byte) error {
	e.eccLevel = e.symbolECCLevels[0] // primary ECC drives default-mode detection

	if err := e.initSymbols(); err != nil {
		return err
	}

	seq, encodedLength := analyzeInputData(data)
	if seq == nil {
		return errEncode
	}
	encoded, err := encodeData(data, encodedLength, seq)
	if err != nil {
		return err
	}

	if err := e.setSecondaryMetadata(); err != nil {
		return err
	}
	if err := e.fitDataIntoSymbols(encoded); err != nil {
		return err
	}
	if !e.isDefaultMode() {
		e.encodePrimaryMetadata()
	}

	for i := 0; i < e.symbolNumber; i++ {
		codeword := ecc.EncodeLDPC(e.symbols[i].data, e.symbols[i].wcwr[0], e.symbols[i].wcwr[1])
		ecc.Interleave(codeword)
		e.createMatrix(i, codeword)
	}

	cp := e.getCodeParaMulti()
	if e.isDefaultMode() {
		e.maskSymbolsMulti(defaultMaskingReference, nil, nil)
	} else {
		maskRef := e.maskCodeMulti(cp)
		if maskRef != defaultMaskingReference {
			e.updatePrimaryMetadataPartII(maskRef)
			e.placePrimaryMetadataPartII()
		}
	}
	e.createBitmapMulti(cp)
	return nil
}

// initSymbols validates the symbol configuration, moves the primary symbol
// first, assigns docked symbols to hosts and sets side sizes (InitSymbols).
func (e *Encoder) initSymbols() error {
	// Work on copies so reordering does not mutate the caller's option slices.
	e.symbolPositions = slices.Clone(e.symbolPositions)
	e.symbolVersions = slices.Clone(e.symbolVersions)
	e.symbolECCLevels = slices.Clone(e.symbolECCLevels)

	for i := 0; i < e.symbolNumber; i++ {
		v := e.symbolVersions[i]
		if v.X < 1 || v.X > 32 || v.Y < 1 || v.Y > 32 {
			return fmt.Errorf("jabcode: incorrect symbol version for symbol %d", i)
		}
		if e.symbolPositions[i] < 0 || e.symbolPositions[i] >= len(symbolPos) {
			return fmt.Errorf("jabcode: incorrect symbol position %d for symbol %d", e.symbolPositions[i], i)
		}
	}

	e.symbols = make([]symbol, e.symbolNumber)

	if e.symbolPositions[0] != 0 {
		for i := 0; i < e.symbolNumber; i++ {
			if e.symbolPositions[i] == 0 {
				e.symbolPositions[i], e.symbolPositions[0] = e.symbolPositions[0], e.symbolPositions[i]
				e.symbolVersions[i], e.symbolVersions[0] = e.symbolVersions[0], e.symbolVersions[i]
				e.symbolECCLevels[i], e.symbolECCLevels[0] = e.symbolECCLevels[0], e.symbolECCLevels[i]
				break
			}
		}
	}
	if e.symbolPositions[0] != 0 {
		return errors.New("jabcode: primary symbol missing")
	}
	for i := 0; i < e.symbolNumber-1; i++ {
		for j := i + 1; j < e.symbolNumber; j++ {
			if e.symbolPositions[i] == e.symbolPositions[j] {
				return errors.New("jabcode: duplicate symbol position")
			}
		}
	}

	if !e.assignDockedSymbols() {
		return errors.New("jabcode: a secondary symbol has no host")
	}
	if !e.checkDockedSymbolSize() {
		return errors.New("jabcode: docked symbol size does not match its host")
	}
	for i := 0; i < e.symbolNumber; i++ {
		e.symbols[i].index = i
		e.symbols[i].sideSize = image.Pt(version2size(e.symbolVersions[i].X), version2size(e.symbolVersions[i].Y))
	}
	return nil
}

// swapSymbols exchanges two symbols and their configuration (swap_symbols).
func (e *Encoder) swapSymbols(i1, i2 int) {
	e.symbolPositions[i1], e.symbolPositions[i2] = e.symbolPositions[i2], e.symbolPositions[i1]
	e.symbolVersions[i1], e.symbolVersions[i2] = e.symbolVersions[i2], e.symbolVersions[i1]
	e.symbolECCLevels[i1], e.symbolECCLevels[i2] = e.symbolECCLevels[i2], e.symbolECCLevels[i1]
	e.symbols[i1], e.symbols[i2] = e.symbols[i2], e.symbols[i1]
}

// assignDockedSymbols pairs each secondary symbol with the host it docks to
// (assignDockedSymbols in encoder.c).
func (e *Encoder) assignDockedSymbols() bool {
	for i := 0; i < e.symbolNumber; i++ {
		e.symbols[i].host = -1
		e.symbols[i].docked = [4]int{}
	}
	assigned := 1
	for i := 0; i < e.symbolNumber-1 && assigned < e.symbolNumber; i++ {
		for j := 0; j < 4 && assigned < e.symbolNumber; j++ {
			for k := i + 1; k < e.symbolNumber && assigned < e.symbolNumber; k++ {
				if e.symbols[k].host != -1 {
					continue
				}
				hpos := symbolPos[e.symbolPositions[i]]
				spos := symbolPos[e.symbolPositions[k]]
				found := false
				switch j {
				case 0: // top
					if hpos.X == spos.X && hpos.Y-1 == spos.Y {
						e.symbols[i].docked[0], e.symbols[k].docked[1] = assigned, -1
						found = true
					}
				case 1: // bottom
					if hpos.X == spos.X && hpos.Y+1 == spos.Y {
						e.symbols[i].docked[1], e.symbols[k].docked[0] = assigned, -1
						found = true
					}
				case 2: // left
					if hpos.Y == spos.Y && hpos.X-1 == spos.X {
						e.symbols[i].docked[2], e.symbols[k].docked[3] = assigned, -1
						found = true
					}
				case 3: // right
					if hpos.Y == spos.Y && hpos.X+1 == spos.X {
						e.symbols[i].docked[3], e.symbols[k].docked[2] = assigned, -1
						found = true
					}
				}
				if found {
					e.swapSymbols(k, assigned)
					e.symbols[assigned].host = i
					assigned++
				}
			}
		}
	}
	for i := 1; i < e.symbolNumber; i++ {
		if e.symbols[i].host == -1 {
			return false
		}
	}
	return true
}

// checkDockedSymbolSize verifies docked symbols share the side size of their
// host on the docked edge (checkDockedSymbolSize in encoder.c).
func (e *Encoder) checkDockedSymbolSize() bool {
	for i := 0; i < e.symbolNumber; i++ {
		for j := range 4 {
			si := e.symbols[i].docked[j]
			if si <= 0 {
				continue
			}
			hpos := symbolPos[e.symbolPositions[i]]
			spos := symbolPos[e.symbolPositions[si]]
			if hpos.X-spos.X == 0 && e.symbolVersions[i].X != e.symbolVersions[si].X {
				return false
			}
			if hpos.Y-spos.Y == 0 && e.symbolVersions[i].Y != e.symbolVersions[si].Y {
				return false
			}
		}
	}
	return true
}

// setSecondaryMetadata builds each secondary symbol's metadata (SS, SE, version,
// ECC) — setSlaveMetadata in encoder.c.
func (e *Encoder) setSecondaryMetadata() error {
	for i := 1; i < e.symbolNumber; i++ {
		host := e.symbols[i].host
		var ss, se, v, e1, e2 int
		metaLen := 2
		switch {
		case e.symbolVersions[i].X != e.symbolVersions[host].X:
			ss, v, metaLen = 1, e.symbolVersions[i].X-1, metaLen+5
		case e.symbolVersions[i].Y != e.symbolVersions[host].Y:
			ss, v, metaLen = 1, e.symbolVersions[i].Y-1, metaLen+5
		default:
			ss = 0
		}
		if e.symbolECCLevels[i] == 0 || e.symbolECCLevels[i] == e.symbolECCLevels[host] {
			se = 0
		} else {
			se = 1
			e1 = ecclevel2wcwr[e.symbolECCLevels[i]][0] - 3
			e2 = ecclevel2wcwr[e.symbolECCLevels[i]][1] - 4
			metaLen += 6
		}
		md := make([]byte, metaLen)
		md[0] = byte(ss)
		md[1] = byte(se)
		if ss == 1 {
			writeBits(md, v, 2, 5)
		}
		if se == 1 {
			start := 2
			if ss == 1 {
				start = 7
			}
			writeBits(md, e1, start, 3)
			writeBits(md, e2, start+3, 3)
		}
		e.symbols[i].metadata = md
	}
	return nil
}

// addE2SecondaryMetadata appends the E (ECC) field to a secondary symbol's
// metadata (addE2SlaveMetadata in encoder.c).
func (e *Encoder) addE2SecondaryMetadata(s *symbol) {
	old := s.metadata
	md := make([]byte, len(old)+6)
	copy(md, old)
	md[1] = 1 // SE = 1
	writeBits(md, s.wcwr[0]-3, len(old), 3)
	writeBits(md, s.wcwr[1]-4, len(old)+3, 3)
	s.metadata = md
}

// updateSecondaryMetadataE rewrites a secondary symbol's E field inside its
// host's data stream after the code rate was optimized (updateSlaveMetadataE).
func (e *Encoder) updateSecondaryMetadataE(hostIndex, secondaryIndex int) {
	host := &e.symbols[hostIndex]
	sec := &e.symbols[secondaryIndex]
	offset := len(host.data) - 1
	for host.data[offset] == 0 {
		offset--
	}
	offset-- // skip the flag bit
	if hostIndex == 0 {
		offset -= 4
	} else {
		offset -= 3
	}
	for i := range 4 {
		if host.docked[i] == secondaryIndex {
			break
		} else if host.docked[i] <= 0 {
			continue
		}
		offset -= len(e.symbols[host.docked[i]].metadata)
	}
	if sec.metadata[0] == 1 {
		offset -= 7
	} else {
		offset -= 2
	}
	var ebits [6]byte
	writeBits(ebits[:], sec.wcwr[0]-3, 0, 3)
	writeBits(ebits[:], sec.wcwr[1]-4, 3, 3)
	for i := range 6 {
		host.data[offset] = ebits[i]
		offset--
	}
}

// fitDataIntoSymbols distributes the encoded data over the symbols and builds
// each symbol's payload, including its in-stream metadata (fitDataIntoSymbols).
func (e *Encoder) fitDataIntoSymbols(encoded []byte) error {
	n := e.symbolNumber
	capacity := make([]int, n)
	netCap := make([]int, n)
	totalNetCap := 0
	for i := range n {
		version := image.Pt(size2version(e.symbols[i].sideSize.X), size2version(e.symbols[i].sideSize.Y))
		capacity[i] = e.symbolCapacity(version, i == 0)
		e.symbols[i].wcwr = [2]int{ecclevel2wcwr[e.symbolECCLevels[i]][0], ecclevel2wcwr[e.symbolECCLevels[i]][1]}
		netCap[i] = netCapacity(capacity[i], e.symbols[i].wcwr[0], e.symbols[i].wcwr[1])
		totalNetCap += netCap[i]
	}

	assigned := 0
	for i := range n {
		var sDataLength int
		if i == n-1 {
			sDataLength = len(encoded) - assigned
		} else {
			sDataLength = int(float32(netCap[i]) / float32(totalNetCap) * float32(len(encoded)))
		}
		sPayloadLength := sDataLength + 1
		if i == 0 {
			sPayloadLength += 4
		} else {
			sPayloadLength += 3
		}
		for j := range 4 {
			if e.symbols[i].docked[j] > 0 {
				sPayloadLength += len(e.symbols[e.symbols[i].docked[j]].metadata)
			}
		}
		if sPayloadLength > netCap[i] {
			return errors.New("jabcode: message does not fit into the symbols; use higher versions")
		}

		for j := 0; netCap[i]-sPayloadLength >= 6 && j < 4; j++ {
			if e.symbols[i].docked[j] > 0 {
				sym := &e.symbols[e.symbols[i].docked[j]]
				if sym.metadata[1] == 0 { // SE
					e.addE2SecondaryMetadata(sym)
					sPayloadLength += 6
				}
			}
		}

		pnLength := netCap[i]
		if i == 0 {
			if !e.isDefaultMode() {
				getOptimalECC(capacity[i], sPayloadLength, &e.symbols[i].wcwr)
				pnLength = netCapacity(capacity[i], e.symbols[i].wcwr[0], e.symbols[i].wcwr[1])
			}
		} else if e.symbols[i].metadata[1] == 1 { // SE = 1
			getOptimalECC(capacity[i], sPayloadLength, &e.symbols[i].wcwr)
			pnLength = netCapacity(capacity[i], e.symbols[i].wcwr[0], e.symbols[i].wcwr[1])
			e.updateSecondaryMetadataE(e.symbols[i].host, i)
		}

		s := &e.symbols[i]
		s.data = make([]byte, pnLength)
		copy(s.data[:sDataLength], encoded[assigned:assigned+sDataLength])
		assigned += sDataLength

		pos := sPayloadLength - 1
		s.data[pos] = 1 // flag bit
		pos--
		for k := range 4 { // host metadata S
			if e.symbols[i].docked[k] > 0 {
				s.data[pos] = 1
				pos--
			} else if e.symbols[i].docked[k] == 0 {
				s.data[pos] = 0
				pos--
			}
		}
		for k := range 4 { // secondary metadata
			if e.symbols[i].docked[k] > 0 {
				md := e.symbols[e.symbols[i].docked[k]].metadata
				for m := range md {
					s.data[pos] = md[m]
					pos--
				}
			}
		}
	}
	return nil
}

// getCodeParaMulti computes the multi-symbol code geometry (getCodePara).
func (e *Encoder) getCodeParaMulti() codeParamsMulti {
	var cp codeParamsMulti
	if e.symbolNumber == 1 {
		// no master width/height override is exposed; use the module size
	}
	cp.dimension = e.moduleSize

	maxX, maxY := 0, 0
	for i := 0; i < e.symbolNumber; i++ {
		p := symbolPos[e.symbolPositions[i]]
		cp.minX = min(cp.minX, p.X)
		cp.minY = min(cp.minY, p.Y)
		maxX = max(maxX, p.X)
		maxY = max(maxY, p.Y)
	}
	cp.rows = maxY - cp.minY + 1
	cp.cols = maxX - cp.minX + 1
	cp.rowHeight = make([]int, cp.rows)
	cp.colWidth = make([]int, cp.cols)

	for x := cp.minX; x <= maxX; x++ {
		for i := 0; i < e.symbolNumber; i++ {
			if symbolPos[e.symbolPositions[i]].X == x {
				cp.colWidth[x-cp.minX] = e.symbols[i].sideSize.X
				cp.codeSize.X += cp.colWidth[x-cp.minX]
				break
			}
		}
	}
	for y := cp.minY; y <= maxY; y++ {
		for i := 0; i < e.symbolNumber; i++ {
			if symbolPos[e.symbolPositions[i]].Y == y {
				cp.rowHeight[y-cp.minY] = e.symbols[i].sideSize.Y
				cp.codeSize.Y += cp.rowHeight[y-cp.minY]
				break
			}
		}
	}
	return cp
}

// symbolStart returns the top-left module coordinate of symbol k in the code.
func (e *Encoder) symbolStart(k int, cp *codeParamsMulti) (startx, starty int) {
	col := symbolPos[e.symbolPositions[k]].X - cp.minX
	row := symbolPos[e.symbolPositions[k]].Y - cp.minY
	for c := range col {
		startx += cp.colWidth[c]
	}
	for r := range row {
		starty += cp.rowHeight[r]
	}
	return startx, starty
}

// maskSymbolsMulti masks the data modules of all symbols. With a buffer it
// writes the whole code for penalty evaluation; otherwise it masks in place
// (maskSymbols in mask.c).
func (e *Encoder) maskSymbolsMulti(maskType int, masked []int, cp *codeParamsMulti) {
	for k := 0; k < e.symbolNumber; k++ {
		startx, starty := 0, 0
		if masked != nil && cp != nil {
			startx, starty = e.symbolStart(k, cp)
		}
		s := &e.symbols[k]
		w, h := s.sideSize.X, s.sideSize.Y
		for y := range h {
			for x := range w {
				index := int(s.matrix[y*w+x])
				if s.dataMap[y*w+x] != 0 {
					index ^= maskValue(maskType, x, y) % e.colors
					if masked != nil && cp != nil {
						masked[(y+starty)*cp.codeSize.X+(x+startx)] = index
					} else {
						s.matrix[y*w+x] = byte(index)
					}
				} else if masked != nil && cp != nil {
					masked[(y+starty)*cp.codeSize.X+(x+startx)] = index
				}
			}
		}
	}
}

// maskCodeMulti selects the lowest-penalty mask for a multi-symbol code, applies
// it in place and returns its reference (maskCode in mask.c).
func (e *Encoder) maskCodeMulti(cp codeParamsMulti) int {
	maskType := 0
	minPenalty := 10000
	masked := make([]int, cp.codeSize.X*cp.codeSize.Y)
	for i := range masked {
		masked[i] = -1
	}
	for t := range numberOfMaskPatterns {
		e.maskSymbolsMulti(t, masked, &cp)
		if p := evaluateMask(masked, cp.codeSize.X, cp.codeSize.Y, e.colors); p < minPenalty {
			maskType = t
			minPenalty = p
		}
	}
	e.maskSymbolsMulti(maskType, nil, nil)
	return maskType
}

// createBitmapMulti renders all symbols into the code bitmap (createBitmap).
func (e *Encoder) createBitmapMulti(cp codeParamsMulti) {
	width := cp.dimension * cp.codeSize.X
	height := cp.dimension * cp.codeSize.Y
	img := image.NewPaletted(image.Rect(0, 0, width, height), rgbPalette(e.palette))
	for k := 0; k < e.symbolNumber; k++ {
		startx, starty := e.symbolStart(k, &cp)
		s := &e.symbols[k]
		w, h := s.sideSize.X, s.sideSize.Y
		for x := startx; x < startx+w; x++ {
			for y := starty; y < starty+h; y++ {
				idx := s.matrix[(y-starty)*w+(x-startx)]
				for i := y * cp.dimension; i < y*cp.dimension+cp.dimension; i++ {
					for j := x * cp.dimension; j < x*cp.dimension+cp.dimension; j++ {
						img.SetColorIndex(j, i, idx)
					}
				}
			}
		}
	}
	e.bitmap = img
}
