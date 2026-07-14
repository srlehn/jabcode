//go:build jabcode_bsi

package encode

import (
	"errors"
	"fmt"
	"image"
	"math"

	"github.com/srlehn/jabcode/internal/ecc"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/wire"
)

var bsiPrimaryPalettePositions = [8]image.Point{
	image.Pt(4, 1), image.Pt(4, 2), image.Pt(5, 1), image.Pt(5, 2),
	image.Pt(2, 4), image.Pt(2, 5), image.Pt(1, 4), image.Pt(1, 5),
}

func (e *encoder) generateBSI(data []byte) error {
	if e.symbolNumber > 1 {
		return errors.New("jabcode: BSI multi-symbol encoding is not implemented")
	}
	e.symbols = []symbol{{index: 0, host: -1}}

	seq, minimumLength := analyzeInputData(data)
	if seq == nil {
		return errEncode
	}
	version, capacity, err := e.selectBSIPrimaryVersion(minimumLength)
	if err != nil {
		return err
	}
	s := &e.symbols[0]
	s.sideSize = image.Pt(spec.VersionToSize(version.X), spec.VersionToSize(version.Y))
	s.wcwr = [2]int{5, 6}
	optimalECC(capacity, minimumLength, &s.wcwr)
	if s.wcwr[0] < 3 || s.wcwr[0] >= s.wcwr[1] {
		return errors.New("jabcode: no BSI error-correction parameters fit the message")
	}
	netLength := netCapacity(capacity, s.wcwr[0], s.wcwr[1])
	encoded, err := encodeData(data, netLength, seq)
	if err != nil {
		return err
	}
	s.data = encoded

	codeword := ecc.EncodeLDPCProfile(s.data, s.wcwr[0], s.wcwr[1], wire.BSI)
	ecc.InterleaveProfile(codeword, wire.BSI)
	paletteIndex := bsiPalettePlacementIndex(min(e.colors, 64), e.colors)
	metadata := e.encodeBSIPrimaryMetadata(1)
	e.createBSIMatrix(metadata, codeword, paletteIndex)
	maskReference := e.maskBSICode(paletteIndex)
	metadata = e.encodeBSIPrimaryMetadata(maskReference)
	e.placeBSIPrimaryMetadata(metadata, paletteIndex)
	e.createBitmap()
	return nil
}

func (e *encoder) selectBSIPrimaryVersion(minimumLength int) (image.Point, int, error) {
	if len(e.symbolVersions) > 0 && e.symbolVersions[0] != (image.Point{}) {
		version := e.symbolVersions[0]
		if version.X < 1 || version.X > 32 || version.Y < 1 || version.Y > 32 {
			return image.Point{}, 0, fmt.Errorf("jabcode: incorrect symbol version %dx%d for the primary symbol", version.X, version.Y)
		}
		capacity := e.bsiSymbolCapacity(version, true)
		if capacity < e.bsiRequiredGrossLength(minimumLength) {
			return image.Point{}, 0, errors.New("jabcode: message does not fit; use a higher BSI symbol version")
		}
		return version, capacity, nil
	}

	required := e.bsiRequiredGrossLength(minimumLength)
	for version := 1; version <= 32; version++ {
		v := image.Pt(version, version)
		capacity := e.bsiSymbolCapacity(v, true)
		if capacity >= required {
			return v, capacity, nil
		}
	}
	return image.Point{}, 0, errors.New("jabcode: message does not fit into one BSI symbol; use more symbols")
}

func (e *encoder) bsiRequiredGrossLength(netLength int) int {
	if e.eccLevel == 0 {
		return int(math.Ceil(float64(netLength*6) / float64(6-5)))
	}
	p := float32(e.eccLevel*5) / 100
	if p <= 0 || p >= 1 {
		return math.MaxInt
	}
	h := -p*float32(math.Log(float64(p))) - (1-p)*float32(math.Log(float64(1-p)))
	codeRate := float32(0)
	for i := 4; i < 1<<8; i++ {
		pi := (1 + float32(math.Pow(float64(1-2*p), float64(i)))) / 2
		hpi := -pi*float32(math.Log(float64(pi))) - (1-pi)*float32(math.Log(float64(1-pi)))
		if hpi != 0 {
			candidate := (hpi - h) / hpi
			if candidate > codeRate {
				codeRate = candidate
			}
		}
	}
	if codeRate <= 0 {
		return math.MaxInt
	}
	return int(math.Ceil(float64(float32(netLength) / codeRate)))
}

func (e *encoder) bsiSymbolCapacity(version image.Point, primary bool) int {
	width := spec.VersionToSize(version.X)
	height := spec.VersionToSize(version.Y)
	patternsX, patternsY := bsiPatternCounts(width, height)
	reservedPatterns := patternsX * patternsY * 7
	if primary {
		reservedPatterns += 4 * 10
	}
	reservedPalette := min(e.colors, 64) * 2
	reservedMetadata := 0
	if primary {
		metadataBits := bsiPrimaryMetadataBitLength(version, false)
		metadataBitsPerModule := min(spec.Log2Int(e.colors), 3)
		reservedMetadata = 6 + (metadataBits-6+metadataBitsPerModule-1)/metadataBitsPerModule
	}
	return (width*height - reservedPatterns - reservedPalette - reservedMetadata) * spec.Log2Int(e.colors)
}

func bsiPatternCounts(width, height int) (int, int) {
	const minimumAlignmentDistance = 16
	patternsX := (width-(spec.DistanceToBorder*2-1))/minimumAlignmentDistance - 1
	patternsY := (height-(spec.DistanceToBorder*2-1))/minimumAlignmentDistance - 1
	return max(patternsX, 0) + 2, max(patternsY, 0) + 2
}

func bsiPrimaryMetadataBitLength(version image.Point, docked bool) int {
	_, vf, versionLength, _ := bsiVersionFields(version)
	rawPart3 := versionLength + vf*2 + 10
	if docked {
		rawPart3 += 4
	}
	return 6 + 14 + rawPart3*2
}

func bsiVersionFields(version image.Point) (square, vf, length, correction int) {
	square = 0
	if version.X != version.Y {
		square = 1
	}
	longer := max(version.X, version.Y)
	switch {
	case longer < 5:
		vf, correction = 0, 0
	case longer < 9:
		vf, correction = 1, 4
	case longer < 17:
		vf, correction = 2, 8
	default:
		vf, correction = 3, 16
	}
	if square == 0 {
		length = vf + 1
		if vf == 0 {
			length = 2
		}
	} else {
		length = vf*2 + 4
	}
	return square, vf, length, correction
}

func (e *encoder) encodeBSIPrimaryMetadata(maskReference int) []byte {
	s := &e.symbols[0]
	version := image.Pt(spec.SizeToVersion(s.sideSize.X), spec.SizeToVersion(s.sideSize.Y))
	ss, vf, versionLength, correction := bsiVersionFields(version)

	part1 := make([]byte, 3)
	writeBits(part1, spec.Log2Int(e.colors)-1, 0, 3)
	part2 := make([]byte, 7)
	part2[0] = byte(ss)
	writeBits(part2, vf, 1, 2)
	writeBits(part2, maskReference, 3, 3)
	part2[6] = 0

	eclLength := vf*2 + 10
	part3 := make([]byte, versionLength+eclLength)
	if ss == 0 {
		writeBits(part3, version.X-correction-1, 0, versionLength)
	} else {
		half := versionLength / 2
		writeBits(part3, version.X-1, 0, half)
		writeBits(part3, version.Y-1, half, half)
	}
	halfECL := eclLength / 2
	writeBits(part3, s.wcwr[0]-3, versionLength, halfECL)
	writeBits(part3, s.wcwr[1]-4, versionLength+halfECL, halfECL)

	encoded1 := ecc.EncodeLDPCProfile(part1, 2, -1, wire.BSI)
	encoded2 := ecc.EncodeLDPCProfile(part2, 2, -1, wire.BSI)
	encoded3 := ecc.EncodeLDPCProfile(part3, 2, -1, wire.BSI)
	metadata := make([]byte, 0, len(encoded1)+len(encoded2)+len(encoded3))
	metadata = append(metadata, encoded1...)
	metadata = append(metadata, encoded2...)
	metadata = append(metadata, encoded3...)
	return metadata
}

func (e *encoder) createBSIMatrix(metadata, codeword []byte, paletteIndex []int) {
	s := &e.symbols[0]
	width, height := s.sideSize.X, s.sideSize.Y
	s.matrix = make([]byte, width*height)
	s.dataMap = make([]byte, width*height)
	for i := range s.dataMap {
		s.dataMap[i] = 1
	}
	set := func(x, y int, colorIndex int) {
		s.matrix[y*width+x] = byte(colorIndex)
		s.dataMap[y*width+x] = 0
	}

	e.placeBSIAlignmentPatterns(s, set, paletteIndex)
	e.placeBSIPrimaryFinderPatterns(s, set, paletteIndex)
	e.placeBSIPrimaryMetadataAndPalette(metadata, set, paletteIndex)
	e.placeData(s, codeword)
}

func (e *encoder) placeBSIAlignmentPatterns(s *symbol, set func(int, int, int), paletteIndex []int) {
	width, height := s.sideSize.X, s.sideSize.Y
	patternsX, patternsY := bsiPatternCounts(width, height)
	distanceX := float32(width-(spec.DistanceToBorder*2-1)) / float32(patternsX-1)
	distanceY := float32(height-(spec.DistanceToBorder*2-1)) / float32(patternsY-1)
	core, periphery := 0, 0
	switch e.colors {
	case 4:
		core = 3
	default:
		core = paletteIndex[7]
		periphery = paletteIndex[0]
	}
	for xIndex := range patternsX {
		for yIndex := range patternsY {
			if (xIndex == 0 || xIndex == patternsX-1) && (yIndex == 0 || yIndex == patternsY-1) {
				continue
			}
			x := spec.DistanceToBorder - 1 + int(float32(xIndex)*distanceX)
			y := spec.DistanceToBorder - 1 + int(float32(yIndex)*distanceY)
			set(x, y, core)
			set(x-1, y, periphery)
			set(x+1, y, periphery)
			set(x, y-1, periphery)
			set(x, y+1, periphery)
			if xIndex%2 == yIndex%2 {
				set(x-1, y-1, periphery)
				set(x+1, y+1, periphery)
			} else {
				set(x+1, y-1, periphery)
				set(x-1, y+1, periphery)
			}
		}
	}
}

func (e *encoder) placeBSIPrimaryFinderPatterns(s *symbol, set func(int, int, int), paletteIndex []int) {
	width, height := s.sideSize.X, s.sideSize.Y
	const distance = spec.DistanceToBorder
	cores := [4]int{1, 2, 5, 6}
	for layer := range 3 {
		var colors [4]int
		if e.colors == 4 {
			for i := range colors {
				colors[i] = absInt(layer%2*3 - i)
			}
		} else {
			for i, core := range cores {
				colors[i] = paletteIndex[absInt(layer%2*7-core)]
			}
		}
		for i := 0; i < layer+1; i++ {
			for j := 0; j < layer+1; j++ {
				if i != layer && j != layer {
					continue
				}
				set(distance-j-1, distance-(i+1), colors[0])
				set(distance+j-1, distance+(i-1), colors[0])
				set(width-(distance-1)-j-1, distance-(i+1), colors[1])
				set(width-(distance-1)+j-1, distance+(i-1), colors[1])
				set(width-(distance-1)-j-1, height-distance+i, colors[2])
				set(width-(distance-1)+j-1, height-distance-i, colors[2])
				set(distance-j-1, height-distance+i, colors[3])
				set(distance+j-1, height-distance-i, colors[3])
			}
		}
	}
}

func (e *encoder) placeBSIPrimaryMetadataAndPalette(metadata []byte, set func(int, int, int), paletteIndex []int) {
	s := &e.symbols[0]
	width, height := s.sideSize.X, s.sideSize.Y
	x, y := spec.PrimaryMetadataX, spec.PrimaryMetadataY
	moduleCount := 0
	metadataColors := min(e.colors, 8)
	metadataBitsPerModule := spec.Log2Int(metadataColors)
	bitIndex := 0
	for bitIndex < len(metadata) {
		colorIndex := 0
		if bitIndex < 6 {
			colorIndex = int(metadata[bitIndex]) * (metadataColors - 1)
			bitIndex++
		} else {
			for bit := range metadataBitsPerModule {
				if bitIndex < len(metadata) {
					colorIndex |= int(metadata[bitIndex]) << (metadataBitsPerModule - 1 - bit)
				}
				bitIndex++
			}
		}
		set(x, y, paletteIndex[colorIndex])
		moduleCount++
		spec.NextMetadataModuleInPrimary(height, width, moduleCount, &x, &y)
	}

	available := min(e.colors, 64)
	for colorIndex := 0; colorIndex < min(available, 16); colorIndex++ {
		position := bsiPrimaryPalettePositions[colorIndex%8]
		offset := 0
		if colorIndex >= 8 {
			offset = width - 7
		}
		set(position.X+offset, position.Y, paletteIndex[colorIndex])
		set(width-1-position.X-offset, position.Y+height-7, paletteIndex[colorIndex])
	}
	for colorIndex := 16; colorIndex < available-1; colorIndex += 2 {
		for range 2 {
			set(x, y, paletteIndex[colorIndex])
			moduleCount++
			spec.NextMetadataModuleInPrimary(height, width, moduleCount, &x, &y)
			set(x, y, paletteIndex[colorIndex+1])
			moduleCount++
			spec.NextMetadataModuleInPrimary(height, width, moduleCount, &x, &y)
		}
	}
}

func (e *encoder) placeBSIPrimaryMetadata(metadata []byte, paletteIndex []int) {
	s := &e.symbols[0]
	width, height := s.sideSize.X, s.sideSize.Y
	x, y := spec.PrimaryMetadataX, spec.PrimaryMetadataY
	moduleCount := 0
	metadataColors := min(e.colors, 8)
	bitsPerModule := spec.Log2Int(metadataColors)
	bitIndex := 0
	for bitIndex < len(metadata) {
		colorIndex := 0
		if bitIndex < 6 {
			colorIndex = int(metadata[bitIndex]) * (metadataColors - 1)
			bitIndex++
		} else {
			for bit := range bitsPerModule {
				if bitIndex < len(metadata) {
					colorIndex |= int(metadata[bitIndex]) << (bitsPerModule - 1 - bit)
				}
				bitIndex++
			}
		}
		s.matrix[y*width+x] = byte(paletteIndex[colorIndex])
		moduleCount++
		spec.NextMetadataModuleInPrimary(height, width, moduleCount, &x, &y)
	}
}

func (e *encoder) maskBSICode(paletteIndex []int) int {
	maskReference, minimumPenalty := 0, 10000
	params := e.codePara()
	for candidate := range numberOfMaskPatterns {
		masked := e.maskSymbolsToBuffer(candidate, params)
		penalty := applyBSIRule1(masked, params.codeSize.X, params.codeSize.Y, e.colors, paletteIndex) +
			applyRule2(masked, params.codeSize.X, params.codeSize.Y) +
			applyRule3(masked, params.codeSize.X, params.codeSize.Y)
		if penalty < minimumPenalty {
			maskReference, minimumPenalty = candidate, penalty
		}
	}
	e.maskSymbol(0, maskReference)
	return maskReference
}

func applyBSIRule1(matrix []int, width, height, colorNumber int, paletteIndex []int) int {
	first := [4]int{}
	second := [4]int{}
	switch colorNumber {
	case 4:
		first = [4]int{0, 1, 2, 3}
		second = [4]int{3, 2, 1, 0}
	default:
		cores := [4]int{1, 2, 5, 6}
		for i, core := range cores {
			first[i] = paletteIndex[core]
			second[i] = paletteIndex[7-core]
		}
	}
	at := func(x, y int) int { return matrix[y*width+x] }
	score := 0
	for y := 2; y < height-2; y++ {
		for x := 2; x < width-2; x++ {
			for finder := range 4 {
				a, b := first[finder], second[finder]
				if at(x-2, y) == a && at(x-1, y) == b && at(x, y) == a && at(x+1, y) == b && at(x+2, y) == a &&
					at(x, y-2) == a && at(x, y-1) == b && at(x, y) == a && at(x, y+1) == b && at(x, y+2) == a {
					score++
					break
				}
			}
		}
	}
	return maskW1 * score
}

func bsiPalettePlacementIndex(size, colorNumber int) []int {
	index := make([]int, size)
	for i := range index {
		index[i] = i
	}
	switch colorNumber {
	case 16:
		for i := range 4 {
			index[4+i] = 12 + i
		}
		for i := range 8 {
			index[8+i] = 4 + i
		}
	case 32:
		copy(index[2:4], []int{6, 7})
		copy(index[4:6], []int{24, 25})
		copy(index[6:8], []int{30, 31})
		for i := range 4 {
			index[8+i] = 2 + i
		}
		for i := range 16 {
			index[12+i] = 8 + i
		}
		for i := range 4 {
			index[28+i] = 26 + i
		}
	case 64:
		copy(index[1:8], []int{3, 12, 15, 48, 51, 60, 63})
		for i := range 2 {
			index[8+i] = 1 + i
		}
		for i := range 8 {
			index[10+i] = 4 + i
		}
		for i := range 2 {
			index[18+i] = 13 + i
		}
		for i := range 32 {
			index[20+i] = 16 + i
		}
		for i := range 2 {
			index[52+i] = 49 + i
		}
		for i := range 8 {
			index[54+i] = 52 + i
		}
		for i := range 2 {
			index[62+i] = 61 + i
		}
	case 128:
		copy(index[1:8], []int{3, 12, 15, 112, 115, 124, 127})
		for i := range 2 {
			index[8+i] = 1 + i
		}
		for i := range 8 {
			index[10+i] = 4 + i
		}
		for i := range 2 {
			index[18+i] = 13 + i
		}
		for i := range 16 {
			index[20+i] = 32 + i
			index[36+i] = 80 + i
		}
		for i := range 2 {
			index[52+i] = 113 + i
		}
		for i := range 8 {
			index[54+i] = 116 + i
		}
		for i := range 2 {
			index[62+i] = 125 + i
		}
	case 256:
		copy(index[1:8], []int{3, 28, 31, 224, 227, 252, 255})
		for i := range 2 {
			index[8+i] = 1 + i
			index[18+i] = 29 + i
			index[52+i] = 225 + i
			index[62+i] = 253 + i
		}
		for i := range 4 {
			index[10+i] = 8 + i
			index[14+i] = 20 + i
			index[20+i] = 64 + i
			index[24+i] = 72 + i
			index[28+i] = 84 + i
			index[32+i] = 92 + i
			index[36+i] = 160 + i
			index[40+i] = 168 + i
			index[44+i] = 180 + i
			index[48+i] = 188 + i
			index[54+i] = 232 + i
			index[58+i] = 244 + i
		}
	}
	return index
}

func absInt(value int) int {
	if value < 0 {
		return -value
	}
	return value
}
