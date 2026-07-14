//go:build jabcode_non_iso_encode

package encode

import (
	"errors"

	"github.com/srlehn/jabcode/internal/ecc"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/tables"
	"github.com/srlehn/jabcode/internal/wire"
)

func (e *encoder) generateBSIMulti(data []byte) error {
	if len(e.symbolPositions) != e.symbolNumber || len(e.symbolVersions) != e.symbolNumber ||
		len(e.symbolECCLevels) != e.symbolNumber {
		return errors.New("jabcode: BSI multi-symbol configuration is incomplete")
	}
	if err := e.initSymbols(); err != nil {
		return err
	}
	if err := e.resolveBSIMultiErrorCorrection(); err != nil {
		return err
	}

	e.symbols[0].metadata = e.encodeBSIPrimaryMetadata(1)
	for index := 1; index < e.symbolNumber; index++ {
		metadata, err := e.encodeBSISecondaryMetadata(index)
		if err != nil {
			return err
		}
		e.symbols[index].metadata = metadata
	}

	netCapacities := make([]int, e.symbolNumber)
	totalNetCapacity := 0
	for index := range e.symbolNumber {
		capacity := e.bsiSymbolCapacityWithMetadata(
			e.symbolVersions[index], index == 0, len(e.symbols[index].metadata),
		)
		wcwr := e.symbols[index].wcwr
		netCapacities[index] = netCapacity(capacity, wcwr[0], wcwr[1])
		if netCapacities[index] <= 0 {
			return errors.New("jabcode: BSI symbol has no payload capacity")
		}
		totalNetCapacity += netCapacities[index]
	}

	seq, minimumLength := analyzeInputData(data)
	if seq == nil {
		return errEncode
	}
	if minimumLength > totalNetCapacity {
		return errors.New("jabcode: message does not fit into the BSI symbols; use higher versions")
	}
	encoded, err := encodeData(data, totalNetCapacity, seq)
	if err != nil {
		return err
	}
	offset := 0
	for index, length := range netCapacities {
		e.symbols[index].data = append(e.symbols[index].data[:0], encoded[offset:offset+length]...)
		offset += length
	}

	paletteIndex := tables.BSIPalettePlacementIndex(min(e.colors, 64), e.colors)
	for index := range e.symbolNumber {
		s := &e.symbols[index]
		codeword := ecc.EncodeLDPCVariant(s.data, s.wcwr[0], s.wcwr[1], wire.BSI)
		ecc.InterleaveVariant(codeword, wire.BSI)
		e.createBSIMatrix(index, s.metadata, codeword, paletteIndex)
	}

	cp := e.codeParaMulti()
	maskReference := e.maskBSICodeMulti(cp, paletteIndex)
	e.symbols[0].metadata = e.encodeBSIPrimaryMetadata(maskReference)
	e.placeBSIPrimaryMetadata(e.symbols[0].metadata, paletteIndex)
	e.createBitmapMulti(cp)
	return nil
}

func (e *encoder) resolveBSIMultiErrorCorrection() error {
	e.eccLevel = e.symbolECCLevels[0]
	for index, level := range e.symbolECCLevels {
		if level < 0 || level >= len(spec.ECCWeights) {
			return errors.New("jabcode: invalid BSI error-correction level")
		}
		if index > 0 && level == 0 {
			e.symbols[index].wcwr = e.symbols[e.symbols[index].host].wcwr
			continue
		}
		if level == 0 {
			// BSI TR-03137-2 defines (4, 7) as the default data code.
			e.symbols[index].wcwr = [2]int{4, 7}
			continue
		}
		e.symbols[index].wcwr = spec.ECCWeights[level]
	}
	return nil
}

func (e *encoder) encodeBSISecondaryMetadata(index int) ([]byte, error) {
	s := &e.symbols[index]
	if s.host < 0 || s.host >= index {
		return nil, errors.New("jabcode: BSI secondary symbol has no preceding host")
	}
	host := &e.symbols[s.host]
	hostPosition := bsiSecondaryHostPosition(s)
	if hostPosition < 0 {
		return nil, errors.New("jabcode: BSI secondary symbol has no host side")
	}

	ss := s.sideSize != host.sideSize
	se := s.wcwr != host.wcwr
	sf := false
	for _, docked := range s.docked {
		if docked > 0 {
			sf = true
			break
		}
	}
	part1 := make([]byte, 3)
	if ss {
		part1[0] = 1
	}
	if se {
		part1[1] = 1
	}
	if sf {
		part1[2] = 1
	}

	part2Length := 0
	if ss {
		part2Length += 5
	}
	if sf {
		part2Length += 3
	}
	part2 := make([]byte, part2Length)
	bitIndex := 0
	if ss {
		version := e.symbolVersions[index].Y
		if hostPosition == 2 || hostPosition == 3 {
			version = e.symbolVersions[index].X
		}
		writeBits(part2, version-1, bitIndex, 5)
		bitIndex += 5
	}
	if sf {
		for position, docked := range s.docked {
			if position == hostPosition {
				continue
			}
			if docked > 0 {
				part2[bitIndex] = 1
			}
			bitIndex++
		}
	}

	var part3 []byte
	if se {
		length := spec.BSIErrorCorrectionBitLength(e.symbolVersions[index])
		part3 = make([]byte, length)
		half := length / 2
		writeBits(part3, s.wcwr[0]-3, 0, half)
		writeBits(part3, s.wcwr[1]-4, half, half)
	}

	encoded1 := ecc.EncodeLDPCVariant(part1, 2, -1, wire.BSI)
	metadata := make([]byte, 0, len(encoded1)+len(part2)*2+len(part3)*2)
	metadata = append(metadata, encoded1...)
	if len(part2) > 0 {
		metadata = append(metadata, ecc.EncodeLDPCVariant(part2, 2, -1, wire.BSI)...)
	}
	if len(part3) > 0 {
		metadata = append(metadata, ecc.EncodeLDPCVariant(part3, 2, -1, wire.BSI)...)
	}
	return metadata, nil
}

func bsiSecondaryHostPosition(s *symbol) int {
	for position, docked := range s.docked {
		if docked < 0 {
			return position
		}
	}
	return -1
}

func (e *encoder) placeBSISecondaryFinderPatterns(s *symbol, set func(int, int, int), paletteIndex []int) {
	width, height := s.sideSize.X, s.sideSize.Y
	const distance = spec.DistanceToBorder
	colors := [2]int{paletteIndex[0], paletteIndex[min(e.colors-1, 7)]}
	for layer := range 2 {
		for i := 0; i < layer+1; i++ {
			for j := 0; j < layer+1; j++ {
				if i != layer && j != layer {
					continue
				}
				colorIndex := colors[layer]
				set(distance-j-1, distance-(i+1), colorIndex)
				set(distance+j-1, distance+(i-1), colorIndex)
				set(width-(distance-1)-j-1, distance-(i+1), colorIndex)
				set(width-(distance-1)+j-1, distance+(i-1), colorIndex)
				set(width-(distance-1)-j-1, height-distance+i, colorIndex)
				set(width-(distance-1)+j-1, height-distance-i, colorIndex)
				set(distance-j-1, height-distance+i, colorIndex)
				set(distance+j-1, height-distance-i, colorIndex)
			}
		}
	}
}

func (e *encoder) placeBSISecondaryMetadataAndPalette(index int, metadata []byte, set func(int, int, int), paletteIndex []int) {
	s := &e.symbols[index]
	width, height := s.sideSize.X, s.sideSize.Y
	hostPosition := bsiSecondaryHostPosition(s)
	metadataColors := min(e.colors, 8)
	bitsPerMetadataModule := spec.Log2Int(metadataColors)
	x, y := 0, 1
	moduleCount := 0
	for bitIndex := 0; bitIndex < len(metadata); {
		colorIndex := 0
		for bit := range bitsPerMetadataModule {
			if bitIndex < len(metadata) {
				colorIndex |= int(metadata[bitIndex]) << (bitsPerMetadataModule - 1 - bit)
			}
			bitIndex++
		}
		moduleX, moduleY := bsiSecondaryMetadataPosition(hostPosition, width, height, x, y)
		set(moduleX, moduleY, paletteIndex[colorIndex])
		moduleCount++
		spec.NextMetadataModuleInSecondary(moduleCount, &x, &y)
	}

	available := min(e.colors, 64)
	for colorIndex := range available {
		moduleX, moduleY := bsiSecondaryPalettePosition(hostPosition, width, height, available, colorIndex)
		set(moduleX, moduleY, paletteIndex[colorIndex])
		set(width-1-moduleX, height-1-moduleY, paletteIndex[colorIndex])
	}
}

func bsiSecondaryMetadataPosition(hostPosition, width, height, x, y int) (int, int) {
	switch hostPosition {
	case 3:
		return width - 1 - x, height - 1 - y
	case 0:
		return width - 1 - y, x
	case 1:
		return y, height - 1 - x
	default:
		return x, y
	}
}

func bsiSecondaryPalettePosition(hostPosition, width, height, available, colorIndex int) (int, int) {
	positionIndex := colorIndex
	if available > 8 && colorIndex >= available/2 {
		positionIndex -= available / 2
	}
	position := tables.SecondaryPalettePosition[positionIndex]
	x, y := position.X, position.Y
	if hostPosition == 2 || hostPosition == 3 {
		if available > 8 && colorIndex >= available/2 {
			if width > height {
				x, y = position.Y, height-1-position.X
			} else {
				x, y = width-1-position.Y, position.X
			}
		}
	} else if available <= 8 || colorIndex < available/2 {
		if width > height {
			x, y = position.Y, height-1-position.X
		} else {
			x, y = width-1-position.Y, position.X
		}
	}
	return x, y
}

func (e *encoder) maskBSICodeMulti(cp codeParamsMulti, paletteIndex []int) int {
	maskReference, minimumPenalty := 0, 10000
	masked := make([]int, cp.codeSize.X*cp.codeSize.Y)
	for i := range masked {
		masked[i] = -1
	}
	for candidate := range numberOfMaskPatterns {
		e.maskSymbolsMulti(candidate, masked, &cp)
		penalty := bsiMaskPenalty(masked, cp.codeSize.X, cp.codeSize.Y, e.colors, paletteIndex)
		if penalty < minimumPenalty {
			maskReference, minimumPenalty = candidate, penalty
		}
	}
	e.maskSymbolsMulti(maskReference, nil, nil)
	return maskReference
}
