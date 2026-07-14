//go:build jabcode_legacy

package decode

import (
	"image"
	"math"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/ecc"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/wire"
)

const (
	preV2CPrimaryMetadataPart1Length = 6
	preV2CPrimaryMetadataPart2Length = 12
	preV2CPaletteCopies              = 2
	preV2CLogicalPaletteCopies       = 4
)

var preV2CSecondaryPalettePositions = [32]image.Point{
	image.Pt(4, 5), image.Pt(4, 6), image.Pt(4, 7), image.Pt(4, 8),
	image.Pt(4, 9), image.Pt(4, 10), image.Pt(4, 11), image.Pt(4, 12),
	image.Pt(5, 12), image.Pt(5, 11), image.Pt(5, 10), image.Pt(5, 9),
	image.Pt(5, 8), image.Pt(5, 7), image.Pt(5, 6), image.Pt(5, 5),
	image.Pt(6, 5), image.Pt(6, 6), image.Pt(6, 7), image.Pt(6, 8),
	image.Pt(6, 9), image.Pt(6, 10), image.Pt(6, 11), image.Pt(6, 12),
	image.Pt(7, 12), image.Pt(7, 11), image.Pt(7, 10), image.Pt(7, 9),
	image.Pt(7, 8), image.Pt(7, 7), image.Pt(7, 6), image.Pt(7, 5),
}

// DecodePreV2CPrimary decodes the primary-symbol wire layout emitted by
// pre-v2.0 JAB Code releases of the C reference implementation. The
// caller must already have identified the BSI-era finder family; this function
// does not weaken current-format metadata admission.
func DecodePreV2CPrimary(matrix *core.Bitmap, symbol *core.DecodedSymbol) int {
	if matrix == nil || !spec.ValidSideSize(matrix.Width) || !spec.ValidSideSize(matrix.Height) {
		return core.Failure
	}
	symbol.WireVariant = wire.PreV2C
	symbol.SideSize = image.Pt(matrix.Width, matrix.Height)
	dataMap := make([]byte, matrix.Width*matrix.Height)

	ret := decodePreV2CPrimaryMetadata(matrix, symbol, dataMap)
	if ret == MetadataFailed {
		clear(dataMap)
		LoadDefaultPrimaryMetadata(matrix, symbol)
		x, y, moduleCount := spec.PrimaryMetadataX, spec.PrimaryMetadataY, 0
		if readPreV2CPrimaryPalette(matrix, symbol, dataMap, &moduleCount, &x, &y) != core.Success {
			return core.Failure
		}
	} else if ret != core.Success {
		return ret
	}

	colorNumber := 1 << (symbol.Meta.NC + 1)
	normPalette := make([]float64, colorNumber*4*preV2CLogicalPaletteCopies)
	normalizePreV2CPalette(symbol.Palette, normPalette, colorNumber)
	palThs := preV2CPaletteThresholds(symbol.Palette, colorNumber)
	return decodePreV2CSymbol(matrix, symbol, dataMap, normPalette, palThs, 0)
}

// DecodePreV2CSecondary decodes a sampled JAB Code secondary symbol emitted by
// the pre-v2.0 C reference implementation. Its wire metadata must already have
// been recovered from the host primary data stream.
func DecodePreV2CSecondary(matrix *core.Bitmap, symbol *core.DecodedSymbol) int {
	if matrix == nil || !spec.ValidSideSize(matrix.Width) || !spec.ValidSideSize(matrix.Height) {
		return core.Failure
	}
	expected := image.Pt(
		spec.VersionToSize(symbol.Meta.SideVersion.X),
		spec.VersionToSize(symbol.Meta.SideVersion.Y),
	)
	if expected.X != matrix.Width || expected.Y != matrix.Height {
		return core.Failure
	}
	symbol.WireVariant = wire.PreV2C
	symbol.SideSize = expected
	dataMap := make([]byte, matrix.Width*matrix.Height)
	if readPreV2CSecondaryPalette(matrix, symbol, dataMap) != core.Success {
		return core.Failure
	}
	colorNumber := 1 << (symbol.Meta.NC + 1)
	normPalette := make([]float64, colorNumber*4*preV2CLogicalPaletteCopies)
	normalizePreV2CPalette(symbol.Palette, normPalette, colorNumber)
	palThs := preV2CPaletteThresholds(symbol.Palette, colorNumber)
	return decodePreV2CSymbol(matrix, symbol, dataMap, normPalette, palThs, 1)
}

func decodePreV2CPrimaryMetadata(matrix *core.Bitmap, symbol *core.DecodedSymbol, dataMap []byte) int {
	x, y, moduleCount := spec.PrimaryMetadataX, spec.PrimaryMetadataY, 0
	part1 := make([]byte, preV2CPrimaryMetadataPart1Length)
	for i := range part1 {
		if !preV2CMetadataPositionValid(matrix, x, y) {
			return core.Failure
		}
		off := (y*matrix.Width + x) * matrix.Channels
		rgb := matrix.Pix[off : off+3]
		moduleColor := DecodeModuleNC(rgb)
		if moduleColor != 0 && moduleColor != 7 {
			return MetadataFailed
		}
		if moduleColor == 7 {
			part1[i] = 1
		}
		dataMap[y*matrix.Width+x] = 1
		moduleCount++
		spec.NextMetadataModuleInPrimary(matrix.Height, matrix.Width, moduleCount, &x, &y)
	}
	part1Decoded, ok := ecc.DecodeLDPCHardVariant(part1, 3, 0, wire.PreV2C)
	if !ok || len(part1Decoded) < 3 {
		return MetadataFailed
	}
	symbol.Meta.NC = int(part1Decoded[0])<<2 | int(part1Decoded[1])<<1 | int(part1Decoded[2])
	if symbol.Meta.NC < 0 || symbol.Meta.NC > 7 {
		return MetadataFailed
	}

	if readPreV2CPrimaryPalette(matrix, symbol, dataMap, &moduleCount, &x, &y) != core.Success {
		return core.Failure
	}
	colorNumber := 1 << (symbol.Meta.NC + 1)
	normPalette := make([]float64, colorNumber*4*preV2CLogicalPaletteCopies)
	normalizePreV2CPalette(symbol.Palette, normPalette, colorNumber)
	palThs := preV2CPaletteThresholds(symbol.Palette, colorNumber)

	part2 := make([]byte, preV2CPrimaryMetadataPart2Length)
	part3 := make([]byte, 0, 32)
	part2Count := 0
	for part2Count < len(part2) {
		if !preV2CMetadataPositionValid(matrix, x, y) {
			return core.Failure
		}
		bits := decodePreV2CModuleHD(matrix, symbol.Palette, colorNumber, normPalette, palThs, x, y)
		for i := 0; i < symbol.Meta.NC+1; i++ {
			bit := (bits >> (symbol.Meta.NC - i)) & 1
			if part2Count < len(part2) {
				part2[part2Count] = bit
				part2Count++
			} else {
				part3 = append(part3, bit)
			}
		}
		dataMap[y*matrix.Width+x] = 1
		moduleCount++
		spec.NextMetadataModuleInPrimary(matrix.Height, matrix.Width, moduleCount, &x, &y)
	}
	part2Decoded, ok := ecc.DecodeLDPCHardVariant(part2, 3, 0, wire.PreV2C)
	if !ok || len(part2Decoded) < 6 {
		return MetadataFailed
	}
	ss := int(part2Decoded[0])
	vf := int(part2Decoded[1])<<1 | int(part2Decoded[2])
	symbol.Meta.MaskType = int(part2Decoded[3])<<2 | int(part2Decoded[4])<<1 | int(part2Decoded[5])

	var versionLength int
	if ss == 0 {
		if vf == 0 {
			versionLength = 2
		} else {
			versionLength = vf + 1
		}
	} else {
		versionLength = vf*2 + 4
	}
	part3Length := versionLength*2 + 12
	for len(part3) < part3Length {
		if !preV2CMetadataPositionValid(matrix, x, y) {
			return core.Failure
		}
		bits := decodePreV2CModuleHD(matrix, symbol.Palette, colorNumber, normPalette, palThs, x, y)
		for i := 0; i < symbol.Meta.NC+1 && len(part3) < part3Length; i++ {
			part3 = append(part3, (bits>>(symbol.Meta.NC-i))&1)
		}
		dataMap[y*matrix.Width+x] = 1
		moduleCount++
		spec.NextMetadataModuleInPrimary(matrix.Height, matrix.Width, moduleCount, &x, &y)
	}
	wc := 3
	if part3Length > 36 {
		wc = 4
	}
	part3Decoded, ok := ecc.DecodeLDPCHardVariant(part3, wc, 0, wire.PreV2C)
	if !ok || len(part3Decoded) < versionLength+6 {
		return MetadataFailed
	}

	bitIndex := 0
	if ss == 0 {
		v := preV2CBitsValue(part3Decoded[:versionLength])
		if vf == 0 {
			v++
		} else {
			v += 1<<(vf+1) + 1
		}
		symbol.Meta.SideVersion = image.Pt(v, v)
	} else {
		half := versionLength / 2
		symbol.Meta.SideVersion.X = preV2CBitsValue(part3Decoded[:half]) + 1
		symbol.Meta.SideVersion.Y = preV2CBitsValue(part3Decoded[half:versionLength]) + 1
	}
	bitIndex += versionLength
	symbol.Meta.ECL.X = preV2CBitsValue(part3Decoded[bitIndex:bitIndex+3]) + 3
	bitIndex += 3
	symbol.Meta.ECL.Y = preV2CBitsValue(part3Decoded[bitIndex:bitIndex+3]) + 4
	symbol.Meta.DockedPosition = 0
	symbol.Meta.DefaultMode = false
	symbol.SideSize = image.Pt(
		spec.VersionToSize(symbol.Meta.SideVersion.X),
		spec.VersionToSize(symbol.Meta.SideVersion.Y),
	)
	if matrix.Width != symbol.SideSize.X || matrix.Height != symbol.SideSize.Y {
		return core.Failure
	}
	if symbol.Meta.ECL.X >= symbol.Meta.ECL.Y {
		return MetadataFailed
	}
	return core.Success
}

func preV2CMetadataPositionValid(matrix *core.Bitmap, x, y int) bool {
	return x >= 0 && y >= 0 && x < matrix.Width && y < matrix.Height
}

func preV2CBitsValue(bits []byte) int {
	v := 0
	for _, bit := range bits {
		v = v<<1 | int(bit)
	}
	return v
}

func readPreV2CPrimaryPalette(matrix *core.Bitmap, symbol *core.DecodedSymbol, dataMap []byte, moduleCount, x, y *int) int {
	colorNumber := 1 << (symbol.Meta.NC + 1)
	physical := make([]byte, colorNumber*3*preV2CPaletteCopies)
	paletteOffset := 0
	if *moduleCount != 0 {
		paletteOffset = colorNumber * 3
	}
	switchOnOdd := matrix.Width > matrix.Height
	colorIndex, counter := 0, 0
	for colorIndex < min(colorNumber, 64) {
		if !preV2CMetadataPositionValid(matrix, *x, *y) {
			return MetadataFailed
		}
		off := ((*y)*matrix.Width + *x) * matrix.Channels
		copy(physical[paletteOffset+colorIndex*3:paletteOffset+colorIndex*3+3], matrix.Pix[off:off+3])
		dataMap[(*y)*matrix.Width+*x] = 1
		(*moduleCount)++
		spec.NextMetadataModuleInPrimary(matrix.Height, matrix.Width, *moduleCount, x, y)

		counter++
		switch counter % 4 {
		case 1:
			colorIndex++
			if switchOnOdd {
				paletteOffset = colorNumber*3 - paletteOffset
			}
		case 2:
			colorIndex--
			if !switchOnOdd {
				paletteOffset = colorNumber*3 - paletteOffset
			}
		case 3:
			colorIndex++
			if switchOnOdd {
				paletteOffset = colorNumber*3 - paletteOffset
			}
		case 0:
			colorIndex++
			if !switchOnOdd {
				paletteOffset = colorNumber*3 - paletteOffset
			}
		}
	}
	if colorNumber > 64 {
		interpolatePalette(physical, colorNumber)
	}

	symbol.Palette = expandPreV2CPalette(physical, colorNumber, matrix.Width, matrix.Height)
	return core.Success
}

func readPreV2CSecondaryPalette(matrix *core.Bitmap, symbol *core.DecodedSymbol, dataMap []byte) int {
	colorNumber := 1 << (symbol.Meta.NC + 1)
	physical := make([]byte, colorNumber*3*preV2CPaletteCopies)
	for colorIndex := 0; colorIndex < min(colorNumber, 64); colorIndex++ {
		pos := preV2CSecondaryPalettePositions[colorIndex/2]
		var x, y int
		if colorIndex%2 == 0 {
			x, y = pos.X, pos.Y
		} else if matrix.Width > matrix.Height {
			x, y = pos.Y, matrix.Height-1-pos.X
		} else {
			x, y = matrix.Width-1-pos.Y, pos.X
		}
		if !preV2CMetadataPositionValid(matrix, x, y) {
			return MetadataFailed
		}
		off := (y*matrix.Width + x) * matrix.Channels
		copy(physical[colorIndex*3:colorIndex*3+3], matrix.Pix[off:off+3])
		dataMap[y*matrix.Width+x] = 1

		x, y = matrix.Width-1-x, matrix.Height-1-y
		off = (y*matrix.Width + x) * matrix.Channels
		base := colorNumber*3 + colorIndex*3
		copy(physical[base:base+3], matrix.Pix[off:off+3])
		dataMap[y*matrix.Width+x] = 1
	}
	if colorNumber > 64 {
		interpolatePalette(physical, colorNumber)
	}
	symbol.Palette = expandPreV2CPalette(physical, colorNumber, matrix.Width, matrix.Height)
	return core.Success
}

func expandPreV2CPalette(physical []byte, colorNumber, width, height int) []byte {
	logical := make([]byte, colorNumber*3*preV2CLogicalPaletteCopies)
	physicalForCorner := [4]int{0, 0, 1, 1}
	if width > height {
		physicalForCorner = [4]int{0, 1, 1, 0}
	}
	copySize := colorNumber * 3
	for corner, physicalIndex := range physicalForCorner {
		copy(logical[corner*copySize:(corner+1)*copySize], physical[physicalIndex*copySize:(physicalIndex+1)*copySize])
	}
	return logical
}

func normalizePreV2CPalette(palette []byte, norm []float64, colorNumber int) {
	for i := 0; i < colorNumber*preV2CLogicalPaletteCopies; i++ {
		rgbMax := float64(max(palette[i*3], palette[i*3+1], palette[i*3+2]))
		if rgbMax == 0 {
			continue
		}
		norm[i*4] = float64(palette[i*3]) / rgbMax
		norm[i*4+1] = float64(palette[i*3+1]) / rgbMax
		norm[i*4+2] = float64(palette[i*3+2]) / rgbMax
		norm[i*4+3] = float64(int(palette[i*3])+int(palette[i*3+1])+int(palette[i*3+2])) / (3 * 255)
	}
}

func preV2CPaletteThresholds(palette []byte, colorNumber int) []float64 {
	ths := make([]float64, 3*preV2CLogicalPaletteCopies)
	for i := range preV2CLogicalPaletteCopies {
		p := palette[i*colorNumber*3 : (i+1)*colorNumber*3]
		var t [3]float64
		if colorNumber == 2 {
			for c := range 3 {
				t[c] = float64(int(p[c])+int(p[3+c])) / 2
			}
		} else {
			t = PaletteThreshold(p, colorNumber)
		}
		copy(ths[i*3:i*3+3], t[:])
	}
	return ths
}

func decodePreV2CModuleHD(matrix *core.Bitmap, palette []byte, colorNumber int, normPalette, palThs []float64, x, y int) byte {
	if colorNumber <= 8 {
		return DecodeModuleHD(matrix, palette, colorNumber, normPalette, palThs, x, y)
	}
	pIndex := nearestPalette(matrix, x, y)
	off := (y*matrix.Width + x) * matrix.Channels
	rgb := [3]byte{matrix.Pix[off], matrix.Pix[off+1], matrix.Pix[off+2]}
	best, bestIndex := math.Inf(1), byte(0)
	base := colorNumber * 3 * pIndex
	for i := range colorNumber {
		dr := float64(rgb[0]) - float64(palette[base+i*3])
		dg := float64(rgb[1]) - float64(palette[base+i*3+1])
		db := float64(rgb[2]) - float64(palette[base+i*3+2])
		if d := dr*dr + dg*dg + db*db; d < best {
			best, bestIndex = d, byte(i)
		}
	}
	return bestIndex
}

func decodePreV2CSymbol(matrix *core.Bitmap, symbol *core.DecodedSymbol, dataMap []byte, normPalette, palThs []float64, symbolType int) int {
	colorNumber := 1 << (symbol.Meta.NC + 1)
	if symbol.SideSize != image.Pt(matrix.Width, matrix.Height) ||
		len(symbol.Palette) < colorNumber*3*preV2CLogicalPaletteCopies ||
		symbol.Meta.ECL.X < 3 || symbol.Meta.ECL.X >= symbol.Meta.ECL.Y || symbol.Meta.ECL.Y > 11 {
		return core.Failure
	}
	fillPreV2CDataMap(dataMap, matrix.Width, matrix.Height, symbolType)
	rawModules := make([]byte, 0, matrix.Width*matrix.Height)
	for x := 0; x < matrix.Width; x++ {
		for y := 0; y < matrix.Height; y++ {
			if dataMap[y*matrix.Width+x] == 0 {
				rawModules = append(rawModules, decodePreV2CModuleHD(matrix, symbol.Palette, colorNumber, normPalette, palThs, x, y))
			}
		}
	}
	demaskSymbol(rawModules, dataMap, symbol.SideSize, symbol.Meta.MaskType, colorNumber)
	rawData := rawModuleData2RawData(rawModules, symbol.Meta.NC+1)
	wc, wr := symbol.Meta.ECL.X, symbol.Meta.ECL.Y
	pg := len(rawData) / wr * wr
	pn := pg * (wr - wc) / wr
	if pg == 0 {
		return core.Failure
	}
	rawData = rawData[:pg]
	ecc.DeinterleaveVariant(rawData, wire.PreV2C)
	decoded, ok := ecc.DecodeLDPCHardVariant(rawData, wc, wr, wire.PreV2C)
	if !ok || len(decoded) != pn {
		decoded = decodePreV2CSymbolSoft(matrix, symbol, dataMap, normPalette, rawData, wc, wr, pn)
		if decoded == nil {
			return core.Failure
		}
	}
	return decodeSymbolStream(decoded, symbol, symbolType)
}

func fillPreV2CDataMap(dataMap []byte, width, height, symbolType int) {
	const minimumAlignmentDistance = 16
	patternsX := (width-(spec.DistanceToBorder*2-1))/minimumAlignmentDistance - 1
	patternsY := (height-(spec.DistanceToBorder*2-1))/minimumAlignmentDistance - 1
	patternsX = max(patternsX, 0) + 2
	patternsY = max(patternsY, 0) + 2
	distanceX := float32(width-(spec.DistanceToBorder*2-1)) / float32(patternsX-1)
	distanceY := float32(height-(spec.DistanceToBorder*2-1)) / float32(patternsY-1)
	set := func(x, y int) {
		if x >= 0 && y >= 0 && x < width && y < height {
			dataMap[y*width+x] = 1
		}
	}
	for i := range patternsY {
		for j := range patternsX {
			x := spec.DistanceToBorder - 1 + int(float32(j)*distanceX)
			y := spec.DistanceToBorder - 1 + int(float32(i)*distanceY)
			set(x, y)
			set(x-1, y)
			set(x+1, y)
			set(x, y-1)
			set(x, y+1)

			switch {
			case i == 0 && (j == 0 || j == patternsX-1):
				set(x-1, y-1)
				set(x+1, y+1)
				if symbolType == 0 {
					set(x-2, y-2)
					set(x-1, y-2)
					set(x, y-2)
					set(x-2, y-1)
					set(x-2, y)
					set(x+2, y+2)
					set(x+1, y+2)
					set(x, y+2)
					set(x+2, y+1)
					set(x+2, y)
				}
			case i == patternsY-1 && (j == 0 || j == patternsX-1):
				set(x+1, y-1)
				set(x-1, y+1)
				if symbolType == 0 {
					set(x+2, y-2)
					set(x+1, y-2)
					set(x, y-2)
					set(x+2, y-1)
					set(x+2, y)
					set(x-2, y+2)
					set(x-1, y+2)
					set(x, y+2)
					set(x-2, y+1)
					set(x-2, y)
				}
			default:
				if i%2 == j%2 {
					set(x-1, y-1)
					set(x+1, y+1)
				} else {
					set(x+1, y-1)
					set(x-1, y+1)
				}
			}
		}
	}
}

func decodePreV2CSymbolSoft(matrix *core.Bitmap, symbol *core.DecodedSymbol, dataMap []byte, normPalette []float64, hard []byte, wc, wr, pn int) []byte {
	colorNumber := 1 << (symbol.Meta.NC + 1)
	rel := make([]float64, 0, len(hard))
	for x := 0; x < matrix.Width; x++ {
		for y := 0; y < matrix.Height; y++ {
			if dataMap[y*matrix.Width+x] == 0 {
				rel = appendPreV2CModuleReliabilities(rel, matrix, symbol.Palette, normPalette, colorNumber, x, y)
			}
		}
	}
	if len(rel) < len(hard) {
		return nil
	}
	rel = rel[:len(hard)]
	ecc.DeinterleaveFloatVariant(rel, wire.PreV2C)
	decoded, ok := ecc.DecodeLDPCSoftVariant(rel, hard, wc, wr, wire.PreV2C)
	if !ok || len(decoded) != pn {
		return nil
	}
	return decoded
}

func appendPreV2CModuleReliabilities(dst []float64, matrix *core.Bitmap, palette []byte, normPalette []float64, colorNumber, x, y int) []float64 {
	pIndex := nearestPalette(matrix, x, y)
	off := (y*matrix.Width + x) * matrix.Channels
	dist := make([]float64, colorNumber)
	if colorNumber > 8 {
		base := colorNumber * 3 * pIndex
		for i := range colorNumber {
			dr := float64(matrix.Pix[off]) - float64(palette[base+i*3])
			dg := float64(matrix.Pix[off+1]) - float64(palette[base+i*3+1])
			db := float64(matrix.Pix[off+2]) - float64(palette[base+i*3+2])
			dist[i] = dr*dr + dg*dg + db*db
		}
	} else {
		rgbMax := float64(max(matrix.Pix[off], matrix.Pix[off+1], matrix.Pix[off+2]))
		if rgbMax == 0 {
			rgbMax = 1
		}
		r := float64(matrix.Pix[off]) / rgbMax
		g := float64(matrix.Pix[off+1]) / rgbMax
		b := float64(matrix.Pix[off+2]) / rgbMax
		for i := range colorNumber {
			base := colorNumber*4*pIndex + i*4
			dr := normPalette[base] - r
			dg := normPalette[base+1] - g
			db := normPalette[base+2] - b
			dist[i] = dr*dr + dg*dg + db*db
		}
	}
	bitsPerModule := spec.Log2Int(colorNumber)
	for bit := range bitsPerModule {
		shift := uint(bitsPerModule - 1 - bit)
		min0, min1 := math.Inf(1), math.Inf(1)
		for colorIndex := range colorNumber {
			if colorIndex>>shift&1 == 0 {
				min0 = min(min0, dist[colorIndex])
			} else {
				min1 = min(min1, dist[colorIndex])
			}
		}
		dst = append(dst, math.Abs(min1-min0))
	}
	return dst
}
