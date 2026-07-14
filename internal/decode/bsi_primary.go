//go:build jabcode_bsi

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
	bsiPrimaryMetadataPart1Length = 6
	bsiPrimaryMetadataPart2Length = 14
	bsiPhysicalPaletteCopies      = 2
	bsiLogicalPaletteCopies       = 4
)

var bsiPrimaryPalettePositions = [8]image.Point{
	image.Pt(4, 1), image.Pt(4, 2), image.Pt(5, 1), image.Pt(5, 2),
	image.Pt(2, 4), image.Pt(2, 5), image.Pt(1, 4), image.Pt(1, 5),
}

// DecodeBSIPrimary decodes the primary-symbol layout specified by
// BSI TR-03137-2. The caller must already have identified the BSI finder
// family; this function does not probe other wire profiles.
func DecodeBSIPrimary(matrix *core.Bitmap, symbol *core.DecodedSymbol) int {
	if matrix == nil || symbol == nil || matrix.Channels < 3 ||
		!spec.ValidSideSize(matrix.Width) || !spec.ValidSideSize(matrix.Height) {
		return core.Failure
	}

	symbol.WireProfile = wire.BSI
	symbol.SideSize = image.Pt(matrix.Width, matrix.Height)
	dataMap := make([]byte, matrix.Width*matrix.Height)
	if ret := decodeBSIPrimaryMetadata(matrix, symbol, dataMap); ret != core.Success {
		return ret
	}
	return decodeBSISymbol(matrix, symbol, dataMap, 0)
}

func decodeBSIPrimaryMetadata(matrix *core.Bitmap, symbol *core.DecodedSymbol, dataMap []byte) int {
	x, y, moduleCount := spec.PrimaryMetadataX, spec.PrimaryMetadataY, 0
	part1 := make([]byte, bsiPrimaryMetadataPart1Length)
	for i := range part1 {
		if !bsiPositionValid(matrix, x, y) {
			return core.Failure
		}
		off := (y*matrix.Width + x) * matrix.Channels
		rgb := matrix.Pix[off : off+3]
		brightChannels := 0
		for _, channel := range rgb {
			if channel > 100 {
				brightChannels++
			}
		}
		if brightChannels > 1 {
			part1[i] = 1
		}
		dataMap[y*matrix.Width+x] = 1
		moduleCount++
		spec.NextMetadataModuleInPrimary(matrix.Height, matrix.Width, moduleCount, &x, &y)
	}
	part1Decoded, ok := ecc.DecodeLDPCHardProfile(part1, 2, -1, wire.BSI)
	if !ok || len(part1Decoded) < 3 {
		return MetadataFailed
	}
	symbol.Meta.NC = bsiBitsValue(part1Decoded[:3])
	if symbol.Meta.NC < 0 || symbol.Meta.NC > 7 {
		return MetadataFailed
	}

	colorNumber := 1 << (symbol.Meta.NC + 1)
	physicalPalette := make([]byte, colorNumber*3*bsiPhysicalPaletteCopies)
	metadataColorNumber := min(colorNumber, 8)
	if !readBSIFirstPrimaryPalette(matrix, physicalPalette, colorNumber, metadataColorNumber, dataMap) {
		return core.Failure
	}
	metadataPalette := bsiFirstPaletteColors(physicalPalette, colorNumber, metadataColorNumber)
	metadataPalette = expandBSIPalette(metadataPalette, metadataColorNumber, matrix.Width, matrix.Height)
	normPalette := make([]float64, metadataColorNumber*4*bsiLogicalPaletteCopies)
	bsiNormalizePalette(metadataPalette, normPalette, metadataColorNumber)
	paletteThresholds := bsiPaletteThresholds(metadataPalette, metadataColorNumber)

	reader := bsiMetadataBitReader{
		matrix: matrix, palette: metadataPalette, colorNumber: metadataColorNumber,
		normPalette: normPalette, paletteThresholds: paletteThresholds,
		dataMap: dataMap, moduleCount: &moduleCount, x: &x, y: &y,
	}
	part2, ok := reader.read(bsiPrimaryMetadataPart2Length)
	if !ok {
		return core.Failure
	}
	part2Decoded, ok := ecc.DecodeLDPCHardProfile(part2, 2, -1, wire.BSI)
	if !ok || len(part2Decoded) < 7 {
		return MetadataFailed
	}
	ss := int(part2Decoded[0])
	vf := bsiBitsValue(part2Decoded[1:3])
	symbol.Meta.MaskType = bsiBitsValue(part2Decoded[3:6])
	sf := part2Decoded[6]

	versionLength := vf + 1
	if ss == 0 {
		if vf == 0 {
			versionLength = 2
		}
	} else {
		versionLength = vf*2 + 4
	}
	eclLength := vf*2 + 10
	dockedLength := 0
	if sf != 0 {
		dockedLength = 4
	}
	part3RawLength := versionLength + eclLength + dockedLength
	part3, ok := reader.read(part3RawLength * 2)
	if !ok {
		return core.Failure
	}
	part3Decoded, ok := ecc.DecodeLDPCHardProfile(part3, 2, -1, wire.BSI)
	if !ok || len(part3Decoded) < part3RawLength {
		return MetadataFailed
	}

	bitIndex := 0
	if ss == 0 {
		v := bsiBitsValue(part3Decoded[:versionLength])
		if vf == 0 {
			v++
		} else {
			v += 1<<(vf+1) + 1
		}
		symbol.Meta.SideVersion = image.Pt(v, v)
	} else {
		half := versionLength / 2
		symbol.Meta.SideVersion = image.Pt(
			bsiBitsValue(part3Decoded[:half])+1,
			bsiBitsValue(part3Decoded[half:versionLength])+1,
		)
	}
	bitIndex += versionLength
	halfECL := eclLength / 2
	symbol.Meta.ECL = image.Pt(
		bsiBitsValue(part3Decoded[bitIndex:bitIndex+halfECL])+3,
		bsiBitsValue(part3Decoded[bitIndex+halfECL:bitIndex+eclLength])+4,
	)
	bitIndex += eclLength
	symbol.Meta.DockedPosition = 0
	if dockedLength != 0 {
		symbol.Meta.DockedPosition = bsiBitsValue(part3Decoded[bitIndex : bitIndex+dockedLength])
	}
	symbol.Meta.DefaultMode = false

	if symbol.Meta.SideVersion.X < 1 || symbol.Meta.SideVersion.X > 32 ||
		symbol.Meta.SideVersion.Y < 1 || symbol.Meta.SideVersion.Y > 32 ||
		symbol.Meta.ECL.X < 3 || symbol.Meta.ECL.X >= symbol.Meta.ECL.Y || symbol.Meta.ECL.Y > 35 {
		return MetadataFailed
	}
	symbol.SideSize = image.Pt(
		spec.VersionToSize(symbol.Meta.SideVersion.X),
		spec.VersionToSize(symbol.Meta.SideVersion.Y),
	)
	if symbol.SideSize.X != matrix.Width || symbol.SideSize.Y != matrix.Height {
		return core.Failure
	}

	if !readBSIRemainingPrimaryPalette(matrix, physicalPalette, colorNumber, dataMap, &moduleCount, &x, &y) {
		return core.Failure
	}
	deinterleaveBSIPalette(physicalPalette, colorNumber)
	if colorNumber > 64 {
		interpolatePalette(physicalPalette, colorNumber)
	}
	symbol.Palette = expandBSIPalette(physicalPalette, colorNumber, matrix.Width, matrix.Height)
	return core.Success
}

type bsiMetadataBitReader struct {
	matrix            *core.Bitmap
	palette           []byte
	colorNumber       int
	normPalette       []float64
	paletteThresholds []float64
	dataMap           []byte
	moduleCount       *int
	x, y              *int
	pending           []byte
}

func (r *bsiMetadataBitReader) read(length int) ([]byte, bool) {
	bitsPerModule := spec.Log2Int(r.colorNumber)
	for len(r.pending) < length {
		if !bsiPositionValid(r.matrix, *r.x, *r.y) {
			return nil, false
		}
		value := decodeBSIModuleHD(r.matrix, r.palette, r.colorNumber, r.normPalette, r.paletteThresholds, *r.x, *r.y)
		for i := range bitsPerModule {
			r.pending = append(r.pending, (value>>uint(bitsPerModule-1-i))&1)
		}
		r.dataMap[*r.y*r.matrix.Width+*r.x] = 1
		*r.moduleCount++
		spec.NextMetadataModuleInPrimary(r.matrix.Height, r.matrix.Width, *r.moduleCount, r.x, r.y)
	}
	out := append([]byte(nil), r.pending[:length]...)
	r.pending = r.pending[length:]
	return out, true
}

func readBSIFirstPrimaryPalette(matrix *core.Bitmap, physical []byte, colorNumber, available int, dataMap []byte) bool {
	for colorIndex := 0; colorIndex < available; colorIndex++ {
		pos := bsiPrimaryPalettePositions[colorIndex]
		if !readBSIPaletteColor(matrix, physical, colorNumber, 0, colorIndex, pos.X, pos.Y, dataMap) {
			return false
		}
		x := matrix.Width - 1 - pos.X
		y := matrix.Height - 7 + pos.Y
		if !readBSIPaletteColor(matrix, physical, colorNumber, 1, colorIndex, x, y, dataMap) {
			return false
		}
	}
	return true
}

func readBSIRemainingPrimaryPalette(matrix *core.Bitmap, physical []byte, colorNumber int, dataMap []byte, moduleCount, x, y *int) bool {
	available := min(colorNumber, 64)
	for colorIndex := 8; colorIndex < min(available, 16); colorIndex++ {
		pos := bsiPrimaryPalettePositions[colorIndex-8]
		var x0, y0 int
		if matrix.Width > matrix.Height {
			x0 = 6 - pos.X
			y0 = matrix.Height - 7 + pos.Y
		} else {
			x0 = matrix.Width - 7 + pos.X
			y0 = pos.Y
		}
		if !readBSIPaletteColor(matrix, physical, colorNumber, 0, colorIndex, x0, y0, dataMap) {
			return false
		}
		x1 := matrix.Width - 1 - x0
		y1 := matrix.Height - 7 + y0
		if matrix.Width > matrix.Height {
			y1 = y0 - (matrix.Height - 7)
		}
		if !readBSIPaletteColor(matrix, physical, colorNumber, 1, colorIndex, x1, y1, dataMap) {
			return false
		}
	}
	if available <= 16 {
		return true
	}

	paletteCopy := 0
	switchOnFirstAndThird := false
	if matrix.Width > matrix.Height {
		switch *moduleCount % 4 {
		case 0:
			paletteCopy, switchOnFirstAndThird = 0, true
		case 3:
			paletteCopy, switchOnFirstAndThird = 0, false
		case 1:
			paletteCopy, switchOnFirstAndThird = 1, false
		case 2:
			paletteCopy, switchOnFirstAndThird = 1, true
		}
	} else {
		switch *moduleCount % 4 {
		case 0:
			paletteCopy, switchOnFirstAndThird = 0, false
		case 1:
			paletteCopy, switchOnFirstAndThird = 0, true
		case 2:
			paletteCopy, switchOnFirstAndThird = 1, false
		case 3:
			paletteCopy, switchOnFirstAndThird = 1, true
		}
	}

	colorIndex, counter := 16, 0
	for colorIndex < available {
		if !readBSIPaletteColor(matrix, physical, colorNumber, paletteCopy, colorIndex, *x, *y, dataMap) {
			return false
		}
		*moduleCount++
		spec.NextMetadataModuleInPrimary(matrix.Height, matrix.Width, *moduleCount, x, y)
		counter++
		switch counter % 4 {
		case 1:
			colorIndex++
			if switchOnFirstAndThird {
				paletteCopy = 1 - paletteCopy
			}
		case 2:
			colorIndex--
			if !switchOnFirstAndThird {
				paletteCopy = 1 - paletteCopy
			}
		case 3:
			colorIndex++
			if switchOnFirstAndThird {
				paletteCopy = 1 - paletteCopy
			}
		case 0:
			colorIndex++
			if !switchOnFirstAndThird {
				paletteCopy = 1 - paletteCopy
			}
		}
	}
	return true
}

func readBSIPaletteColor(matrix *core.Bitmap, palette []byte, colorNumber, copyIndex, colorIndex, x, y int, dataMap []byte) bool {
	if !bsiPositionValid(matrix, x, y) || copyIndex < 0 || copyIndex >= bsiPhysicalPaletteCopies ||
		colorIndex < 0 || colorIndex >= colorNumber {
		return false
	}
	matrixOffset := (y*matrix.Width + x) * matrix.Channels
	paletteOffset := (copyIndex*colorNumber + colorIndex) * 3
	copy(palette[paletteOffset:paletteOffset+3], matrix.Pix[matrixOffset:matrixOffset+3])
	dataMap[y*matrix.Width+x] = 1
	return true
}

func bsiFirstPaletteColors(physical []byte, colorNumber, available int) []byte {
	p := make([]byte, available*3*bsiPhysicalPaletteCopies)
	for copyIndex := range bsiPhysicalPaletteCopies {
		for colorIndex := range available {
			src := (copyIndex*colorNumber + colorIndex) * 3
			dst := (copyIndex*available + colorIndex) * 3
			copy(p[dst:dst+3], physical[src:src+3])
		}
	}
	return p
}

func deinterleaveBSIPalette(physical []byte, colorNumber int) {
	available := min(colorNumber, 64)
	if available < 16 {
		return
	}
	// The BSI reader deinterleaves the 64 carried representatives through the
	// 64-color permutation even when the full palette has 128 or 256 entries.
	// That compacts the representatives into the layout interpolatePalette
	// expands, matching the published reference algorithm.
	placement := bsiPalettePlacementIndex(available, available)
	copySize := colorNumber * 3
	for copyIndex := range bsiPhysicalPaletteCopies {
		base := copyIndex * copySize
		tmp := append([]byte(nil), physical[base:base+copySize]...)
		for observed, canonical := range placement {
			copy(physical[base+canonical*3:base+canonical*3+3], tmp[observed*3:observed*3+3])
		}
	}
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
		for i, value := range []int{6, 7} {
			index[2+i] = value
		}
		for i, value := range []int{24, 25} {
			index[4+i] = value
		}
		for i, value := range []int{30, 31} {
			index[6+i] = value
		}
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
		values := []int{3, 12, 15, 48, 51, 60, 63}
		copy(index[1:8], values)
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
		}
		for i := range 16 {
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
		for i := range 2 {
			index[18+i] = 29 + i
			index[52+i] = 225 + i
			index[62+i] = 253 + i
		}
	}
	return index
}

func expandBSIPalette(physical []byte, colorNumber, width, height int) []byte {
	logical := make([]byte, colorNumber*3*bsiLogicalPaletteCopies)
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

func bsiNormalizePalette(palette []byte, norm []float64, colorNumber int) {
	for i := 0; i < colorNumber*bsiLogicalPaletteCopies; i++ {
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

func bsiPaletteThresholds(palette []byte, colorNumber int) []float64 {
	thresholds := make([]float64, 3*bsiLogicalPaletteCopies)
	for i := range bsiLogicalPaletteCopies {
		p := palette[i*colorNumber*3 : (i+1)*colorNumber*3]
		var t [3]float64
		if colorNumber == 2 {
			for channel := range 3 {
				t[channel] = float64(int(p[channel])+int(p[3+channel])) / 2
			}
		} else {
			t = PaletteThreshold(p, colorNumber)
		}
		copy(thresholds[i*3:i*3+3], t[:])
	}
	return thresholds
}

func decodeBSIModuleHD(matrix *core.Bitmap, palette []byte, colorNumber int, normPalette, paletteThresholds []float64, x, y int) byte {
	if colorNumber <= 8 {
		return DecodeModuleHD(matrix, palette, colorNumber, normPalette, paletteThresholds, x, y)
	}
	pIndex := nearestPalette(matrix, x, y)
	off := (y*matrix.Width + x) * matrix.Channels
	best, bestIndex := math.Inf(1), byte(0)
	base := colorNumber * 3 * pIndex
	for i := range colorNumber {
		dr := float64(matrix.Pix[off]) - float64(palette[base+i*3])
		dg := float64(matrix.Pix[off+1]) - float64(palette[base+i*3+1])
		db := float64(matrix.Pix[off+2]) - float64(palette[base+i*3+2])
		if distance := dr*dr + dg*dg + db*db; distance < best {
			best, bestIndex = distance, byte(i)
		}
	}
	return bestIndex
}

func decodeBSISymbol(matrix *core.Bitmap, symbol *core.DecodedSymbol, dataMap []byte, symbolType int) int {
	colorNumber := 1 << (symbol.Meta.NC + 1)
	if symbol.SideSize != image.Pt(matrix.Width, matrix.Height) ||
		len(symbol.Palette) < colorNumber*3*bsiLogicalPaletteCopies ||
		symbol.Meta.ECL.X < 3 || symbol.Meta.ECL.X >= symbol.Meta.ECL.Y {
		return core.Failure
	}

	fillBSIDataMap(dataMap, matrix.Width, matrix.Height, symbolType)
	normPalette := make([]float64, colorNumber*4*bsiLogicalPaletteCopies)
	bsiNormalizePalette(symbol.Palette, normPalette, colorNumber)
	paletteThresholds := bsiPaletteThresholds(symbol.Palette, colorNumber)
	rawModules := make([]byte, 0, matrix.Width*matrix.Height)
	for x := range matrix.Width {
		for y := range matrix.Height {
			if dataMap[y*matrix.Width+x] == 0 {
				rawModules = append(rawModules, decodeBSIModuleHD(matrix, symbol.Palette, colorNumber, normPalette, paletteThresholds, x, y))
			}
		}
	}
	demaskSymbol(rawModules, dataMap, symbol.SideSize, symbol.Meta.MaskType, colorNumber)
	rawData := rawModuleData2RawData(rawModules, symbol.Meta.NC+1)
	wc, wr := symbol.Meta.ECL.X, symbol.Meta.ECL.Y
	pg := len(rawData) / wr * wr
	if pg == 0 {
		return core.Failure
	}
	rawData = rawData[:pg]
	ecc.DeinterleaveProfile(rawData, wire.BSI)
	decoded, ok := ecc.DecodeLDPCHardProfile(rawData, wc, wr, wire.BSI)
	if !ok {
		decoded = decodeBSISymbolSoft(matrix, symbol, dataMap, normPalette, rawData, wc, wr)
		if decoded == nil {
			return core.Failure
		}
	}
	symbol.Data = append(symbol.Data[:0], decoded...)
	return core.Success
}

func decodeBSISymbolSoft(matrix *core.Bitmap, symbol *core.DecodedSymbol, dataMap []byte, normPalette []float64, hard []byte, wc, wr int) []byte {
	colorNumber := 1 << (symbol.Meta.NC + 1)
	reliabilities := make([]float64, 0, len(hard))
	for x := range matrix.Width {
		for y := range matrix.Height {
			if dataMap[y*matrix.Width+x] == 0 {
				reliabilities = appendBSIModuleReliabilities(reliabilities, matrix, symbol.Palette, normPalette, colorNumber, x, y)
			}
		}
	}
	if len(reliabilities) < len(hard) {
		return nil
	}
	reliabilities = reliabilities[:len(hard)]
	ecc.DeinterleaveFloatProfile(reliabilities, wire.BSI)
	decoded, ok := ecc.DecodeLDPCSoftProfile(reliabilities, hard, wc, wr, wire.BSI)
	if !ok {
		return nil
	}
	return decoded
}

func appendBSIModuleReliabilities(dst []float64, matrix *core.Bitmap, palette []byte, normPalette []float64, colorNumber, x, y int) []float64 {
	pIndex := nearestPalette(matrix, x, y)
	off := (y*matrix.Width + x) * matrix.Channels
	distances := make([]float64, colorNumber)
	if colorNumber > 8 {
		base := colorNumber * 3 * pIndex
		for i := range colorNumber {
			dr := float64(matrix.Pix[off]) - float64(palette[base+i*3])
			dg := float64(matrix.Pix[off+1]) - float64(palette[base+i*3+1])
			db := float64(matrix.Pix[off+2]) - float64(palette[base+i*3+2])
			distances[i] = dr*dr + dg*dg + db*db
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
			distances[i] = dr*dr + dg*dg + db*db
		}
	}
	bitsPerModule := spec.Log2Int(colorNumber)
	for bit := range bitsPerModule {
		shift := uint(bitsPerModule - 1 - bit)
		minimum0, minimum1 := math.Inf(1), math.Inf(1)
		for colorIndex := range colorNumber {
			if colorIndex>>shift&1 == 0 {
				minimum0 = min(minimum0, distances[colorIndex])
			} else {
				minimum1 = min(minimum1, distances[colorIndex])
			}
		}
		dst = append(dst, math.Abs(minimum1-minimum0))
	}
	return dst
}

func fillBSIDataMap(dataMap []byte, width, height, symbolType int) {
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

func bsiPositionValid(matrix *core.Bitmap, x, y int) bool {
	return x >= 0 && y >= 0 && x < matrix.Width && y < matrix.Height
}

func bsiBitsValue(bits []byte) int {
	value := 0
	for _, bit := range bits {
		value = value<<1 | int(bit&1)
	}
	return value
}
