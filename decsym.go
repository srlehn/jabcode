package jabcode

import (
	"image"

	"github.com/srlehn/jabcode/internal/ecc"
)

// Decoder status/constant values (decoder.h, jabcode.h).
const (
	decodeMetadataFailed   = -1
	defaultModuleColorMode = 2 // DEFAULT_MODULE_COLOR_MODE -> Nc, color count = 2^(Nc+1) = 8
)

// fillDataMap marks the finder/alignment pattern modules as reserved (non-data)
// in the data map (fillDataMap in decoder.c). type 0 = primary, 1 = secondary.
func fillDataMap(dataMap []byte, w, h, typ int) {
	vx := size2version(w) - 1
	vy := size2version(h) - 1
	nApX := apNum[vx]
	nApY := apNum[vy]
	set := func(x, y int) { dataMap[y*w+x] = 1 }

	for i := range nApY {
		for j := range nApX {
			xo := apPos[vx][j] - 1
			yo := apPos[vy][i] - 1
			set(xo, yo)
			set(xo-1, yo)
			set(xo+1, yo)
			set(xo, yo-1)
			set(xo, yo+1)

			switch {
			case i == 0 && (j == 0 || j == nApX-1): // FP0/FP1 corners
				set(xo-1, yo-1)
				set(xo+1, yo+1)
				if typ == 0 {
					set(xo-2, yo-2)
					set(xo-1, yo-2)
					set(xo, yo-2)
					set(xo-2, yo-1)
					set(xo-2, yo)
					set(xo+2, yo+2)
					set(xo+1, yo+2)
					set(xo, yo+2)
					set(xo+2, yo+1)
					set(xo+2, yo)
				}
			case i == nApY-1 && (j == 0 || j == nApX-1): // FP2/FP3 corners
				set(xo+1, yo-1)
				set(xo-1, yo+1)
				if typ == 0 {
					set(xo+2, yo-2)
					set(xo+1, yo-2)
					set(xo, yo-2)
					set(xo+2, yo-1)
					set(xo+2, yo)
					set(xo-2, yo+2)
					set(xo-1, yo+2)
					set(xo, yo+2)
					set(xo-2, yo+1)
					set(xo-2, yo)
				}
			default:
				if (i%2 == 0 && j%2 == 0) || (i%2 == 1 && j%2 == 1) {
					set(xo-1, yo-1)
					set(xo+1, yo+1)
				} else {
					set(xo+1, yo-1)
					set(xo-1, yo+1)
				}
			}
		}
	}
}

// loadDefaultPrimaryMetadata sets the metadata used when a primary symbol carries
// no explicit metadata (loadDefaultPrimaryMetadata in decoder.c).
func loadDefaultPrimaryMetadata(matrix *bitmap, symbol *decodedSymbol) {
	symbol.meta.defaultMode = true
	symbol.meta.Nc = defaultModuleColorMode
	symbol.meta.ecl = image.Pt(ecclevel2wcwr[defaultEccLevel][0], ecclevel2wcwr[defaultEccLevel][1])
	symbol.meta.maskType = defaultMaskingReference
	symbol.meta.dockedPosition = 0
	symbol.meta.sideVersion = image.Pt(size2version(matrix.width), size2version(matrix.height))
}

// decodeNcModuleColor maps a pair of metadata module colors to the encoded 3-bit
// value, or 8 if invalid (decodeNcModuleColor in decoder.c).
func decodeNcModuleColor(m1, m2 byte) byte {
	for i := range 8 {
		if int(m1) == ncColorEncode[i][0] && int(m2) == ncColorEncode[i][1] {
			return byte(i)
		}
	}
	return 8
}

// decodePrimaryMetadataPartI decodes Nc from the four Part I metadata modules
// (decodePrimaryMetadataPartI in decoder.c). Returns jabSuccess, jabFailure, or
// decodeMetadataFailed (the latter triggers the default-metadata fallback,
// which is what happens for default-mode symbols).
func decodePrimaryMetadataPartI(matrix *bitmap, symbol *decodedSymbol, dataMap []byte, moduleCount, x, y *int) int {
	var moduleColor [primaryMetadataPart1ModuleNumber]byte
	bpp := matrix.channels
	bytesPerRow := matrix.width * bpp
	for *moduleCount < primaryMetadataPart1ModuleNumber {
		off := (*y)*bytesPerRow + (*x)*bpp
		nc := decodeModuleNc(matrix.pix[off : off+3])
		if nc != 0 && nc != 3 && nc != 6 {
			return decodeMetadataFailed
		}
		moduleColor[*moduleCount] = nc
		dataMap[(*y)*matrix.width+(*x)] = 1
		(*moduleCount)++
		getNextMetadataModuleInPrimary(matrix.height, matrix.width, *moduleCount, x, y)
	}
	b0 := decodeNcModuleColor(moduleColor[0], moduleColor[1])
	b1 := decodeNcModuleColor(moduleColor[2], moduleColor[3])
	if b0 > 7 || b1 > 7 {
		return decodeMetadataFailed
	}
	part1 := make([]byte, primaryMetadataPart1Length)
	bc := 0
	for _, b := range [2]byte{b0, b1} {
		for i := range 3 {
			part1[bc] = (b >> (2 - i)) & 1
			bc++
		}
	}
	wc := 3
	if primaryMetadataPart1Length > 36 {
		wc = 4
	}
	dec := ecc.DecodeLDPCHard(part1, wc, 0)
	if len(dec) < 3 {
		return jabFailure
	}
	symbol.meta.Nc = int(dec[0])<<2 + int(dec[1])<<1 + int(dec[2])
	return jabSuccess
}

// decodePrimaryMetadataPartII decodes the version, ECC level and mask reference
// from Part II of the primary metadata (decodePrimaryMetadataPartII).
func decodePrimaryMetadataPartII(matrix *bitmap, symbol *decodedSymbol, dataMap []byte, normPalette, palThs []float64, moduleCount, x, y *int) int {
	part2 := make([]byte, primaryMetadataPart2Length)
	colorNumber := 1 << (symbol.meta.Nc + 1)
	bitsPerModule := log2int(colorNumber)

	bitCount := 0
	for bitCount < primaryMetadataPart2Length {
		bits := decodeModuleHD(matrix, symbol.palette, colorNumber, normPalette, palThs, *x, *y)
		for i := 0; i < bitsPerModule && bitCount < primaryMetadataPart2Length; i++ {
			part2[bitCount] = (bits >> (bitsPerModule - 1 - i)) & 1
			bitCount++
		}
		dataMap[(*y)*matrix.width+(*x)] = 1
		(*moduleCount)++
		getNextMetadataModuleInPrimary(matrix.height, matrix.width, *moduleCount, x, y)
	}

	wc := 3
	if primaryMetadataPart2Length > 36 {
		wc = 4
	}
	dec := ecc.DecodeLDPCHard(part2, wc, 0)
	if len(dec) == 0 {
		return decodeMetadataFailed
	}

	const vLen, eLen = 10, 6
	v := 0
	for i := range vLen / 2 {
		v += int(dec[i]) << (vLen/2 - 1 - i)
	}
	symbol.meta.sideVersion.X = v + 1
	v = 0
	for i := range vLen / 2 {
		v += int(dec[i+vLen/2]) << (vLen/2 - 1 - i)
	}
	symbol.meta.sideVersion.Y = v + 1

	e := 0
	for i := range eLen / 2 {
		e += int(dec[vLen+i]) << (eLen/2 - 1 - i)
	}
	symbol.meta.ecl.X = e + 3
	e = 0
	for i := range eLen / 2 {
		e += int(dec[vLen+eLen/2+i]) << (eLen/2 - 1 - i)
	}
	symbol.meta.ecl.Y = e + 4

	bi := vLen + eLen
	symbol.meta.maskType = int(dec[bi])<<2 + int(dec[bi+1])<<1 + int(dec[bi+2])
	symbol.meta.dockedPosition = 0

	symbol.sideSize = image.Pt(version2size(symbol.meta.sideVersion.X), version2size(symbol.meta.sideVersion.Y))
	if matrix.width != symbol.sideSize.X || matrix.height != symbol.sideSize.Y {
		return jabFailure
	}
	if symbol.meta.ecl.X >= symbol.meta.ecl.Y {
		return decodeMetadataFailed
	}
	return jabSuccess
}

// decodeSecondaryMetadata decodes a docked secondary symbol's metadata from the
// host data stream, returning the number of bits read or decodeMetadataFailed
// (decodeSlaveMetadata in decoder.c).
func decodeSecondaryMetadata(symbol *decodedSymbol, dockedPosition int, data []byte, offset int) int {
	symbol.secondaryMeta[dockedPosition].Nc = symbol.meta.Nc
	symbol.secondaryMeta[dockedPosition].maskType = symbol.meta.maskType
	symbol.secondaryMeta[dockedPosition].dockedPosition = 0

	index := offset
	if index < 0 {
		return decodeMetadataFailed
	}
	ss := data[index]
	index--
	if ss == 0 {
		symbol.secondaryMeta[dockedPosition].sideVersion = symbol.meta.sideVersion
	}
	if index < 0 {
		return decodeMetadataFailed
	}
	se := data[index]
	index--
	if se == 0 {
		symbol.secondaryMeta[dockedPosition].ecl = symbol.meta.ecl
	}

	if ss == 1 {
		if index-4 < 0 {
			return decodeMetadataFailed
		}
		v := 0
		for i := range 5 {
			v += int(data[index]) << (4 - i)
			index--
		}
		sideVersion := v + 1
		if dockedPosition == 2 || dockedPosition == 3 {
			symbol.secondaryMeta[dockedPosition].sideVersion.Y = symbol.meta.sideVersion.Y
			symbol.secondaryMeta[dockedPosition].sideVersion.X = sideVersion
		} else {
			symbol.secondaryMeta[dockedPosition].sideVersion.X = symbol.meta.sideVersion.X
			symbol.secondaryMeta[dockedPosition].sideVersion.Y = sideVersion
		}
	}
	if se == 1 {
		if index-5 < 0 {
			return decodeMetadataFailed
		}
		e := 0
		for i := range 3 {
			e += int(data[index]) << (2 - i)
			index--
		}
		symbol.secondaryMeta[dockedPosition].ecl.X = e + 3
		e = 0
		for i := range 3 {
			e += int(data[index]) << (2 - i)
			index--
		}
		symbol.secondaryMeta[dockedPosition].ecl.Y = e + 4
		if symbol.secondaryMeta[dockedPosition].ecl.X >= symbol.secondaryMeta[dockedPosition].ecl.Y {
			return decodeMetadataFailed
		}
	}
	return offset - index
}

// readRawModuleData reads the color index of every data module in column-major
// order (readRawModuleData in decoder.c).
func readRawModuleData(matrix *bitmap, symbol *decodedSymbol, dataMap []byte, normPalette, palThs []float64) []byte {
	colorNumber := 1 << (symbol.meta.Nc + 1)
	data := make([]byte, 0, matrix.width*matrix.height)
	for j := 0; j < matrix.width; j++ {
		for i := 0; i < matrix.height; i++ {
			if dataMap[i*matrix.width+j] == 0 {
				data = append(data, decodeModuleHD(matrix, symbol.palette, colorNumber, normPalette, palThs, j, i))
			}
		}
	}
	return data
}

// rawModuleData2RawData expands per-module color indices into a one-bit-per-byte
// stream (rawModuleData2RawData in decoder.c).
func rawModuleData2RawData(raw []byte, bitsPerModule int) []byte {
	out := make([]byte, len(raw)*bitsPerModule)
	for i, m := range raw {
		for j := range bitsPerModule {
			out[i*bitsPerModule+j] = (m >> (bitsPerModule - 1 - j)) & 1
		}
	}
	return out
}

// decodeSymbol reads, demasks, deinterleaves and error-corrects a symbol's data
// modules, storing the net payload in symbol.data (decodeSymbol in decoder.c).
func decodeSymbol(matrix *bitmap, symbol *decodedSymbol, dataMap []byte, normPalette, palThs []float64, typ int) int {
	fillDataMap(dataMap, matrix.width, matrix.height, typ)

	rawModuleData := readRawModuleData(matrix, symbol, dataMap, normPalette, palThs)
	demaskSymbol(rawModuleData, dataMap, symbol.sideSize, symbol.meta.maskType, 1<<(symbol.meta.Nc+1))
	rawData := rawModuleData2RawData(rawModuleData, symbol.meta.Nc+1)

	wc := symbol.meta.ecl.X
	wr := symbol.meta.ecl.Y
	Pg := (len(rawData) / wr) * wr
	Pn := Pg * (wr - wc) / wr

	rawData = rawData[:Pg] // drop padding bits
	ecc.Deinterleave(rawData)

	dec := ecc.DecodeLDPCHard(rawData, wc, wr)
	if len(dec) != Pn {
		return jabFailure
	}

	// Locate the start flag (last set bit) of the in-stream metadata.
	metaOffset := Pn - 1
	for dec[metaOffset] == 0 {
		metaOffset--
	}
	metaOffset-- // skip the flag bit

	symbol.meta.dockedPosition = 0
	for i := range 4 {
		if typ == 1 && i == symbol.hostPosition {
			continue
		}
		symbol.meta.dockedPosition += int(dec[metaOffset]) << (3 - i)
		metaOffset--
	}
	for i := range 4 {
		if symbol.meta.dockedPosition&(0x08>>i) != 0 {
			readBitLength := decodeSecondaryMetadata(symbol, i, dec, metaOffset)
			if readBitLength == decodeMetadataFailed {
				return decodeMetadataFailed
			}
			metaOffset -= readBitLength
		}
	}

	netDataLength := metaOffset + 1
	symbol.data = make([]byte, netDataLength)
	copy(symbol.data, dec[:netDataLength])
	return jabSuccess
}

// decodePrimary decodes a primary symbol from its sampled matrix (decodePrimary in
// decoder.c).
func decodePrimary(matrix *bitmap, symbol *decodedSymbol) int {
	if matrix == nil {
		return fatalError
	}
	symbol.sideSize = image.Pt(matrix.width, matrix.height)
	dataMap := make([]byte, matrix.width*matrix.height)

	x, y := primaryMetadataX, primaryMetadataY
	moduleCount := 0

	partIRet := decodePrimaryMetadataPartI(matrix, symbol, dataMap, &moduleCount, &x, &y)
	if partIRet == jabFailure {
		return jabFailure
	}
	if partIRet == decodeMetadataFailed {
		x, y = primaryMetadataX, primaryMetadataY
		moduleCount = 0
		clear(dataMap)
		loadDefaultPrimaryMetadata(matrix, symbol)
	}

	if readColorPaletteInPrimary(matrix, symbol, dataMap, &moduleCount, &x, &y) < 0 {
		return jabFailure
	}

	colorNumber := 1 << (symbol.meta.Nc + 1)
	normPalette := make([]float64, colorNumber*4*colorPaletteNumber)
	normalizeColorPalette(symbol, normPalette, colorNumber)
	palThs := make([]float64, 3*colorPaletteNumber)
	for i := range colorPaletteNumber {
		t := getPaletteThreshold(symbol.palette[colorNumber*3*i:], colorNumber)
		palThs[i*3+0], palThs[i*3+1], palThs[i*3+2] = t[0], t[1], t[2]
	}

	if partIRet == jabSuccess {
		if decodePrimaryMetadataPartII(matrix, symbol, dataMap, normPalette, palThs, &moduleCount, &x, &y) <= 0 {
			return jabFailure
		}
	}

	return decodeSymbol(matrix, symbol, dataMap, normPalette, palThs, 0)
}
