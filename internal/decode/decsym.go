package decode

import (
	"image"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/ecc"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/tables"
)

// Decoder status/constant values.
const (
	MetadataFailed         = -1
	defaultModuleColorMode = 2 // DEFAULT_MODULE_COLOR_MODE -> Nc, color count = 2^(Nc+1) = 8
)

// fillDataMap marks the finder/alignment pattern modules as reserved (non-data)
// in the data map. type 0 = primary, 1 = secondary.
func fillDataMap(dataMap []byte, w, h, typ int) {
	// Ports fillDataMap in decoder.c.
	vx := spec.SizeToVersion(w) - 1
	vy := spec.SizeToVersion(h) - 1
	nApX := tables.APNum[vx]
	nApY := tables.APNum[vy]
	set := func(x, y int) { dataMap[y*w+x] = 1 }

	for i := range nApY {
		for j := range nApX {
			xo := tables.APPos[vx][j] - 1
			yo := tables.APPos[vy][i] - 1
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

// LoadDefaultPrimaryMetadata sets the metadata used when a primary symbol carries
// no explicit metadata.
func LoadDefaultPrimaryMetadata(matrix *core.Bitmap, symbol *core.DecodedSymbol) {
	// Ports loadDefaultPrimaryMetadata in decoder.c.
	symbol.Meta.DefaultMode = true
	symbol.Meta.NC = defaultModuleColorMode
	symbol.Meta.ECL = image.Pt(spec.ECCWeights[spec.DefaultECCLevel][0], spec.ECCWeights[spec.DefaultECCLevel][1])
	symbol.Meta.MaskType = spec.DefaultMaskingReference
	symbol.Meta.DockedPosition = 0
	symbol.Meta.SideVersion = image.Pt(spec.SizeToVersion(matrix.Width), spec.SizeToVersion(matrix.Height))
}

// ncPairValues validates the four Part I module colours (each must be one of the
// three Nc-encoding colours) and maps them to the two encoded 3-bit values.
func ncPairValues(moduleColor [spec.PrimaryMetadataPart1ModuleNumber]byte) (b0, b1 byte, ok bool) {
	for _, c := range moduleColor {
		if c != 0 && c != 3 && c != 6 {
			return 0, 0, false
		}
	}
	b0 = decodeNcModuleColor(moduleColor[0], moduleColor[1])
	b1 = decodeNcModuleColor(moduleColor[2], moduleColor[3])
	return b0, b1, b0 <= 7 && b1 <= 7
}

// decodeNcModuleColor maps a pair of metadata module colors to the encoded 3-bit
// value, or 8 if invalid.
func decodeNcModuleColor(m1, m2 byte) byte {
	// Ports decodeNcModuleColor in decoder.c.
	for i := range 8 {
		if int(m1) == tables.NcColorEncode[i][0] && int(m2) == tables.NcColorEncode[i][1] {
			return byte(i)
		}
	}
	return 8
}

// DecodePrimaryMetadataPartI decodes Nc from the four Part I metadata modules.
// Returns Success, Failure, or MetadataFailed (the latter triggers
// the default-metadata fallback, which is what happens for default-mode symbols).
//
// The plain per-module classification decides from absolute channel values, which
// a display cast can defeat (a screen's black is bright enough in blue to fail the
// black test). When it produces an invalid Part I, the same four modules are
// re-classified against references derived from the symbol's own finder cores
// (partIColorRefs) before falling back to default metadata: a genuinely default
// symbol has palette colours in these positions that still classify outside the
// Part I set, so the fallback semantics are preserved.
func DecodePrimaryMetadataPartI(matrix *core.Bitmap, symbol *core.DecodedSymbol, dataMap []byte, moduleCount, x, y *int) int {
	// Ports decodePrimaryMetadataPartI in decoder.c, plus the reference-anchored retry.
	var moduleColor [spec.PrimaryMetadataPart1ModuleNumber]byte
	var moduleRGB [spec.PrimaryMetadataPart1ModuleNumber][3]byte
	bpp := matrix.Channels
	bytesPerRow := matrix.Width * bpp
	for *moduleCount < spec.PrimaryMetadataPart1ModuleNumber {
		off := (*y)*bytesPerRow + (*x)*bpp
		copy(moduleRGB[*moduleCount][:], matrix.Pix[off:off+3])
		moduleColor[*moduleCount] = DecodeModuleNC(matrix.Pix[off : off+3])
		dataMap[(*y)*matrix.Width+(*x)] = 1
		(*moduleCount)++
		spec.NextMetadataModuleInPrimary(matrix.Height, matrix.Width, *moduleCount, x, y)
	}
	b0, b1, ok := ncPairValues(moduleColor)
	if !ok {
		if refs, refsOK := partIColorRefs(matrix); refsOK {
			for i := range moduleColor {
				moduleColor[i] = decodeModuleNCRef(moduleRGB[i][:], &refs)
			}
			b0, b1, ok = ncPairValues(moduleColor)
		}
		if !ok {
			return MetadataFailed
		}
	}
	part1 := make([]byte, spec.PrimaryMetadataPart1Length)
	bc := 0
	for _, b := range [2]byte{b0, b1} {
		for i := range 3 {
			part1[bc] = (b >> (2 - i)) & 1
			bc++
		}
	}
	wc := 3
	if spec.PrimaryMetadataPart1Length > 36 {
		wc = 4
	}
	// The syndrome result is not enforced here: metadata has fallback ladders
	// of its own, and gating them is a separate measured change.
	dec, _ := ecc.DecodeLDPCHard(part1, wc, 0)
	if len(dec) < 3 {
		return core.Failure
	}
	symbol.Meta.NC = int(dec[0])<<2 + int(dec[1])<<1 + int(dec[2])
	return core.Success
}

// DecodePrimaryMetadataPartII decodes the version, ECC level and mask reference
// from Part II of the primary metadata.
func DecodePrimaryMetadataPartII(matrix *core.Bitmap, symbol *core.DecodedSymbol, dataMap []byte, normPalette, palThs []float64, moduleCount, x, y *int) int {
	// Ports decodePrimaryMetadataPartII in decoder.c.
	part2 := make([]byte, spec.PrimaryMetadataPart2Length)
	colorNumber := 1 << (symbol.Meta.NC + 1)
	bitsPerModule := spec.Log2Int(colorNumber)

	bitCount := 0
	for bitCount < spec.PrimaryMetadataPart2Length {
		bits := DecodeModuleHD(matrix, symbol.Palette, colorNumber, normPalette, palThs, *x, *y)
		for i := 0; i < bitsPerModule && bitCount < spec.PrimaryMetadataPart2Length; i++ {
			part2[bitCount] = (bits >> (bitsPerModule - 1 - i)) & 1
			bitCount++
		}
		dataMap[(*y)*matrix.Width+(*x)] = 1
		(*moduleCount)++
		spec.NextMetadataModuleInPrimary(matrix.Height, matrix.Width, *moduleCount, x, y)
	}

	wc := 3
	if spec.PrimaryMetadataPart2Length > 36 {
		wc = 4
	}
	// The syndrome result is not enforced here either (see Part I).
	dec, _ := ecc.DecodeLDPCHard(part2, wc, 0)
	if len(dec) == 0 {
		return MetadataFailed
	}

	const vLen, eLen = 10, 6
	v := 0
	for i := range vLen / 2 {
		v += int(dec[i]) << (vLen/2 - 1 - i)
	}
	symbol.Meta.SideVersion.X = v + 1
	v = 0
	for i := range vLen / 2 {
		v += int(dec[i+vLen/2]) << (vLen/2 - 1 - i)
	}
	symbol.Meta.SideVersion.Y = v + 1

	e := 0
	for i := range eLen / 2 {
		e += int(dec[vLen+i]) << (eLen/2 - 1 - i)
	}
	symbol.Meta.ECL.X = e + 3
	e = 0
	for i := range eLen / 2 {
		e += int(dec[vLen+eLen/2+i]) << (eLen/2 - 1 - i)
	}
	symbol.Meta.ECL.Y = e + 4

	bi := vLen + eLen
	symbol.Meta.MaskType = int(dec[bi])<<2 + int(dec[bi+1])<<1 + int(dec[bi+2])
	symbol.Meta.DockedPosition = 0

	symbol.SideSize = image.Pt(spec.VersionToSize(symbol.Meta.SideVersion.X), spec.VersionToSize(symbol.Meta.SideVersion.Y))
	if matrix.Width != symbol.SideSize.X || matrix.Height != symbol.SideSize.Y {
		return core.Failure
	}
	if symbol.Meta.ECL.X >= symbol.Meta.ECL.Y {
		return MetadataFailed
	}
	return core.Success
}

// decodeSecondaryMetadata decodes a docked secondary symbol's metadata from the
// host data stream, returning the number of bits read or MetadataFailed.
func decodeSecondaryMetadata(symbol *core.DecodedSymbol, dockedPosition int, data []byte, offset int) int {
	// Ports decodeSlaveMetadata in decoder.c.
	symbol.SecondaryMeta[dockedPosition].NC = symbol.Meta.NC
	symbol.SecondaryMeta[dockedPosition].MaskType = symbol.Meta.MaskType
	symbol.SecondaryMeta[dockedPosition].DockedPosition = 0

	index := offset
	if index < 0 {
		return MetadataFailed
	}
	ss := data[index]
	index--
	if ss == 0 {
		symbol.SecondaryMeta[dockedPosition].SideVersion = symbol.Meta.SideVersion
	}
	if index < 0 {
		return MetadataFailed
	}
	se := data[index]
	index--
	if se == 0 {
		symbol.SecondaryMeta[dockedPosition].ECL = symbol.Meta.ECL
	}

	if ss == 1 {
		if index-4 < 0 {
			return MetadataFailed
		}
		v := 0
		for i := range 5 {
			v += int(data[index]) << (4 - i)
			index--
		}
		sideVersion := v + 1
		if dockedPosition == 2 || dockedPosition == 3 {
			symbol.SecondaryMeta[dockedPosition].SideVersion.Y = symbol.Meta.SideVersion.Y
			symbol.SecondaryMeta[dockedPosition].SideVersion.X = sideVersion
		} else {
			symbol.SecondaryMeta[dockedPosition].SideVersion.X = symbol.Meta.SideVersion.X
			symbol.SecondaryMeta[dockedPosition].SideVersion.Y = sideVersion
		}
	}
	if se == 1 {
		if index-5 < 0 {
			return MetadataFailed
		}
		e := 0
		for i := range 3 {
			e += int(data[index]) << (2 - i)
			index--
		}
		symbol.SecondaryMeta[dockedPosition].ECL.X = e + 3
		e = 0
		for i := range 3 {
			e += int(data[index]) << (2 - i)
			index--
		}
		symbol.SecondaryMeta[dockedPosition].ECL.Y = e + 4
		if symbol.SecondaryMeta[dockedPosition].ECL.X >= symbol.SecondaryMeta[dockedPosition].ECL.Y {
			return MetadataFailed
		}
	}
	return offset - index
}

// readRawModuleData reads the color index of every data module in column-major
// order.
func readRawModuleData(matrix *core.Bitmap, symbol *core.DecodedSymbol, dataMap []byte, normPalette, palThs []float64) []byte {
	// Ports readRawModuleData in decoder.c.
	colorNumber := 1 << (symbol.Meta.NC + 1)
	data := make([]byte, 0, matrix.Width*matrix.Height)
	for j := 0; j < matrix.Width; j++ {
		for i := 0; i < matrix.Height; i++ {
			if dataMap[i*matrix.Width+j] == 0 {
				data = append(data, DecodeModuleHD(matrix, symbol.Palette, colorNumber, normPalette, palThs, j, i))
			}
		}
	}
	return data
}

// readModuleReliabilities reads the soft-decision per-bit reliabilities of every
// data module in the same column-major order as readRawModuleData, so the result
// aligns bit-for-bit with rawModuleData2RawData's output.
func readModuleReliabilities(matrix *core.Bitmap, symbol *core.DecodedSymbol, dataMap []byte, normPalette []float64) []float64 {
	colorNumber := 1 << (symbol.Meta.NC + 1)
	rel := make([]float64, 0, matrix.Width*matrix.Height*(symbol.Meta.NC+1))
	for j := 0; j < matrix.Width; j++ {
		for i := 0; i < matrix.Height; i++ {
			if dataMap[i*matrix.Width+j] == 0 {
				rel = moduleReliabilities(matrix, colorNumber, symbol.Palette, normPalette, j, i, rel)
			}
		}
	}
	return rel
}

// rawModuleData2RawData expands per-module color indices into a one-bit-per-byte
// stream.
func rawModuleData2RawData(raw []byte, bitsPerModule int) []byte {
	// Ports rawModuleData2RawData in decoder.c.
	out := make([]byte, len(raw)*bitsPerModule)
	for i, m := range raw {
		for j := range bitsPerModule {
			out[i*bitsPerModule+j] = (m >> (bitsPerModule - 1 - j)) & 1
		}
	}
	return out
}

// DecodeSymbol reads, demasks, deinterleaves and error-corrects a symbol's data
// modules, storing the net payload in symbol.data.
func DecodeSymbol(matrix *core.Bitmap, symbol *core.DecodedSymbol, dataMap []byte, normPalette, palThs []float64, typ int) int {
	// Ports decodeSymbol in decoder.c.
	fillDataMap(dataMap, matrix.Width, matrix.Height, typ)

	rawModuleData := readRawModuleData(matrix, symbol, dataMap, normPalette, palThs)
	demaskSymbol(rawModuleData, dataMap, symbol.SideSize, symbol.Meta.MaskType, 1<<(symbol.Meta.NC+1))
	rawData := rawModuleData2RawData(rawModuleData, symbol.Meta.NC+1)

	wc := symbol.Meta.ECL.X
	wr := symbol.Meta.ECL.Y
	Pg := (len(rawData) / wr) * wr
	Pn := Pg * (wr - wc) / wr

	rawData = rawData[:Pg] // drop padding bits
	ecc.Deinterleave(rawData)

	// A nonzero post-correction syndrome means the bit-flipping corrector gave
	// up: the stream is garbage of the right length, and parsing it can return
	// err=nil with a corrupted payload.
	dec, eccOK := ecc.DecodeLDPCHard(rawData, wc, wr)
	if !eccOK || len(dec) != Pn {
		// Hard decoding failed. Retry with soft decoding: the classification
		// margins give belief propagation per-bit confidences that recover
		// colour confusions (from misregistration, ink spread) the hard
		// decoder cannot. A clean capture decodes hard and never reaches here,
		// so the clean path stays byte-identical.
		if soft := decodeSymbolSoft(matrix, symbol, dataMap, normPalette, rawData, wc, wr, Pn); soft != nil {
			return decodeSymbolStream(soft, symbol, typ)
		}
		return core.Failure
	}
	return decodeSymbolStream(dec, symbol, typ)
}

// decodeSymbolSoft re-decodes the data modules with soft-decision LDPC, reusing
// the deinterleaved hard bits as belief propagation's starting point and the
// classification margins as its per-bit reliabilities. It returns the net data
// stream, or nil when soft decoding also fails.
func decodeSymbolSoft(matrix *core.Bitmap, symbol *core.DecodedSymbol, dataMap []byte, normPalette []float64, hard []byte, wc, wr, Pn int) []byte {
	rel := readModuleReliabilities(matrix, symbol, dataMap, normPalette)
	if len(rel) < len(hard) {
		return nil
	}
	rel = rel[:len(hard)] // drop padding bits, matching hard's Pg length
	ecc.DeinterleaveFloat(rel)
	dec, ok := ecc.DecodeLDPCSoft(rel, hard, wc, wr)
	if !ok || len(dec) != Pn {
		return nil
	}
	return dec
}

// decodeSymbolStream parses the error-corrected data stream's in-stream
// metadata (the docked-position field and any docked secondaries' metadata)
// and stores the net payload in symbol.data.
//
// The hard LDPC decode is best-effort, so the stream can be garbage that kept
// the right length: a stream with no set start flag, too few bits for the
// docked-position field, or unparseable secondary metadata is such garbage,
// not a valid symbol stream. All three cases return Failure - a failed
// decode, not a fatal condition - so the caller's alignment-pattern resample
// still gets its chance. (The C reference scans for the start flag unbounded,
// undefined behaviour on an all-zero stream, and propagates a fatal status on
// unparseable secondary metadata, forfeiting that retry.)
func decodeSymbolStream(dec []byte, symbol *core.DecodedSymbol, typ int) int {
	// Locate the start flag (last set bit) of the in-stream metadata.
	metaOffset := len(dec) - 1
	for metaOffset >= 0 && dec[metaOffset] == 0 {
		metaOffset--
	}
	metaOffset-- // skip the flag bit

	symbol.Meta.DockedPosition = 0
	for i := range 4 {
		if typ == 1 && i == symbol.HostPosition {
			continue
		}
		if metaOffset < 0 {
			return core.Failure
		}
		symbol.Meta.DockedPosition += int(dec[metaOffset]) << (3 - i)
		metaOffset--
	}
	for i := range 4 {
		if symbol.Meta.DockedPosition&(0x08>>i) != 0 {
			readBitLength := decodeSecondaryMetadata(symbol, i, dec, metaOffset)
			if readBitLength == MetadataFailed {
				return core.Failure
			}
			metaOffset -= readBitLength
		}
	}

	netDataLength := metaOffset + 1
	symbol.Data = make([]byte, netDataLength)
	copy(symbol.Data, dec[:netDataLength])
	return core.Success
}

// DecodePrimary decodes a primary symbol from its sampled matrix.
func DecodePrimary(matrix *core.Bitmap, symbol *core.DecodedSymbol) int {
	// Ports decodePrimary in decoder.c.
	if matrix == nil {
		return core.FatalError
	}
	symbol.SideSize = image.Pt(matrix.Width, matrix.Height)
	dataMap := make([]byte, matrix.Width*matrix.Height)

	x, y := spec.PrimaryMetadataX, spec.PrimaryMetadataY
	moduleCount := 0

	partIRet := DecodePrimaryMetadataPartI(matrix, symbol, dataMap, &moduleCount, &x, &y)
	if partIRet == core.Failure {
		return core.Failure
	}
	if partIRet == MetadataFailed {
		x, y = spec.PrimaryMetadataX, spec.PrimaryMetadataY
		moduleCount = 0
		clear(dataMap)
		LoadDefaultPrimaryMetadata(matrix, symbol)
	}

	if ReadColorPaletteInPrimary(matrix, symbol, dataMap, &moduleCount, &x, &y) < 0 {
		return core.Failure
	}

	colorNumber := 1 << (symbol.Meta.NC + 1)
	normPalette := make([]float64, colorNumber*4*spec.ColorPaletteNumber)
	NormalizeColorPalette(symbol, normPalette, colorNumber)
	palThs := make([]float64, 3*spec.ColorPaletteNumber)
	for i := range spec.ColorPaletteNumber {
		t := PaletteThreshold(symbol.Palette[colorNumber*3*i:], colorNumber)
		palThs[i*3+0], palThs[i*3+1], palThs[i*3+2] = t[0], t[1], t[2]
	}

	if partIRet == core.Success {
		if DecodePrimaryMetadataPartII(matrix, symbol, dataMap, normPalette, palThs, &moduleCount, &x, &y) <= 0 {
			return core.Failure
		}
	}

	return DecodeSymbol(matrix, symbol, dataMap, normPalette, palThs, 0)
}
