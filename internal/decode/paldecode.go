package decode

import (
	"image"
	"math"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/tables"
)

// writeColorPalette records the RGB of module (x,y) as a palette entry.
func writeColorPalette(matrix *core.Bitmap, symbol *core.DecodedSymbol, pIndex, colorIndex, x, y int) {
	// Ports writeColorPalette in decoder.c.
	colorNumber := 1 << (symbol.Meta.NC + 1)
	bpp := matrix.Channels
	bytesPerRow := matrix.Width * bpp
	po := colorNumber * 3 * pIndex
	mo := y*bytesPerRow + x*bpp
	symbol.Palette[po+colorIndex*3+0] = matrix.Pix[mo+0]
	symbol.Palette[po+colorIndex*3+1] = matrix.Pix[mo+1]
	symbol.Palette[po+colorIndex*3+2] = matrix.Pix[mo+2]
}

// colorPalettePosInFP returns the two finder-pattern module positions that
// carry palette colors 0 and 1.
func colorPalettePosInFP(pIndex, w, h int) (p1, p2 image.Point) {
	// Ports getColorPalettePosInFP in decoder.c.
	switch pIndex {
	case 0:
		p1 = image.Pt(spec.DistanceToBorder-1, spec.DistanceToBorder-1)
		p2 = image.Pt(p1.X+1, p1.Y)
	case 1:
		p1 = image.Pt(w-spec.DistanceToBorder, spec.DistanceToBorder-1)
		p2 = image.Pt(p1.X-1, p1.Y)
	case 2:
		p1 = image.Pt(w-spec.DistanceToBorder, h-spec.DistanceToBorder)
		p2 = image.Pt(p1.X-1, p1.Y)
	case 3:
		p1 = image.Pt(spec.DistanceToBorder-1, h-spec.DistanceToBorder)
		p2 = image.Pt(p1.X+1, p1.Y)
	}
	return p1, p2
}

// ReadColorPaletteInPrimary reconstructs the four color palettes embedded in the
// primary symbol.
func ReadColorPaletteInPrimary(matrix *core.Bitmap, symbol *core.DecodedSymbol, dataMap []byte, moduleCount, x, y *int) int {
	// Ports readColorPaletteInPrimary in decoder.c.
	colorNumber := 1 << (symbol.Meta.NC + 1)
	if colorNumber != 4 && colorNumber != 8 {
		// Only 4- and 8-color symbols are defined (colour modes 1 and 2); higher
		// modes are reserved. Reject rather than index the palette table OOB.
		return MetadataFailed
	}
	symbol.Palette = make([]byte, colorNumber*3*spec.ColorPaletteNumber)

	for i := range spec.ColorPaletteNumber {
		p1, p2 := colorPalettePosInFP(i, matrix.Width, matrix.Height)
		writeColorPalette(matrix, symbol, i, tables.PrimaryPalettePlacement[i][0]%colorNumber, p1.X, p1.Y)
		writeColorPalette(matrix, symbol, i, tables.PrimaryPalettePlacement[i][1]%colorNumber, p2.X, p2.Y)
	}

	for colorCounter := 2; colorCounter < min(colorNumber, 64); colorCounter++ {
		for p := range 4 {
			writeColorPalette(matrix, symbol, p, tables.PrimaryPalettePlacement[p][colorCounter]%colorNumber, *x, *y)
			dataMap[(*y)*matrix.Width+(*x)] = 1
			(*moduleCount)++
			spec.NextMetadataModuleInPrimary(matrix.Height, matrix.Width, *moduleCount, x, y)
		}
	}
	if colorNumber > 64 {
		interpolatePalette(symbol.Palette, colorNumber)
	}
	return core.Success
}

// nearestPalette returns the index of the embedded palette nearest to module
// (x,y), so distortions are corrected per-corner.
func nearestPalette(matrix *core.Bitmap, x, y int) int {
	// Ports getNearestPalette in decoder.c.
	px := [4]int{spec.DistanceToBorder - 1 + 3, matrix.Width - spec.DistanceToBorder - 3, matrix.Width - spec.DistanceToBorder - 3, spec.DistanceToBorder - 1 + 3}
	py := [4]int{spec.DistanceToBorder - 1, spec.DistanceToBorder - 1, matrix.Height - spec.DistanceToBorder, matrix.Height - spec.DistanceToBorder}
	best := math.Hypot(float64(matrix.Width), float64(matrix.Height))
	pIndex := 0
	for i := range spec.ColorPaletteNumber {
		d := math.Hypot(float64(x-px[i]), float64(y-py[i]))
		if d < best {
			best = d
			pIndex = i
		}
	}
	return pIndex
}

// DecodeModuleHD maps the sampled RGB of module (x,y) to its palette index by
// nearest normalized color, with a black check and a black/white tie-break.
func DecodeModuleHD(matrix *core.Bitmap, palette []byte, colorNumber int, normPalette, palThs []float64, x, y int) byte {
	// Ports decodeModuleHD in decoder.c.
	pIndex := nearestPalette(matrix, x, y)
	bpp := matrix.Channels
	off := y*matrix.Width*bpp + x*bpp
	rgb := [3]byte{matrix.Pix[off], matrix.Pix[off+1], matrix.Pix[off+2]}

	var index1 byte
	if float64(rgb[0]) < palThs[pIndex*3+0] && float64(rgb[1]) < palThs[pIndex*3+1] && float64(rgb[2]) < palThs[pIndex*3+2] {
		return 0
	}
	if palette == nil {
		c := core.BoolColor(rgb[0] > 100)/255 + core.BoolColor(rgb[1] > 100)/255 + core.BoolColor(rgb[2] > 100)/255
		if c > 1 {
			return 1
		}
		return 0
	}

	rgbMax := float64(max(rgb[0], rgb[1], rgb[2]))
	r := float64(rgb[0]) / rgbMax
	g := float64(rgb[1]) / rgbMax
	b := float64(rgb[2]) / rgbMax
	min1, min2 := 255.0*255*3, 255.0*255*3
	var index2 byte
	for i := range colorNumber {
		base := colorNumber*4*pIndex + i*4
		pr, pg, pb := normPalette[base+0], normPalette[base+1], normPalette[base+2]
		diff := (pr-r)*(pr-r) + (pg-g)*(pg-g) + (pb-b)*(pb-b)
		if diff < min1 {
			min2, index2 = min1, index1
			min1, index1 = diff, byte(i)
		} else if diff < min2 {
			min2, index2 = diff, byte(i)
		}
	}
	_ = index2

	// The black/white tie-break only exists in the 8-colour palette; a 4-colour
	// palette has no white entry and indexing entry 7 would read past the
	// corner palette's four entries.
	if colorNumber == 8 && (index1 == 0 || index1 == 7) {
		rgbSum := int(rgb[0]) + int(rgb[1]) + int(rgb[2])
		p0 := colorNumber * 3 * pIndex
		p7 := p0 + 7*3
		p0Sum := int(palette[p0+0]) + int(palette[p0+1]) + int(palette[p0+2])
		p7Sum := int(palette[p7+0]) + int(palette[p7+1]) + int(palette[p7+2])
		if rgbSum < (p0Sum+p7Sum)/2 {
			index1 = 0
		} else {
			index1 = 7
		}
	}
	return index1
}

// moduleReliabilities appends the max-log soft-decision reliabilities of module
// (x,y)'s bitsPerModule index bits (MSB first, matching rawModuleData2RawData) to
// dst. A bit's reliability is the gap between the nearest candidate colour whose
// index has that bit set and the nearest whose index has it clear, in the same
// normalized-RGB distance DecodeModuleHD ranks by: a wide gap is a confident bit,
// a near-tie an uncertain one. Belief propagation uses these to correct colour
// confusions the hard decoder cannot. The magnitude is independent of the data
// mask (an XOR only flips the hard bit, not its confidence), so it needs no
// demasking before it rides the deinterleave alongside the bits.
func moduleReliabilities(matrix *core.Bitmap, colorNumber int, normPalette []float64, x, y int, dst []float64) []float64 {
	pIndex := nearestPalette(matrix, x, y)
	bpp := matrix.Channels
	off := y*matrix.Width*bpp + x*bpp
	rgbMax := float64(max(matrix.Pix[off], matrix.Pix[off+1], matrix.Pix[off+2]))
	if rgbMax == 0 {
		rgbMax = 1
	}
	r := float64(matrix.Pix[off+0]) / rgbMax
	g := float64(matrix.Pix[off+1]) / rgbMax
	b := float64(matrix.Pix[off+2]) / rgbMax
	var dist [8]float64
	for i := range colorNumber {
		base := colorNumber*4*pIndex + i*4
		dr := normPalette[base+0] - r
		dg := normPalette[base+1] - g
		db := normPalette[base+2] - b
		dist[i] = dr*dr + dg*dg + db*db
	}
	bpm := spec.Log2Int(colorNumber)
	for p := range bpm {
		shift := uint(bpm - 1 - p)
		min0, min1 := math.Inf(1), math.Inf(1)
		for c := range colorNumber {
			if (c>>shift)&1 == 0 {
				if dist[c] < min0 {
					min0 = dist[c]
				}
			} else if dist[c] < min1 {
				min1 = dist[c]
			}
		}
		rel := min1 - min0
		if rel < 0 {
			rel = -rel
		}
		dst = append(dst, rel)
	}
	return dst
}

// partIColorRefs derives the eight expected module colours of the sampled
// matrix from its finder cores, under an offset plus per-channel-gain display
// model: the two black cores give the offset, the cyan core (FP3) the green and
// blue gains, the yellow core (FP2) the red and green gains. The cores carry
// these colours in both 4- and 8-colour modes, and they are readable before any
// palette or metadata module, so the references need nothing but geometry. ok is
// false when a gain is non-positive - degenerate anchors on a wrongly-sampled
// matrix, in which case callers keep the plain classification.
func partIColorRefs(matrix *core.Bitmap) (refs [8][3]float64, ok bool) {
	w, h := matrix.Width, matrix.Height
	at := func(x, y int) [3]float64 {
		off := (y*w + x) * matrix.Channels
		return [3]float64{float64(matrix.Pix[off]), float64(matrix.Pix[off+1]), float64(matrix.Pix[off+2])}
	}
	b0, b1 := at(3, 3), at(w-4, 3)
	black := [3]float64{(b0[0] + b1[0]) / 2, (b0[1] + b1[1]) / 2, (b0[2] + b1[2]) / 2}
	yellow := at(w-4, h-4)
	cyan := at(3, h-4)
	gr := yellow[0] - black[0]
	gg := (yellow[1] - black[1] + cyan[1] - black[1]) / 2
	gb := cyan[2] - black[2]
	if gr <= 0 || gg <= 0 || gb <= 0 {
		return refs, false
	}
	for c := range 8 {
		refs[c] = [3]float64{
			black[0] + float64((c>>2)&1)*gr,
			black[1] + float64((c>>1)&1)*gg,
			black[2] + float64(c&1)*gb,
		}
	}
	return refs, true
}

// decodeModuleNCRef classifies a module colour to the nearest of the eight
// reference colours, returning the canonical palette index.
func decodeModuleNCRef(rgb []byte, refs *[8][3]float64) byte {
	best, bi := math.Inf(1), 0
	for c := range 8 {
		dr := float64(rgb[0]) - refs[c][0]
		dg := float64(rgb[1]) - refs[c][1]
		db := float64(rgb[2]) - refs[c][2]
		if d := dr*dr + dg*dg + db*db; d < best {
			best, bi = d, c
		}
	}
	return byte(bi)
}

// DecodeModuleNC decodes a primary-metadata Part I module color into its 3-bit
// value.
func DecodeModuleNC(rgb []byte) byte {
	// Ports decodeModuleNc in decoder.c.
	const thsBlack = 80
	const thsStd = 0.08
	if rgb[0] < thsBlack && rgb[1] < thsBlack && rgb[2] < thsBlack {
		return 0
	}
	_, variance := core.AvgVar(rgb)
	std := math.Sqrt(variance)
	_, _, mx, iMin, iMid, iMax := core.MinMax(rgb)
	std /= float64(mx)
	if std <= thsStd {
		return 7
	}
	var bits [3]byte
	bits[iMax] = 1
	bits[iMin] = 0
	if float64(rgb[iMid])/float64(rgb[iMin]) > float64(rgb[iMax])/float64(rgb[iMid]) {
		bits[iMid] = 1
	}
	return (bits[0] << 2) + (bits[1] << 1) + bits[2]
}

// PaletteThreshold returns the per-channel black thresholds, midway between
// the dark and light palette colors.
func PaletteThreshold(palette []byte, colorNumber int) [3]float64 {
	// Ports getPaletteThreshold in decoder.c.
	var ths [3]float64
	switch colorNumber {
	case 4:
		ths[0] = float64(int(max(palette[0], palette[3]))+int(min(palette[6], palette[9]))) / 2.0
		ths[1] = float64(int(max(palette[1], palette[7]))+int(min(palette[4], palette[10]))) / 2.0
		ths[2] = float64(int(max(palette[8], palette[11]))+int(min(palette[2], palette[5]))) / 2.0
	case 8:
		ths[0] = float64(int(max(palette[0], palette[3], palette[6], palette[9]))+int(min(palette[12], palette[15], palette[18], palette[21]))) / 2.0
		ths[1] = float64(int(max(palette[1], palette[4], palette[13], palette[16]))+int(min(palette[7], palette[10], palette[19], palette[22]))) / 2.0
		ths[2] = float64(int(max(palette[2], palette[8], palette[14], palette[20]))+int(min(palette[5], palette[11], palette[17], palette[23]))) / 2.0
	}
	return ths
}

// NormalizeColorPalette precomputes per-color normalized RGB + luminance values
// for nearest-color matching.
func NormalizeColorPalette(symbol *core.DecodedSymbol, normPalette []float64, colorNumber int) {
	// Ports normalizeColorPalette in decoder.c.
	p := symbol.Palette
	for i := 0; i < colorNumber*spec.ColorPaletteNumber; i++ {
		rgbMax := float64(max(p[i*3+0], p[i*3+1], p[i*3+2]))
		normPalette[i*4+0] = float64(p[i*3+0]) / rgbMax
		normPalette[i*4+1] = float64(p[i*3+1]) / rgbMax
		normPalette[i*4+2] = float64(p[i*3+2]) / rgbMax
		normPalette[i*4+3] = (float64(int(p[i*3+0])+int(p[i*3+1])+int(p[i*3+2])) / 3.0) / 255.0
	}
}

// interpolatePalette reconstructs the full palette for 128/256-color symbols.
func interpolatePalette(palette []byte, colorNumber int) {
	// Stub: not needed for <=64-color symbols. Ports interpolatePalette in
	// decoder.c; implement when adding high-color support.
}
