package encode

import (
	"image"
	"image/color"

	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/tables"
)

// colorPaletteIndex returns the placement order of palette color indices.
func colorPaletteIndex(size, colorNumber int) []byte {
	// Ports getColorPaletteIndex in encoder.c.
	index := make([]byte, size)
	for i := range index {
		index[i] = byte(i)
	}
	if colorNumber < 128 {
		return index
	}
	switch colorNumber {
	case 128:
		for k, src := range []int{0, 32, 80, 112} {
			for i := range 16 {
				index[k*16+i] = byte(src + i)
			}
		}
	case 256:
		for k, src := range []int{0, 8, 20, 28, 64, 72, 84, 92, 160, 168, 180, 188, 224, 232, 244, 252} {
			for i := range 4 {
				index[k*4+i] = byte(src + i)
			}
		}
	}
	return index
}

// createMatrix builds the module matrix for symbol index: finder and alignment
// patterns, the embedded color palette and metadata, and the data modules. ecc
// is the interleaved, LDPC-encoded payload.
func (e *encoder) createMatrix(index int, ecc []byte) {
	// Ports createMatrix in encoder.c.
	s := &e.symbols[index]
	w, h := s.sideSize.X, s.sideSize.Y
	s.matrix = make([]byte, w*h)
	s.dataMap = make([]byte, w*h)
	for i := range s.dataMap {
		s.dataMap[i] = 1
	}
	// set places a module color and marks the cell as reserved (non-data).
	set := func(x, y int, color byte) {
		s.matrix[y*w+x] = color
		s.dataMap[y*w+x] = 0
	}

	nc := spec.Log2Int(e.colors) - 1
	e.placeAlignmentPatterns(s, set, nc)
	if index == 0 {
		e.placePrimaryFinderPatterns(s, set, nc)
	} else {
		e.placeSecondaryFinderPatterns(s, set, nc)
	}
	e.placePaletteAndMetadata(index, set)
	e.placeData(s, ecc)
}

// placeAlignmentPatterns places the interior alignment patterns (the APX cross
// shared by primary and secondary symbols).
func (e *encoder) placeAlignmentPatterns(s *symbol, set func(int, int, byte), nc int) {
	w, h := s.sideSize.X, s.sideSize.Y
	apxCore := byte(tables.APXCoreColorIndex(nc, e.profile))
	apxPeri := byte(tables.APNCoreColorIndex(nc, e.profile))
	vx := spec.SizeToVersion(w) - 1
	vy := spec.SizeToVersion(h) - 1
	for x := 0; x < tables.APNum[vx]; x++ {
		left := x%2 == 0
		for y := 0; y < tables.APNum[vy]; y++ {
			xo := tables.APPos[vx][x] - 1
			yo := tables.APPos[vy][y] - 1
			corner := (x == 0 || x == tables.APNum[vx]-1) && (y == 0 || y == tables.APNum[vy]-1)
			if !corner {
				if left {
					set(xo-1, yo-1, apxPeri)
					set(xo, yo-1, apxPeri)
					set(xo-1, yo, apxPeri)
					set(xo+1, yo, apxPeri)
					set(xo, yo+1, apxPeri)
					set(xo+1, yo+1, apxPeri)
				} else {
					set(xo+1, yo-1, apxPeri)
					set(xo, yo-1, apxPeri)
					set(xo-1, yo, apxPeri)
					set(xo+1, yo, apxPeri)
					set(xo, yo+1, apxPeri)
					set(xo-1, yo+1, apxPeri)
				}
				set(xo, yo, apxCore)
			}
			left = !left
		}
	}
}

// placePrimaryFinderPatterns places the four 3-layer finder patterns at the
// corners of a primary symbol.
func (e *encoder) placePrimaryFinderPatterns(s *symbol, set func(int, int, byte), nc int) {
	w, h := s.sideSize.X, s.sideSize.Y
	const d = spec.DistanceToBorder
	for k := range 3 {
		fp0, fp1, fp2, fp3 := e.fpLayerColors(k, nc)
		for i := 0; i < k+1; i++ {
			for j := 0; j < k+1; j++ {
				if i != k && j != k {
					continue
				}
				// FP0 top-left, FP1 top-right, FP2 bottom-right, FP3 bottom-left.
				set(d-j-1, d-(i+1), fp0)
				set(d+j-1, d+(i-1), fp0)
				set(w-(d-1)-j-1, d-(i+1), fp1)
				set(w-(d-1)+j-1, d+(i-1), fp1)
				set(w-(d-1)-j-1, h-d+i, fp2)
				set(w-(d-1)+j-1, h-d-i, fp2)
				set(d-j-1, h-d+i, fp3)
				set(d+j-1, h-d-i, fp3)
			}
		}
	}
}

// placeSecondaryFinderPatterns places the four 2-layer alignment patterns at the
// corners of a secondary symbol.
func (e *encoder) placeSecondaryFinderPatterns(s *symbol, set func(int, int, byte), nc int) {
	w, h := s.sideSize.X, s.sideSize.Y
	const d = spec.DistanceToBorder
	for k := range 2 {
		var color byte
		if k%2 == 1 {
			color = byte(tables.APXCoreColorIndex(nc, e.profile))
		} else {
			color = byte(tables.APNCoreColorIndex(nc, e.profile))
		}
		for i := 0; i < k+1; i++ {
			for j := 0; j < k+1; j++ {
				if i != k && j != k {
					continue
				}
				set(d-j-1, d-(i+1), color)
				set(d+j-1, d+(i-1), color)
				set(w-(d-1)-j-1, d-(i+1), color)
				set(w-(d-1)+j-1, d+(i-1), color)
				set(w-(d-1)-j-1, h-d+i, color)
				set(w-(d-1)+j-1, h-d-i, color)
				set(d-j-1, h-d+i, color)
				set(d+j-1, h-d-i, color)
			}
		}
	}
}

// fpLayerColors returns the four finder-pattern colors for concentric layer k
// (alternating per layer).
func (e *encoder) fpLayerColors(k, nc int) (fp0, fp1, fp2, fp3 byte) {
	c0 := byte(tables.FPCoreColorIndex(0, nc, e.profile))
	c1 := byte(tables.FPCoreColorIndex(1, nc, e.profile))
	c2 := byte(tables.FPCoreColorIndex(2, nc, e.profile))
	c3 := byte(tables.FPCoreColorIndex(3, nc, e.profile))
	if k%2 == 1 {
		return c3, c2, c1, c0
	}
	return c0, c1, c2, c3
}

// placePaletteAndMetadata embeds the color palette (and, for non-default primary
// symbols, the metadata) into the reserved module positions.
func (e *encoder) placePaletteAndMetadata(index int, set func(int, int, byte)) {
	s := &e.symbols[index]
	w, h := s.sideSize.X, s.sideSize.Y

	palSize := min(e.colors, 64)
	palIndex := colorPaletteIndex(palSize, e.colors)
	paletteCount := min(e.colors, 64)
	copies := spec.PaletteCopies(e.colors)

	if index == 0 {
		x, y := spec.PrimaryMetadataX, spec.PrimaryMetadataY
		count := 0
		mi := 0

		// Non-default symbols embed metadata Part I (Nc) before the palette: two
		// modules per 3 bits, via the Nc color-encoding table. The colors are
		// placed at the per-mode palette index carrying them, so Part I reads back
		// by color pattern before the palette is known.
		nc := spec.Log2Int(e.colors) - 1
		if !e.isDefaultMode() {
			for mi < len(s.metadata) && mi < spec.PrimaryMetadataPart1Length {
				val := int(s.metadata[mi])<<2 + int(s.metadata[mi+1])<<1 + int(s.metadata[mi+2])
				for k := range 2 {
					set(x, y, byte(tables.NcMetadataColorIndexProfile(tables.NcColorEncode[val][k], nc, e.profile)))
					count++
					spec.NextMetadataModuleInPrimary(h, w, count, &x, &y)
				}
				mi += 3
			}
		}

		// Color palette. In 4/8-color symbols the first two colors live in the
		// finder pattern; the higher modes embed every color here instead (the 64
		// representatives for 128/256, interpolated back on decode), since their
		// finder cores are not palette colors 0 and 1 (ISO/IEC 23634 Annex G:
		// "all available colours should be included in the embedded colour
		// palettes").
		firstColor := spec.PaletteFinderColors(e.colors)
		for i := firstColor; i < paletteCount; i++ {
			for p := range copies {
				set(x, y, palIndex[tables.PrimaryPalettePlacementIndexProfile(p, i, e.colors, e.profile)%e.colors])
				count++
				spec.NextMetadataModuleInPrimary(h, w, count, &x, &y)
			}
		}

		// Non-default symbols then embed metadata Part II.
		if !e.isDefaultMode() {
			bpm := spec.Log2Int(e.colors)
			for mi < len(s.metadata) {
				colorIndex := 0
				for j := 0; j < bpm && mi < len(s.metadata); j++ {
					colorIndex += int(s.metadata[mi]) << (bpm - 1 - j)
					mi++
				}
				set(x, y, byte(colorIndex))
				count++
				spec.NextMetadataModuleInPrimary(h, w, count, &x, &y)
			}
		}
		return
	}

	// Secondary symbol: palette is placed at up to four rotations around the
	// border - one per embedded copy (four for 4/8-color, two for the higher
	// modes). In 4/8-color symbols the first two colors live in the alignment
	// patterns, so the position table starts at color 2; the higher modes embed
	// every color here (ISO Annex G), so it starts at color 0.
	firstColor := spec.PaletteFinderColors(e.colors)
	for i := firstColor; i < paletteCount; i++ {
		pos := tables.SecondaryPalettePosition[i-firstColor]
		color := palIndex[tables.SecondaryPalettePlacementIndexProfile(i, e.colors, e.profile)%e.colors]
		rot := [4][2]int{{pos.X, pos.Y}, {w - 1 - pos.Y, pos.X}, {w - 1 - pos.X, h - 1 - pos.Y}, {pos.Y, h - 1 - pos.X}}
		for p := range copies {
			set(rot[p][0], rot[p][1], color)
		}
	}
}

// placeData writes the ecc payload (and padding) into the data modules in
// column-major order, packing log2(colors) bits per module.
func (e *encoder) placeData(s *symbol, ecc []byte) {
	w, h := s.sideSize.X, s.sideSize.Y
	bpm := spec.Log2Int(e.colors)
	written := 0
	padding := 0
	for startX := range w {
		for i := startX; i < w*h; i += w {
			if s.dataMap[i] == 0 {
				continue
			}
			colorIndex := 0
			for j := range bpm {
				if written < len(ecc) {
					colorIndex += int(ecc[written]) << (bpm - 1 - j)
				} else {
					colorIndex += padding << (bpm - 1 - j)
					padding = 1 - padding
				}
				written++
			}
			s.matrix[i] = byte(colorIndex)
		}
	}
}

// maskSymbol XORs the mask pattern into the data modules of a symbol.
func (e *encoder) maskSymbol(index, maskType int) {
	s := &e.symbols[index]
	w, h := s.sideSize.X, s.sideSize.Y
	for y := range h {
		for x := range w {
			if s.dataMap[y*w+x] == 0 {
				continue
			}
			v := int(s.matrix[y*w+x])
			v ^= spec.MaskValue(maskType, x, y) % e.colors
			s.matrix[y*w+x] = byte(v)
		}
	}
}

// rgbPalette converts packed RGB triples into a color.Palette of opaque colors.
func rgbPalette(rgb []byte) color.Palette {
	pal := make(color.Palette, len(rgb)/3)
	for i := range pal {
		pal[i] = color.NRGBA{R: rgb[i*3], G: rgb[i*3+1], B: rgb[i*3+2], A: 255}
	}
	return pal
}

// createBitmap renders the (single) symbol matrix into a paletted image, scaling
// each module to moduleSize pixels. A JAB Code is naturally a paletted image:
// every module is an index into the color palette.
func (e *encoder) createBitmap() {
	// Ports createBitmap in encoder.c.
	s := &e.symbols[0]
	dim := e.moduleSize
	w := s.sideSize.X
	img := image.NewPaletted(image.Rect(0, 0, dim*w, dim*s.sideSize.Y), rgbPalette(e.palette))
	for my := 0; my < s.sideSize.Y; my++ {
		for mx := range w {
			idx := s.matrix[my*w+mx]
			for py := my * dim; py < my*dim+dim; py++ {
				for px := mx * dim; px < mx*dim+dim; px++ {
					img.SetColorIndex(px, py, idx)
				}
			}
		}
	}
	e.bitmap = img
}
