package jabcode

import (
	"image"
	"image/color"
)

// getNextMetadataModuleInPrimary advances (x, y) to the next metadata/palette
// module position in a primary symbol (the primary-symbol metadata module walk
// in decoder.c). count is the running module index.
func getNextMetadataModuleInPrimary(height, width, count int, x, y *int) {
	if count%4 == 0 || count%4 == 2 {
		*y = height - 1 - *y
	}
	if count%4 == 1 || count%4 == 3 {
		*x = width - 1 - *x
	}
	if count%4 == 0 {
		switch {
		case count <= 20 || (count >= 44 && count <= 68) || (count >= 96 && count <= 124) || (count >= 156 && count <= 172):
			*y++
		case (count > 20 && count < 44) || (count > 68 && count < 96) || (count > 124 && count < 156):
			*x--
		}
	}
	if count == 44 || count == 96 || count == 156 {
		*x, *y = *y, *x
	}
}

// colorPaletteIndex returns the placement order of palette color indices
// (getColorPaletteIndex in encoder.c).
func colorPaletteIndex(size, colorNumber int) []byte {
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
// patterns, the embedded color palette and metadata, and the data modules
// (createMatrix in encoder.c). ecc is the interleaved, LDPC-encoded payload.
func (e *Encoder) createMatrix(index int, ecc []byte) {
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

	nc := log2int(e.colors) - 1
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
func (e *Encoder) placeAlignmentPatterns(s *symbol, set func(int, int, byte), nc int) {
	w, h := s.sideSize.X, s.sideSize.Y
	apxCore := byte(apxCoreColor[nc])
	apxPeri := byte(apnCoreColor[nc])
	vx := size2version(w) - 1
	vy := size2version(h) - 1
	for x := 0; x < apNum[vx]; x++ {
		left := x%2 == 0
		for y := 0; y < apNum[vy]; y++ {
			xo := apPos[vx][x] - 1
			yo := apPos[vy][y] - 1
			corner := (x == 0 || x == apNum[vx]-1) && (y == 0 || y == apNum[vy]-1)
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
func (e *Encoder) placePrimaryFinderPatterns(s *symbol, set func(int, int, byte), nc int) {
	w, h := s.sideSize.X, s.sideSize.Y
	const d = distanceToBorder
	for k := range 3 {
		fp0, fp1, fp2, fp3 := fpLayerColors(k, nc)
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
func (e *Encoder) placeSecondaryFinderPatterns(s *symbol, set func(int, int, byte), nc int) {
	w, h := s.sideSize.X, s.sideSize.Y
	const d = distanceToBorder
	for k := range 2 {
		var color byte
		if k%2 == 1 {
			color = byte(apxCoreColor[nc])
		} else {
			color = byte(apnCoreColor[nc])
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
func fpLayerColors(k, nc int) (fp0, fp1, fp2, fp3 byte) {
	c0 := byte(fpCoreColor[0][nc])
	c1 := byte(fpCoreColor[1][nc])
	c2 := byte(fpCoreColor[2][nc])
	c3 := byte(fpCoreColor[3][nc])
	if k%2 == 1 {
		return c3, c2, c1, c0
	}
	return c0, c1, c2, c3
}

// placePaletteAndMetadata embeds the color palette (and, for non-default primary
// symbols, the metadata) into the reserved module positions.
func (e *Encoder) placePaletteAndMetadata(index int, set func(int, int, byte)) {
	s := &e.symbols[index]
	w, h := s.sideSize.X, s.sideSize.Y

	palSize := min(e.colors, 64)
	palIndex := colorPaletteIndex(palSize, e.colors)
	paletteCount := min(e.colors, 64)

	if index == 0 {
		x, y := primaryMetadataX, primaryMetadataY
		count := 0
		mi := 0

		// Non-default symbols embed metadata Part I (Nc) before the palette: two
		// modules per 3 bits, via the Nc color-encoding table.
		if !e.isDefaultMode() {
			for mi < len(s.metadata) && mi < primaryMetadataPart1Length {
				val := int(s.metadata[mi])<<2 + int(s.metadata[mi+1])<<1 + int(s.metadata[mi+2])
				for k := range 2 {
					set(x, y, byte(ncColorEncode[val][k]%e.colors))
					count++
					getNextMetadataModuleInPrimary(h, w, count, &x, &y)
				}
				mi += 3
			}
		}

		// Color palette (first two colors live in the finder pattern).
		for i := 2; i < paletteCount; i++ {
			for p := range 4 {
				set(x, y, palIndex[primaryPalettePlacement[p][i]%e.colors])
				count++
				getNextMetadataModuleInPrimary(h, w, count, &x, &y)
			}
		}

		// Non-default symbols then embed metadata Part II.
		if !e.isDefaultMode() {
			bpm := log2int(e.colors)
			for mi < len(s.metadata) {
				colorIndex := 0
				for j := 0; j < bpm && mi < len(s.metadata); j++ {
					colorIndex += int(s.metadata[mi]) << (bpm - 1 - j)
					mi++
				}
				set(x, y, byte(colorIndex))
				count++
				getNextMetadataModuleInPrimary(h, w, count, &x, &y)
			}
		}
		return
	}

	// Secondary symbol: palette is placed at four rotations around the border.
	for i := 2; i < paletteCount; i++ {
		px := secondaryPalettePosition[i-2].X
		py := secondaryPalettePosition[i-2].Y
		color := palIndex[secondaryPalettePlacement[i]%e.colors]
		set(px, py, color)         // left
		set(w-1-py, px, color)     // top
		set(w-1-px, h-1-py, color) // right
		set(py, h-1-px, color)     // bottom
	}
}

// placeData writes the ecc payload (and padding) into the data modules in
// column-major order, packing log2(colors) bits per module.
func (e *Encoder) placeData(s *symbol, ecc []byte) {
	w, h := s.sideSize.X, s.sideSize.Y
	bpm := log2int(e.colors)
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

// maskValue returns the mask offset for module (x, y) under the given pattern
// (maskSymbols/demaskSymbol in mask.c).
func maskValue(maskType, x, y int) int {
	switch maskType {
	case 0:
		return x + y
	case 1:
		return x
	case 2:
		return y
	case 3:
		return x/2 + y/3
	case 4:
		return x/3 + y/2
	case 5:
		return (x+y)/2 + (x+y)/3
	case 6:
		return (x*x*y)%7 + (2*x*x+2*y)%19
	case 7:
		return (x*y*y)%5 + (2*x+y*y)%13
	}
	return 0
}

// maskSymbol XORs the mask pattern into the data modules of a symbol.
func (e *Encoder) maskSymbol(index, maskType int) {
	s := &e.symbols[index]
	w, h := s.sideSize.X, s.sideSize.Y
	for y := range h {
		for x := range w {
			if s.dataMap[y*w+x] == 0 {
				continue
			}
			v := int(s.matrix[y*w+x])
			v ^= maskValue(maskType, x, y) % e.colors
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
// each module to moduleSize pixels (createBitmap in encoder.c). A JAB Code is
// naturally a paletted image: every module is an index into the color palette.
func (e *Encoder) createBitmap() {
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
