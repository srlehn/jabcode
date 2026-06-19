package jabcode

// readColorPaletteInSecondary reconstructs the four color palettes embedded in a
// secondary symbol (readColorPaletteInSlave in decoder.c).
func readColorPaletteInSecondary(matrix *bitmap, symbol *decodedSymbol, dataMap []byte) int {
	colorNumber := 1 << (symbol.meta.Nc + 1)
	if colorNumber != 4 && colorNumber != 8 {
		// Only 4- and 8-color symbols are defined; higher modes are reserved.
		return decodeMetadataFailed
	}
	symbol.palette = make([]byte, colorNumber*3*colorPaletteNumber)

	for i := range colorPaletteNumber {
		p1, p2 := getColorPalettePosInFP(i, matrix.width, matrix.height)
		writeColorPalette(matrix, symbol, i, secondaryPalettePlacement[0]%colorNumber, p1.X, p1.Y)
		writeColorPalette(matrix, symbol, i, secondaryPalettePlacement[1]%colorNumber, p2.X, p2.Y)
	}

	for colorCounter := 2; colorCounter < min(colorNumber, 64); colorCounter++ {
		ci := secondaryPalettePlacement[colorCounter] % colorNumber
		pos := secondaryPalettePosition[colorCounter-2]

		px, py := pos.X, pos.Y
		writeColorPalette(matrix, symbol, 0, ci, px, py)
		dataMap[py*matrix.width+px] = 1

		px, py = matrix.width-1-pos.Y, pos.X
		writeColorPalette(matrix, symbol, 1, ci, px, py)
		dataMap[py*matrix.width+px] = 1

		px, py = matrix.width-1-pos.X, matrix.height-1-pos.Y
		writeColorPalette(matrix, symbol, 2, ci, px, py)
		dataMap[py*matrix.width+px] = 1

		px, py = pos.Y, matrix.height-1-pos.X
		writeColorPalette(matrix, symbol, 3, ci, px, py)
		dataMap[py*matrix.width+px] = 1
	}
	if colorNumber > 64 {
		interpolatePalette(symbol.palette, colorNumber)
	}
	return jabSuccess
}

// decodeSecondary decodes a secondary symbol from its sampled matrix
// (decodeSlave in decoder.c).
func decodeSecondary(matrix *bitmap, symbol *decodedSymbol) int {
	if matrix == nil {
		return fatalError
	}
	dataMap := make([]byte, matrix.width*matrix.height)
	if readColorPaletteInSecondary(matrix, symbol, dataMap) < 0 {
		return fatalError
	}

	colorNumber := 1 << (symbol.meta.Nc + 1)
	normPalette := make([]float64, colorNumber*4*colorPaletteNumber)
	normalizeColorPalette(symbol, normPalette, colorNumber)
	palThs := make([]float64, 3*colorPaletteNumber)
	for i := range colorPaletteNumber {
		// Note: the reference offsets by i*3 (not colorNumber*3*i) here; kept identical.
		t := getPaletteThreshold(symbol.palette[i*3:], colorNumber)
		palThs[i*3+0], palThs[i*3+1], palThs[i*3+2] = t[0], t[1], t[2]
	}
	return decodeSymbol(matrix, symbol, dataMap, normPalette, palThs, 1)
}
