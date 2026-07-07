package decode

import (
	"image"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/tables"
)

// readColorPaletteInSecondary reconstructs the four color palettes embedded in a
// secondary symbol.
func readColorPaletteInSecondary(matrix *core.Bitmap, symbol *core.DecodedSymbol, dataMap []byte) int {
	// Ports readColorPaletteInSlave in decoder.c.
	colorNumber := 1 << (symbol.Meta.NC + 1)
	// The secondary palette-position table places at most 32 embedded colors;
	// beyond that (64-color and up) the layout is undefined, so those symbols
	// cannot be a docked secondary. Reject rather than index the table OOB.
	if min(colorNumber, 64)-2 > len(tables.SecondaryPalettePosition) {
		return MetadataFailed
	}
	symbol.Palette = make([]byte, colorNumber*3*spec.ColorPaletteNumber)

	for i := range spec.ColorPaletteNumber {
		p1, p2 := colorPalettePosInFP(i, matrix.Width, matrix.Height)
		writeColorPalette(matrix, symbol, i, tables.SecondaryPalettePlacementIndex(0)%colorNumber, p1.X, p1.Y)
		writeColorPalette(matrix, symbol, i, tables.SecondaryPalettePlacementIndex(1)%colorNumber, p2.X, p2.Y)
	}

	for colorCounter := 2; colorCounter < min(colorNumber, 64); colorCounter++ {
		ci := tables.SecondaryPalettePlacementIndex(colorCounter) % colorNumber
		pos := tables.SecondaryPalettePosition[colorCounter-2]
		// The palette is placed at four rotations around the border. Skip any
		// rotation landing outside the matrix so a wrongly-sized symbol fails
		// downstream, not by indexing out of range.
		for pIndex, pt := range [4]image.Point{
			{pos.X, pos.Y},
			{matrix.Width - 1 - pos.Y, pos.X},
			{matrix.Width - 1 - pos.X, matrix.Height - 1 - pos.Y},
			{pos.Y, matrix.Height - 1 - pos.X},
		} {
			if pt.X < 0 || pt.Y < 0 || pt.X >= matrix.Width || pt.Y >= matrix.Height {
				continue
			}
			writeColorPalette(matrix, symbol, pIndex, ci, pt.X, pt.Y)
			dataMap[pt.Y*matrix.Width+pt.X] = 1
		}
	}
	if colorNumber > 64 {
		interpolatePalette(symbol.Palette, colorNumber)
	}
	return core.Success
}

// DecodeSecondary decodes a secondary symbol from its sampled matrix.
func DecodeSecondary(matrix *core.Bitmap, symbol *core.DecodedSymbol) int {
	// Ports decodeSlave in decoder.c.
	if matrix == nil {
		return core.FatalError
	}
	dataMap := make([]byte, matrix.Width*matrix.Height)
	if readColorPaletteInSecondary(matrix, symbol, dataMap) < 0 {
		return core.FatalError
	}

	colorNumber := 1 << (symbol.Meta.NC + 1)
	normPalette := make([]float64, colorNumber*4*spec.ColorPaletteNumber)
	NormalizeColorPalette(symbol, normPalette, colorNumber)
	palThs := make([]float64, 3*spec.ColorPaletteNumber)
	for i := range spec.ColorPaletteNumber {
		// Note: the reference offsets by i*3 (not colorNumber*3*i) here; kept identical.
		t := PaletteThreshold(symbol.Palette[i*3:], colorNumber)
		palThs[i*3+0], palThs[i*3+1], palThs[i*3+2] = t[0], t[1], t[2]
	}
	return DecodeSymbol(matrix, symbol, dataMap, normPalette, palThs, 1)
}
