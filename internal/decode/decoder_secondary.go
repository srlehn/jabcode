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
	firstColor := spec.PaletteFinderColors(colorNumber)
	// The secondary palette-position table places a bounded number of embedded
	// colors; beyond that the layout is undefined, so those symbols cannot be a
	// docked secondary. Reject rather than index the table OOB.
	if min(colorNumber, 64)-firstColor > len(tables.SecondaryPalettePosition) {
		return MetadataFailed
	}
	copies := spec.PaletteCopies(colorNumber)
	symbol.Palette = make([]byte, colorNumber*3*copies)

	// 4/8-color symbols carry the first two colors in the alignment patterns; the
	// higher modes embed every color in the position table (ISO Annex G).
	if firstColor > 0 {
		for i := range copies {
			p1, p2 := colorPalettePosInFP(i, matrix.Width, matrix.Height)
			writeColorPalette(matrix, symbol, i, tables.SecondaryPalettePlacementIndexProfile(0, colorNumber, symbol.WireProfile)%colorNumber, p1.X, p1.Y)
			writeColorPalette(matrix, symbol, i, tables.SecondaryPalettePlacementIndexProfile(1, colorNumber, symbol.WireProfile)%colorNumber, p2.X, p2.Y)
		}
	}

	for colorCounter := firstColor; colorCounter < min(colorNumber, 64); colorCounter++ {
		ci := tables.SecondaryPalettePlacementIndexProfile(colorCounter, colorNumber, symbol.WireProfile) % colorNumber
		pos := tables.SecondaryPalettePosition[colorCounter-firstColor]
		// The palette is placed at up to four rotations around the border, one per
		// embedded copy. Skip any rotation landing outside the matrix so a
		// wrongly-sized symbol fails downstream, not by indexing out of range.
		rot := [4]image.Point{
			{pos.X, pos.Y},
			{matrix.Width - 1 - pos.Y, pos.X},
			{matrix.Width - 1 - pos.X, matrix.Height - 1 - pos.Y},
			{pos.Y, matrix.Height - 1 - pos.X},
		}
		for pIndex := range copies {
			pt := rot[pIndex]
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
	return decodeSecondary(matrix, symbol, nil)
}

// DecodeSecondaryTraced is DecodeSecondary with the actual data-module hard
// classifications retained from the same execution.
func DecodeSecondaryTraced(matrix *core.Bitmap, symbol *core.DecodedSymbol, trace *ModuleClassificationTrace) int {
	return decodeSecondary(matrix, symbol, trace)
}

func decodeSecondary(matrix *core.Bitmap, symbol *core.DecodedSymbol, trace *ModuleClassificationTrace) int {
	// Ports decodeSlave in decoder.c.
	if trace != nil {
		*trace = ModuleClassificationTrace{}
	}
	if matrix == nil {
		return core.FatalError
	}
	dataMap := make([]byte, matrix.Width*matrix.Height)
	if readColorPaletteInSecondary(matrix, symbol, dataMap) < 0 {
		return core.FatalError
	}

	colorNumber := 1 << (symbol.Meta.NC + 1)
	copies := spec.PaletteCopies(colorNumber)
	normPalette := make([]float64, colorNumber*4*copies)
	NormalizeColorPalette(symbol, normPalette, colorNumber)
	palThs := make([]float64, 3*spec.ColorPaletteNumber)
	for i := range copies {
		// Note: the reference offsets by i*3 (not colorNumber*3*i) here; kept identical.
		t := PaletteThreshold(symbol.Palette[i*3:], colorNumber)
		palThs[i*3+0], palThs[i*3+1], palThs[i*3+2] = t[0], t[1], t[2]
	}
	if trace != nil {
		return DecodeSymbolTraced(matrix, symbol, dataMap, normPalette, palThs, 1, trace)
	}
	return DecodeSymbol(matrix, symbol, dataMap, normPalette, palThs, 1)
}
