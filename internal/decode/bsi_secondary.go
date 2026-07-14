//go:build jabcode_bsi

package decode

import (
	"image"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/ecc"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/tables"
	"github.com/srlehn/jabcode/internal/wire"
)

const bsiSecondaryMetadataPart1Length = 6

// DecodeBSISecondaryMetadata decodes the canonical cross-edge metadata sample
// needed to determine a BSI secondary symbol's complete geometry.
func DecodeBSISecondaryMetadata(matrix *core.Bitmap, host, secondary *core.DecodedSymbol) int {
	if matrix == nil || host == nil || secondary == nil || matrix.Channels < 3 ||
		matrix.Width != 5 || matrix.Height != 20 || host.WireVariant != wire.BSI {
		return core.Failure
	}
	if host.Meta.NC < 0 || host.Meta.NC > 7 || host.Meta.MaskType < 0 || host.Meta.MaskType > 7 ||
		secondary.HostPosition < 0 || secondary.HostPosition > 3 {
		return MetadataFailed
	}

	secondary.WireVariant = wire.BSI
	secondary.Meta = core.Metadata{
		NC: host.Meta.NC, MaskType: host.Meta.MaskType,
	}
	colorNumber := 1 << (secondary.Meta.NC + 1)
	metadataColors := min(colorNumber, 8)
	physical := make([]byte, metadataColors*3*bsiPhysicalPaletteCopies)
	for colorIndex := range metadataColors {
		pos := tables.SecondaryPalettePosition[colorIndex]
		if !bsiPositionValid(matrix, pos.X, pos.Y) {
			return core.Failure
		}
		off := (pos.Y*matrix.Width + pos.X) * matrix.Channels
		for copyIndex := range bsiPhysicalPaletteCopies {
			dst := (copyIndex*metadataColors + colorIndex) * 3
			copy(physical[dst:dst+3], matrix.Pix[off:off+3])
		}
	}
	palette := expandBSIPalette(physical, metadataColors, matrix.Width, matrix.Height)
	normPalette := make([]float64, metadataColors*4*bsiLogicalPaletteCopies)
	bsiNormalizePalette(palette, normPalette, metadataColors)
	reader := bsiSecondaryMetadataReader{
		matrix: matrix, palette: palette, colorNumber: metadataColors,
		normPalette: normPalette, paletteThresholds: bsiPaletteThresholds(palette, metadataColors),
		x: 0, y: 1,
	}

	part1, ok := reader.read(bsiSecondaryMetadataPart1Length)
	if !ok {
		return core.Failure
	}
	part1Decoded, ok := ecc.DecodeLDPCHardVariant(part1, 2, -1, wire.BSI)
	if !ok || len(part1Decoded) < 3 {
		return MetadataFailed
	}
	ss, se, sf := part1Decoded[0], part1Decoded[1], part1Decoded[2]
	versionLength, positionLength, eclLength := 0, 0, 0
	if ss == 0 {
		secondary.Meta.SideVersion = host.Meta.SideVersion
	} else {
		versionLength = 5
	}
	if se == 0 {
		secondary.Meta.ECL = host.Meta.ECL
	} else if versionLength == 0 {
		eclLength = spec.BSIErrorCorrectionBitLength(secondary.Meta.SideVersion)
	}
	if sf != 0 {
		positionLength = 3
	}

	part2Length := versionLength + positionLength
	if part2Length != 0 {
		part2, ok := reader.read(part2Length * 2)
		if !ok {
			return core.Failure
		}
		part2Decoded, ok := ecc.DecodeLDPCHardVariant(part2, 2, -1, wire.BSI)
		if !ok || len(part2Decoded) < part2Length {
			return MetadataFailed
		}
		bitIndex := 0
		if versionLength != 0 {
			sideVersion := bsiBitsValue(part2Decoded[:versionLength]) + 1
			secondary.Meta.SideVersion = host.Meta.SideVersion
			if secondary.HostPosition == 2 || secondary.HostPosition == 3 {
				secondary.Meta.SideVersion.X = sideVersion
			} else {
				secondary.Meta.SideVersion.Y = sideVersion
			}
			bitIndex += versionLength
			if se != 0 {
				eclLength = spec.BSIErrorCorrectionBitLength(secondary.Meta.SideVersion)
			}
		}
		if positionLength != 0 {
			for position := range 4 {
				if position == secondary.HostPosition {
					continue
				}
				secondary.Meta.DockedPosition |= int(part2Decoded[bitIndex]&1) << uint(3-position)
				bitIndex++
			}
		}
	}

	if eclLength != 0 {
		part3, ok := reader.read(eclLength * 2)
		if !ok {
			return core.Failure
		}
		part3Decoded, ok := ecc.DecodeLDPCHardVariant(part3, 2, -1, wire.BSI)
		if !ok || len(part3Decoded) < eclLength {
			return MetadataFailed
		}
		half := eclLength / 2
		secondary.Meta.ECL = image.Pt(
			bsiBitsValue(part3Decoded[:half])+3,
			bsiBitsValue(part3Decoded[half:eclLength])+4,
		)
	}

	if secondary.Meta.SideVersion.X < 1 || secondary.Meta.SideVersion.X > 32 ||
		secondary.Meta.SideVersion.Y < 1 || secondary.Meta.SideVersion.Y > 32 ||
		secondary.Meta.ECL.X < 3 || secondary.Meta.ECL.X >= secondary.Meta.ECL.Y || secondary.Meta.ECL.Y > 35 {
		return MetadataFailed
	}
	secondary.SideSize = image.Pt(
		spec.VersionToSize(secondary.Meta.SideVersion.X),
		spec.VersionToSize(secondary.Meta.SideVersion.Y),
	)
	secondary.MetadataModules = reader.moduleCount
	return core.Success
}

type bsiSecondaryMetadataReader struct {
	matrix            *core.Bitmap
	palette           []byte
	colorNumber       int
	normPalette       []float64
	paletteThresholds []float64
	x, y              int
	moduleCount       int
	pending           []byte
}

func (r *bsiSecondaryMetadataReader) read(length int) ([]byte, bool) {
	bitsPerModule := spec.Log2Int(r.colorNumber)
	for len(r.pending) < length {
		if !bsiPositionValid(r.matrix, r.x, r.y) {
			return nil, false
		}
		value := decodeBSIModuleHD(r.matrix, r.palette, r.colorNumber, r.normPalette, r.paletteThresholds, r.x, r.y)
		for i := range bitsPerModule {
			r.pending = append(r.pending, (value>>uint(bitsPerModule-1-i))&1)
		}
		r.moduleCount++
		spec.NextMetadataModuleInSecondary(r.moduleCount, &r.x, &r.y)
	}
	out := append([]byte(nil), r.pending[:length]...)
	r.pending = r.pending[length:]
	return out, true
}

// DecodeBSISecondary decodes a fully sampled BSI secondary symbol whose edge
// metadata has already established its size and error-correction parameters.
func DecodeBSISecondary(matrix *core.Bitmap, symbol *core.DecodedSymbol) int {
	return decodeBSISecondary(matrix, symbol, nil)
}

// DecodeBSISecondaryTraced is DecodeBSISecondary with authoritative hard
// module classifications retained from the same execution.
func DecodeBSISecondaryTraced(matrix *core.Bitmap, symbol *core.DecodedSymbol, trace *ModuleClassificationTrace) int {
	return decodeBSISecondary(matrix, symbol, trace)
}

func decodeBSISecondary(matrix *core.Bitmap, symbol *core.DecodedSymbol, trace *ModuleClassificationTrace) int {
	if matrix == nil || symbol == nil || matrix.Channels < 3 ||
		symbol.SideSize != image.Pt(matrix.Width, matrix.Height) || symbol.WireVariant != wire.BSI {
		return core.Failure
	}
	dataMap := make([]byte, matrix.Width*matrix.Height)
	if !readBSISecondaryPalette(matrix, symbol, dataMap) || !markBSISecondaryMetadata(symbol, dataMap) {
		return core.Failure
	}
	return decodeBSISymbol(matrix, symbol, dataMap, 1, trace)
}

func readBSISecondaryPalette(matrix *core.Bitmap, symbol *core.DecodedSymbol, dataMap []byte) bool {
	colorNumber := 1 << (symbol.Meta.NC + 1)
	available := min(colorNumber, 64)
	physical := make([]byte, colorNumber*3*bsiPhysicalPaletteCopies)
	for colorIndex := range available {
		positionIndex := colorIndex
		if available > 8 && colorIndex >= available/2 {
			positionIndex -= available / 2
		}
		pos := tables.SecondaryPalettePosition[positionIndex]
		x, y := pos.X, pos.Y
		if symbol.HostPosition == 2 || symbol.HostPosition == 3 {
			if available > 8 && colorIndex >= available/2 {
				if matrix.Width > matrix.Height {
					x, y = pos.Y, matrix.Height-1-pos.X
				} else {
					x, y = matrix.Width-1-pos.Y, pos.X
				}
			}
		} else if available <= 8 || colorIndex < available/2 {
			if matrix.Width > matrix.Height {
				x, y = pos.Y, matrix.Height-1-pos.X
			} else {
				x, y = matrix.Width-1-pos.Y, pos.X
			}
		}
		if !readBSIPaletteColor(matrix, physical, colorNumber, 0, colorIndex, x, y, dataMap) ||
			!readBSIPaletteColor(matrix, physical, colorNumber, 1, colorIndex, matrix.Width-1-x, matrix.Height-1-y, dataMap) {
			return false
		}
	}
	deinterleaveBSIPalette(physical, colorNumber)
	if colorNumber > 64 {
		interpolatePalette(physical, colorNumber)
	}
	symbol.Palette = expandBSIPalette(physical, colorNumber, matrix.Width, matrix.Height)
	return true
}

func markBSISecondaryMetadata(symbol *core.DecodedSymbol, dataMap []byte) bool {
	width, height := symbol.SideSize.X, symbol.SideSize.Y
	x, y := 0, 1
	for count := 0; count < symbol.MetadataModules; count++ {
		xx, yy := x, y
		switch symbol.HostPosition {
		case 3:
			xx, yy = width-1-x, height-1-y
		case 0:
			xx, yy = width-1-y, x
		case 1:
			xx, yy = y, height-1-x
		}
		if xx < 0 || yy < 0 || xx >= width || yy >= height {
			return false
		}
		dataMap[yy*width+xx] = 1
		spec.NextMetadataModuleInSecondary(count+1, &x, &y)
	}
	return true
}
