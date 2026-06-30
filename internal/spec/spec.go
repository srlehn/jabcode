// Package spec holds JAB Code symbol geometry, metadata layout, masking and
// finder core-color constants shared by the encoder and decoder.
package spec

import "math/bits"

// VersionToSize returns the module side length of a symbol of the given side
// version.
func VersionToSize(v int) int { return v*4 + 17 }

// SizeToVersion returns the side version of a symbol with the given module side
// length.
func SizeToVersion(s int) int { return (s - 17) / 4 }

// Log2Int returns log2(n) for a power-of-two n (the bits-per-module count).
func Log2Int(n int) int { return bits.Len(uint(n)) - 1 }

// DefaultECCLevel is the error-correction level of a default-mode primary symbol.
const DefaultECCLevel = 3

// DefaultMaskingReference is the fixed mask pattern used in default mode.
const DefaultMaskingReference = 7

// Encoding modes shared by the encoder's mode optimizer and the decoder; they
// index the per-character cost and decoding tables (encoder.h, decoder.h).
const (
	ModeUpper = iota
	ModeLower
	ModeNumeric
	ModePunct
	ModeMixed
	ModeAlphanumeric
	ModeByte
)

// ECCWeights maps an error-correction level to its LDPC (wc, wr) weights.
var ECCWeights = [11][2]int{ // ecclevel2wcwr (encoder.h)
	{4, 9}, {3, 8}, {3, 7}, {4, 9}, {3, 6}, {4, 7}, {4, 6}, {3, 4}, {4, 5}, {5, 6}, {6, 7},
}

// Symbol layout constants (jabcode.h, decoder.h).
const (
	ColorPaletteNumber = 4
	DistanceToBorder   = 4

	PrimaryMetadataX                 = 6
	PrimaryMetadataY                 = 1
	PrimaryMetadataPart1Length       = 6
	PrimaryMetadataPart2Length       = 38
	PrimaryMetadataPart1ModuleNumber = 4
)

// Finder-pattern core color indices in the 8-color default palette, used as
// fixed scalars during detection and masking (the Nc=2 column of
// tables.FPCoreColor).
const (
	FP0CoreColor = 0 // black
	FP1CoreColor = 0 // black
	FP2CoreColor = 6 // yellow
	FP3CoreColor = 3 // cyan
)

// NextMetadataModuleInPrimary advances (x, y) to the next metadata/palette
// module position in a primary symbol. count is the running module index.
func NextMetadataModuleInPrimary(height, width, count int, x, y *int) {
	// Ports the primary-symbol metadata module walk in decoder.c.
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

// MaskValue returns the mask offset for module (x, y) under the given pattern.
func MaskValue(maskType, x, y int) int {
	// Ports maskSymbols/demaskSymbol in mask.c.
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
