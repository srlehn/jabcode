// Package palette holds the JAB Code module color palettes shared by the encoder
// and decoder.
package palette

import "github.com/srlehn/jabcode/internal/wire"

// Default is the 8-color default module palette as RGB triples: black, blue,
// green, cyan, red, magenta, yellow, white.
var Default = [8 * 3]byte{ // jab_default_palette (encoder.h)
	0, 0, 0,
	0, 0, 255,
	0, 255, 0,
	0, 255, 255,
	255, 0, 0,
	255, 0, 255,
	255, 255, 0,
	255, 255, 255,
}

// SetDefault returns the default module color palette as RGB triples for the
// given color count.
func SetDefault(colorNumber int) []byte {
	return SetDefaultVariant(colorNumber, wire.ISO23634)
}

// SetDefaultVariant returns the default module color palette for the selected
// wire-format variant.
func SetDefaultVariant(colorNumber int, variant wire.Variant) []byte {
	// Ports setDefaultPalette in encoder.c.
	switch colorNumber {
	case 4:
		if variant.UsesISO23634Base() {
			p := make([]byte, 4*3)
			for dst, src := range [4]int{0, 3, 5, 6} {
				copy(p[dst*3:], Default[src*3:src*3+3])
			}
			return p
		}
		if variant == wire.BSI {
			p := make([]byte, 4*3)
			for dst, src := range [4]int{1, 2, 5, 6} {
				copy(p[dst*3:], Default[src*3:src*3+3])
			}
			return p
		}
		// Two-bit palette: black 00, magenta 01, yellow 10, cyan 11, picked from
		// the 8-color palette at the indices below. ISO/IEC 23634 Table 4 orders
		// them black, cyan, magenta, yellow instead; this order is the reference
		// C library's and is kept for interop. Since the palette is embedded in
		// the symbol and read back at decode, indices still round-trip; only the
		// physical colors of a 4-color symbol differ from a strict-ISO one.
		p := make([]byte, 4*3)
		for dst, src := range [4]int{0, 5, 6, 3} {
			copy(p[dst*3:], Default[src*3:src*3+3])
		}
		return p
	case 8:
		p := make([]byte, 8*3)
		copy(p, Default[:])
		return p
	default:
		return genColorPalette(colorNumber)
	}
}

// genColorPalette generates a palette for color counts above 8 by sampling the
// RGB cube on a per-channel grid. It returns nil for unsupported counts.
func genColorPalette(colorNumber int) []byte {
	// Ports genColorPalette in encoder.c.
	var vr, vg, vb int // grid steps per channel
	switch colorNumber {
	case 16:
		vr, vg, vb = 4, 2, 2
	case 32:
		vr, vg, vb = 4, 4, 2
	case 64:
		vr, vg, vb = 4, 4, 4
	case 128:
		vr, vg, vb = 8, 4, 4
	case 256:
		vr, vg, vb = 8, 8, 4
	default:
		return nil
	}

	// channelStep mirrors the reference's interval, including its special-case of
	// 85 for a 4-level channel (which would otherwise be 256/3 ≈ 85.33).
	channelStep := func(v int) float32 {
		if v-1 == 3 {
			return 85
		}
		return 256 / float32(v-1)
	}
	dr, dg, db := channelStep(vr), channelStep(vg), channelStep(vb)
	level := func(d float32, i int) byte { return byte(min(int(d*float32(i)), 255)) }

	p := make([]byte, 0, colorNumber*3)
	for i := 0; i < vr; i++ {
		r := level(dr, i)
		for j := 0; j < vg; j++ {
			g := level(dg, j)
			for k := 0; k < vb; k++ {
				p = append(p, r, g, level(db, k))
			}
		}
	}
	return p
}
