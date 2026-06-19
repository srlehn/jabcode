package jabcode

// defaultPalette is the 8-color default module palette as RGB triples
// (jab_default_palette in encoder.h): black, blue, green, cyan, red, magenta,
// yellow, white.
var defaultPalette = [8 * 3]byte{
	0, 0, 0,
	0, 0, 255,
	0, 255, 0,
	0, 255, 255,
	255, 0, 0,
	255, 0, 255,
	255, 255, 0,
	255, 255, 255,
}

// setDefaultPalette returns the default module color palette as RGB triples for
// the given color count (setDefaultPalette in encoder.c).
func setDefaultPalette(colorNumber int) []byte {
	switch colorNumber {
	case 4:
		// Two-bit palette: black 00, magenta 01, yellow 10, cyan 11 — picked from
		// the 8-color palette at indices below.
		p := make([]byte, 4*3)
		for dst, src := range [4]int{0, 5, 6, 3} {
			copy(p[dst*3:], defaultPalette[src*3:src*3+3])
		}
		return p
	case 8:
		p := make([]byte, 8*3)
		copy(p, defaultPalette[:])
		return p
	default:
		return genColorPalette(colorNumber)
	}
}

// genColorPalette generates a palette for color counts above 8 by sampling the
// RGB cube on a per-channel grid (genColorPalette in encoder.c). It returns nil
// for unsupported counts.
func genColorPalette(colorNumber int) []byte {
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
