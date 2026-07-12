package decode

import (
	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/ecc"
	"github.com/srlehn/jabcode/internal/spec"
)

// Per-colour candidate costs and signed bit evidence: the retained
// observation form for cross-frame accumulation. A cost is an uncalibrated
// squared colour distance (smaller = more plausible), kept per candidate so
// no information is lost before frames are combined; signed bit
// log-likelihood ratios are derived from costs only once the colour-to-bit
// mapping is trusted.
//
// The metric is the absolute-RGB distance against the embedded (captured)
// palette for every mode - the same distance the higher-mode classifier
// ranks by, and deliberately NOT the normalized chroma the 8-colour hard
// classifier uses: normalized chroma saturates for dark pixels (any nonzero
// channel reads exactly the hue's corner), so a module halfway between black
// and a hue costs confidently as the hue with no luminance weighting able to
// fix it - measured, and exactly the confident-wrong evidence accumulation
// must not amplify. Absolute distance keeps that midpoint honestly
// ambiguous. Illumination differences between frames are the accumulator's
// job (per-frame weights and photometric compatibility), not this metric's.

// ModuleCosts appends module (x,y)'s per-colour candidate costs to dst, one
// per palette colour, in the classifiers' candidate order.
func (obs *PrimaryObservation) ModuleCosts(x, y int, dst []float64) []float64 {
	m := obs.Matrix
	colorNumber := 1 << (obs.Symbol.Meta.NC + 1)
	pIndex := nearestPalette(m, x, y)
	off := m.Offset(x, y)
	rgb := [3]byte{m.Pix[off], m.Pix[off+1], m.Pix[off+2]}

	pal := obs.Symbol.Palette
	base := colorNumber * 3 * (pIndex % spec.PaletteCopies(colorNumber))
	for i := range colorNumber {
		dr := float64(rgb[0]) - float64(pal[base+i*3+0])
		dg := float64(rgb[1]) - float64(pal[base+i*3+1])
		db := float64(rgb[2]) - float64(pal[base+i*3+2])
		dst = append(dst, dr*dr+dg*dg+db*db)
	}
	return dst
}

// BitEvidence derives a snapshot's signed per-bit evidence in the decoder's
// gross-codeword coordinates: data modules column-major, bits most
// significant first, demasked (a set mask bit flips the sign - the same
// confidence votes for the flipped value), truncated to whole code blocks
// and deinterleaved - exactly the space the soft decoder consumes, so
// evidence from compatible frames aligns bit for bit. Only the accumulating
// colour scope derives evidence (up to eight colours); higher modes return
// nil until a measured extension opens them.
func (s *ObservationSnapshot) BitEvidence() []float64 {
	colorNumber := 1 << (s.Meta.NC + 1)
	if s.Meta.NC < 0 || colorNumber > 8 || len(s.DataMap) != s.Side.X*s.Side.Y {
		return nil
	}
	view := &core.Bitmap{Width: s.Side.X, Height: s.Side.Y, Channels: s.Channels, Pix: s.Modules}
	obs := &PrimaryObservation{Matrix: view, Symbol: &core.DecodedSymbol{Meta: s.Meta, Palette: s.Palette}}

	bpm := spec.Log2Int(colorNumber)
	llrs := make([]float64, 0, s.Side.X*s.Side.Y*bpm)
	var costs []float64
	for x := 0; x < s.Side.X; x++ {
		for y := 0; y < s.Side.Y; y++ {
			if s.DataMap[y*s.Side.X+x] != 0 {
				continue
			}
			costs = obs.ModuleCosts(x, y, costs[:0])
			bits := BitLLRs(costs, llrs)
			mask := spec.MaskValue(s.Meta.MaskType, x, y) % colorNumber
			for p := len(llrs); p < len(bits); p++ {
				if (mask>>(uint(bpm-1-(p-len(llrs)))))&1 == 1 {
					bits[p] = -bits[p]
				}
			}
			llrs = bits
		}
	}

	wr := s.Meta.ECL.Y
	if wr <= 0 {
		return nil
	}
	llrs = llrs[:(len(llrs)/wr)*wr]
	ecc.DeinterleaveFloat(llrs)
	return llrs
}

// BitLLRs converts one module's candidate costs to signed max-log bit
// evidence, most significant bit first, appended to dst. The tested sign
// convention: an LLR is the minimum cost among candidates whose index has
// the bit SET minus the minimum among those with it CLEAR, so POSITIVE
// evidence favors bit zero and agreeing observations add.
func BitLLRs(costs []float64, dst []float64) []float64 {
	bpm := spec.Log2Int(len(costs))
	for p := range bpm {
		shift := uint(bpm - 1 - p)
		minSet, minClear := costs[0], costs[0]
		haveSet, haveClear := false, false
		for i, c := range costs {
			if (i>>shift)&1 == 1 {
				if !haveSet || c < minSet {
					minSet, haveSet = c, true
				}
			} else if !haveClear || c < minClear {
				minClear, haveClear = c, true
			}
		}
		dst = append(dst, minSet-minClear)
	}
	return dst
}
