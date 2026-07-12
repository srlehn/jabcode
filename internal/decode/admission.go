package decode

import (
	"math"

	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/tables"
)

// Admission thresholds for observations whose metadata did not establish the
// layout (default-mode fallback or failed metadata syndromes), measured on
// the full capture-fixture set plus real video frames: every false
// observation (phantom quads, misgridded samples, wrong-mode reads) fails at
// least one of the two checks with a factor-two margin, while true grids of
// the modes that can appear default (8 colours) score fixed agreement of at
// least 0.86 at a palette ratio of at most 1.4. Both signals are
// dimensionless, so the thresholds carry no pixel-scale assumptions.
const (
	admitMinFixedAgreement  = 0.7 // fixed-pattern agreement floor
	admitMaxPaletteRatio    = 3.0 // palette disagreement/separation ceiling
	admitMinCheckedPatterns = 20  // minimum classifiable fixed modules
)

// AdmitPayloadCorrection reports whether the observation is plausibly a
// correctly sampled symbol worth spending payload correction on. A
// non-default symbol whose metadata parts both satisfied their LDPC parity
// checks is admitted outright - across the measured captures that condition
// occurred only on true grids. Anything else (default-mode symbols decode no
// explicit metadata; garbage samples fall back to default metadata) must
// look like a symbol: the format-fixed modules classify to their expected
// colours and the embedded palette copies agree with each other.
func (obs *PrimaryObservation) AdmitPayloadCorrection() bool {
	if !obs.Symbol.Meta.DefaultMode && obs.PartISyndromeOK && obs.PartIISyndromeOK {
		return true
	}
	agree, checked := obs.FixedPatternAgreement()
	if checked < admitMinCheckedPatterns || float64(agree) < admitMinFixedAgreement*float64(checked) {
		return false
	}
	disagreement, separation := obs.PaletteCoherence()
	return disagreement <= admitMaxPaletteRatio*separation
}

// FixedPatternAgreement classifies the sampled modules whose colours the
// format fixes - the four finder-pattern cores and every interior
// alignment-pattern core and periphery - against their expected palette
// indices for the observed colour mode. agree/checked is the admission
// signal: a correctly gridded sample classifies almost all fixed modules
// correctly, a phantom or misgridded sample scores near chance (1/colours).
// The check spends a few dozen classifications and no error correction.
func (obs *PrimaryObservation) FixedPatternAgreement() (agree, checked int) {
	m := obs.Matrix
	nc := obs.Symbol.Meta.NC
	colorNumber := 1 << (nc + 1)

	check := func(x, y int, want byte) {
		if x < 0 || y < 0 || x >= m.Width || y >= m.Height {
			return
		}
		checked++
		if DecodeModuleHD(m, obs.Symbol.Palette, colorNumber, obs.normPalette, obs.palThs, x, y) == want {
			agree++
		}
	}

	// The four finder patterns, all three concentric layers (the layer
	// colours alternate between the core pairs, mirroring the placement).
	const d = spec.DistanceToBorder
	w, h := m.Width, m.Height
	for k := range 3 {
		fp0 := byte(tables.FPCoreColor[0][nc])
		fp1 := byte(tables.FPCoreColor[1][nc])
		fp2 := byte(tables.FPCoreColor[2][nc])
		fp3 := byte(tables.FPCoreColor[3][nc])
		if k%2 == 1 {
			fp0, fp1, fp2, fp3 = fp3, fp2, fp1, fp0
		}
		for i := 0; i <= k; i++ {
			for j := 0; j <= k; j++ {
				if i != k && j != k {
					continue
				}
				check(d-j-1, d-(i+1), fp0)
				check(w-(d-1)-j-1, d-(i+1), fp1)
				check(w-(d-1)-j-1, h-d+i, fp2)
				check(d-j-1, h-d+i, fp3)
				if k == 0 {
					// The two arcs coincide at the core; count it once.
					continue
				}
				check(d+j-1, d+(i-1), fp0)
				check(w-(d-1)+j-1, d+(i-1), fp1)
				check(w-(d-1)+j-1, h-d-i, fp2)
				check(d+j-1, h-d-i, fp3)
			}
		}
	}

	// Interior alignment patterns: core plus the six peripheral modules, with
	// the same left/right alternation the placement walks.
	vx := spec.SizeToVersion(w) - 1
	vy := spec.SizeToVersion(h) - 1
	if vx < 0 || vx >= len(tables.APPos) || vy < 0 || vy >= len(tables.APPos) {
		return agree, checked
	}
	apCore := byte(tables.APXCoreColor[nc])
	apPeri := byte(tables.APNCoreColor[nc])
	for x := 0; x < tables.APNum[vx]; x++ {
		left := x%2 == 0
		for y := 0; y < tables.APNum[vy]; y++ {
			corner := (x == 0 || x == tables.APNum[vx]-1) && (y == 0 || y == tables.APNum[vy]-1)
			if !corner {
				xo := tables.APPos[vx][x] - 1
				yo := tables.APPos[vy][y] - 1
				check(xo, yo, apCore)
				dx := 1
				if left {
					dx = -1
				}
				check(xo+dx, yo-1, apPeri)
				check(xo, yo-1, apPeri)
				check(xo-1, yo, apPeri)
				check(xo+1, yo, apPeri)
				check(xo, yo+1, apPeri)
				check(xo-dx, yo+1, apPeri)
			}
			left = !left
		}
	}
	return agree, checked
}

// PaletteCoherence measures the embedded palette's internal consistency.
// disagreement is the mean RGB distance between corresponding colours across
// the embedded copies; separation is the minimum pairwise RGB distance among
// the per-colour mean values. A well-sampled symbol reads coherent copies
// (low disagreement) of a separable palette (separation well above zero); a
// misaligned or phantom sample reads incoherent copies or a collapsed
// palette. Both values are in raw RGB units, so callers compare them against
// each other, not against fixed constants.
func (obs *PrimaryObservation) PaletteCoherence() (disagreement, separation float64) {
	colorNumber := 1 << (obs.Symbol.Meta.NC + 1)
	copies := spec.PaletteCopies(colorNumber)
	pal := obs.Symbol.Palette
	if copies < 1 || len(pal) < colorNumber*3*copies {
		return 0, 0
	}

	dist := func(a, b []byte) float64 {
		dr := float64(a[0]) - float64(b[0])
		dg := float64(a[1]) - float64(b[1])
		db := float64(a[2]) - float64(b[2])
		return math.Sqrt(dr*dr + dg*dg + db*db)
	}

	mean := make([]byte, colorNumber*3)
	for c := range colorNumber {
		var sum [3]int
		for p := range copies {
			off := (p*colorNumber + c) * 3
			for k := range 3 {
				sum[k] += int(pal[off+k])
			}
		}
		for k := range 3 {
			mean[c*3+k] = byte(sum[k] / copies)
		}
	}

	if copies > 1 {
		pairs := 0
		for c := range colorNumber {
			for p := range copies {
				for q := p + 1; q < copies; q++ {
					disagreement += dist(pal[(p*colorNumber+c)*3:], pal[(q*colorNumber+c)*3:])
					pairs++
				}
			}
		}
		disagreement /= float64(pairs)
	}

	separation = math.Inf(1)
	for a := range colorNumber {
		for b := a + 1; b < colorNumber; b++ {
			if d := dist(mean[a*3:], mean[b*3:]); d < separation {
				separation = d
			}
		}
	}
	return disagreement, separation
}
