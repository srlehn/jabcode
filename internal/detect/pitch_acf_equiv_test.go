package detect

import (
	"math/rand/v2"
	"testing"
)

// acfAccumulateInline is the superseded form that subtracted the mean inside
// the lag sweep, kept purely as the comparison target. `pitch_acf` is one of
// the kernels compiled against softfloat64 so the device reproduces host
// float64 exactly, so this equivalence must be bit-for-bit, not approximate.
func acfAccumulateInline(lines [][]float64, maxLag int) []float64 {
	acf := make([]float64, maxLag+1)
	for _, s := range lines {
		n := len(s)
		if n < 2 {
			continue
		}
		var mean float64
		for _, v := range s {
			mean += v
		}
		mean /= float64(n)
		inv := 1 / float64(n)
		hi := min(maxLag, n-1)
		for lag := 0; lag <= hi; lag++ {
			var sum float64
			for x := 0; x+lag < n; x++ {
				sum += (s[x] - mean) * (s[x+lag] - mean)
			}
			acf[lag] += sum * inv
		}
	}
	return acf
}

// TestACFAccumulateMatchesInline pins the centred sweep bit-for-bit against the
// form that subtracted the mean per product. Lines shorter than the lag bound,
// and lines of differing length in one call, exercise the per-line clamp and
// the reuse of the scratch buffer between lines.
func TestACFAccumulateMatchesInline(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 17))
	for trial := range 300 {
		lineCount := 1 + int(rng.UintN(5))
		lines := make([][]float64, lineCount)
		for i := range lines {
			n := int(rng.UintN(40))
			line := make([]float64, n)
			for x := range line {
				switch trial % 3 {
				case 0:
					line[x] = rng.NormFloat64()
				case 1:
					line[x] = float64(rng.UintN(256))
				default:
					line[x] = rng.NormFloat64() * 1e6
				}
			}
			lines[i] = line
		}
		maxLag := 1 + int(rng.UintN(20))
		got := acfAccumulate(lines, maxLag)
		want := acfAccumulateInline(lines, maxLag)
		if len(got) != len(want) {
			t.Fatalf("trial %d: length %d, want %d", trial, len(got), len(want))
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("trial %d lag %d: got %v, want %v (bit-for-bit equality required)",
					trial, i, got[i], want[i])
			}
		}
	}
}
