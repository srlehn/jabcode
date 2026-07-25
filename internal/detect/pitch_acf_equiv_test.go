package detect

import (
	"math"
	"math/rand/v2"
	"testing"
)

// acfAccumulateSerial is the superseded single-accumulator sweep. The current
// form splits that accumulator four ways, which is a deliberate reassociation
// and therefore does not reproduce this bit for bit. It is kept as a reference
// for the two properties that must still hold: the split computes the same
// quantity, and it does so within the error a reassociation can introduce.
//
// Exact agreement is contracted between acfAccumulate and pitch_acf.wgsl, not
// between the split and this serial form. That pairing is pinned by the
// Float64bits comparison in the resident-ACF GPU test.
func acfAccumulateSerial(lines [][]float64, maxLag int) []float64 {
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

// TestACFAccumulateTracksSerial checks the split accumulator against the serial
// sweep it reassociates. The two are not required to agree bit for bit - that
// is the point of the change - but a reassociation may only move the result by
// rounding, so a relative disagreement beyond a few ULP of the accumulated
// magnitude would mean the split computes something else.
func TestACFAccumulateTracksSerial(t *testing.T) {
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
		want := acfAccumulateSerial(lines, maxLag)
		if len(got) != len(want) {
			t.Fatalf("trial %d: length %d, want %d", trial, len(got), len(want))
		}
		for i := range got {
			scale := max(math.Abs(want[i]), 1)
			if diff := math.Abs(got[i] - want[i]); diff > 1e-9*scale {
				t.Fatalf("trial %d lag %d: got %v, want %v (diff %v exceeds rounding)",
					trial, i, got[i], want[i], diff)
			}
		}
	}
}

// TestACFAccumulateTailBoundary exercises the lengths where the four-wide body
// hands over to the scalar tail, which is where a split accumulator is most
// likely to drop or double-count a term. Against the serial sweep the counts
// must line up exactly even though the rounding need not.
func TestACFAccumulateTailBoundary(t *testing.T) {
	for n := 2; n <= 20; n++ {
		line := make([]float64, n)
		for x := range line {
			line[x] = float64(x%3) - 1
		}
		lines := [][]float64{line}
		for maxLag := 1; maxLag < n; maxLag++ {
			got := acfAccumulate(lines, maxLag)
			want := acfAccumulateSerial(lines, maxLag)
			for i := range got {
				if diff := math.Abs(got[i] - want[i]); diff > 1e-9*max(math.Abs(want[i]), 1) {
					t.Fatalf("n %d maxLag %d lag %d: got %v, want %v", n, maxLag, i, got[i], want[i])
				}
			}
		}
	}
}
