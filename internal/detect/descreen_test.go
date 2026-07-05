package detect

import (
	"math"
	"testing"
)

// TestLinearToSRGBMatchesFormula sweeps the boundary-table encode against the
// closed form: a dense uniform grid over [0,1], every table boundary and its
// float neighbours, and the extremes.
func TestLinearToSRGBMatchesFormula(t *testing.T) {
	check := func(c float64) {
		t.Helper()
		if got, want := linearToSRGB(c), linearToSRGBFormula(c); got != want {
			t.Fatalf("linearToSRGB(%g) = %d, formula gives %d", c, got, want)
		}
	}
	const steps = 1 << 21
	for i := 0; i <= steps; i++ {
		check(float64(i) / steps)
	}
	for _, b := range srgbBounds() {
		check(math.Nextafter(b, 0))
		check(b)
		check(math.Nextafter(b, 1))
	}
	check(0)
	check(1)
	check(math.Nextafter(1, 2))
}
