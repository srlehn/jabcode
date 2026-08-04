package detect

import (
	"math"
	"reflect"
	"testing"
)

func TestDescreenScheduleStaysBelowModuleScale(t *testing.T) {
	tests := []struct {
		name        string
		px, py      int
		moduleScale float64
		want        [][2]int
	}{
		{name: "false scene peak", py: 56, moduleScale: 7.071},
		{name: "first pass only", px: 4, moduleScale: 7, want: [][2]int{{2, 0}}},
		{name: "both passes", px: 3, moduleScale: 7, want: [][2]int{{1, 0}, {2, 0}}},
		{name: "axes independent", px: 3, py: 56, moduleScale: 7, want: [][2]int{{1, 0}, {2, 0}}},
		{name: "no module evidence", px: 3},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := descreenSchedule(test.px, test.py, test.moduleScale); !reflect.DeepEqual(got, test.want) {
				t.Fatalf("descreenSchedule(%d, %d, %.3f) = %v, want %v",
					test.px, test.py, test.moduleScale, got, test.want)
			}
		})
	}
}

func TestDescreenSeedScaleKeepsCurrentFamilyIndependent(t *testing.T) {
	seeds := func(values ...float64) *moduleSeeds {
		acc := &moduleSeeds{}
		for _, value := range values {
			acc.add(value)
		}
		return acc
	}
	// The accumulator answers with its bucket midpoint, so an exact median of 6
	// reports as the centre of the bucket holding it.
	const bucket = 1.0 / (2 * moduleSeedsPerPixel)
	current := seeds(8, 4, 6)
	bsi := seeds(20, 18, 22)
	if got := descreenSeedModuleScale(current, bsi); got != 6+bucket {
		t.Fatalf("combined scale = %v, want current-family median %v", got, 6+bucket)
	}
	if got := descreenSeedModuleScale(&moduleSeeds{}, bsi); got != 20+bucket {
		t.Fatalf("BSI-only scale = %v, want %v", got, 20+bucket)
	}
	if current.len() != 3 || bsi.len() != 3 {
		t.Fatalf("scale measurement consumed inputs: current=%d BSI=%d", current.len(), bsi.len())
	}
}

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
