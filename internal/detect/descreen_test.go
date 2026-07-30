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
	current := []float64{8, 4, 6}
	bsi := []float64{20, 18, 22}
	if got := descreenSeedModuleScale(current, bsi); got != 6 {
		t.Fatalf("combined scale = %v, want current-family median 6", got)
	}
	if got := descreenSeedModuleScale(nil, bsi); got != 20 {
		t.Fatalf("BSI-only scale = %v, want 20", got)
	}
	if !reflect.DeepEqual(current, []float64{8, 4, 6}) || !reflect.DeepEqual(bsi, []float64{20, 18, 22}) {
		t.Fatalf("scale measurement reordered inputs: current=%v BSI=%v", current, bsi)
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
