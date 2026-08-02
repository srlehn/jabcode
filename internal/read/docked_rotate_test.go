package read

import (
	"image"
	"slices"
	"testing"

	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/testutil"
	"github.com/srlehn/jabcode/internal/wire"
)

// A docked secondary carries no finder patterns. It is located from the host's
// docking edge and its own corner alignment patterns, so the directional finder
// scan does nothing for it and it needs the alignment-pattern locator to work
// in the symbol's basis instead of along image rows.
//
// The route assertion is the point. Full-frame rotation would also decode these
// angles - it turns the frame until the patterns fall into the axis-aligned
// walk's roughly plus or minus 20 degree band, which is what the ladder was
// doing for them - so a payload-only test passes with the basis work reverted.
// Requiring the first whole-frame route is what pins that no pixel moved.
func TestDockedSecondaryDecodesAtArbitraryAngles(t *testing.T) {
	payload := []byte("two docked symbols read at an angle")
	v4 := image.Pt(4, 4)
	img, err := encode.Run(encode.Config{
		Colors: 8, ModuleSize: 12, SymbolNumber: 2,
		SymbolPositions: []int{0, 2}, SymbolVersions: []image.Point{v4, v4}, SymbolECCLevels: []int{0, 0},
	}, payload)
	if err != nil {
		t.Fatal(err)
	}
	want := string(isoPayload(payload))
	var settled []float64
	for _, deg := range []float64{0, 10, 20, 25, 30, 40, 45, 50, 60, 70, 80} {
		got, report, err := DecodeWithRouteCapabilities(testutil.RotateImage(img, deg), wire.ISO23634.Mask())
		if err != nil {
			t.Errorf("deg=%3.0f: %v", deg, err)
			continue
		}
		if string(got) != want {
			t.Errorf("deg=%3.0f: payload = %q, want %q", deg, got, want)
		}
		if report.Kind != "frame" || report.Attempts != 1 {
			t.Errorf("deg=%3.0f: report = %v; want a single whole-frame route", deg, report)
		}
		// Kind names the ladder rung and nothing else. Every whole-frame pass
		// sweeps the probe directions when its row walk does not settle, so a
		// rotated capture reads `frame` exactly like an unrotated one; Deg is
		// the only field that separates them, and reading Kind as if it meant
		// "no directional work ran" has already cost one investigation.
		//
		// Which direction settles a given rotation is not fixed: a finder is
		// square, so a modest turn still leaves a row crossing its layers in
		// the right proportions and the row walk keeps the read. So the
		// per-angle assertion is only that Deg names a direction the search
		// actually probes, with the collected set checked below.
		if !slices.Contains(ProbeDegrees(), report.Deg) {
			t.Errorf("deg=%3.0f: report.Deg = %v, not a probed direction", deg, report.Deg)
		}
		settled = append(settled, report.Deg)
	}
	// A field that is always zero would pass every check above. Some of these
	// rotations must reach the directional sweep.
	if !slices.ContainsFunc(settled, func(d float64) bool { return d != 0 }) {
		t.Errorf("every rotation reported the row walk: %v", settled)
	}
}
