package read

import (
	"image"
	"testing"

	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/encode"
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
// Requiring the first upright route is what pins that no pixel moved.
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
	for _, deg := range []float64{0, 10, 20, 25, 30, 40, 45, 50, 60, 70, 80} {
		got, report, err := DecodeWithRouteCapabilities(detect.RotateImage(img, deg), wire.ISO23634.Mask())
		if err != nil {
			t.Errorf("deg=%3.0f: %v", deg, err)
			continue
		}
		if string(got) != want {
			t.Errorf("deg=%3.0f: payload = %q, want %q", deg, got, want)
		}
		if report.Kind != "upright" || report.Angle != 0 || report.Attempts != 1 {
			t.Errorf("deg=%3.0f: report = %v; want a single upright route", deg, report)
		}
	}
}
