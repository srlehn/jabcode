//go:build jabcode_bsi && jabcode_non_iso_encode

package read

import (
	"testing"

	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/wire"
)

// Arbitrary-angle decoding is a property of the reader, not of one wire family.
// A BSI finder is the same joined-squares construction as the current one, so
// it fails an axis-aligned row scan off upright for the same reason and needs
// the same directional scan.
//
// Scope, because a pass here is easy to over-read. This covers a single-primary
// BSI symbol, clean synthetic render, no perspective, uniform scale, and a
// forced wire.BSI capability rather than the ordinary additive Decode. It says
// nothing about docked or recursive BSI: a secondary carries no finder patterns
// at all, being located from the host's docking edge and its corner alignment
// patterns, which still go through the axis-aligned locator.
func TestBSISinglePrimaryDecodesAtArbitraryAngles(t *testing.T) {
	payload := []byte("BSI at an angle")
	img, err := encode.Run(encode.Config{
		Colors: 8, ModuleSize: 12, Format: wire.EncodeBSI, SymbolNumber: 1,
	}, payload)
	if err != nil {
		t.Fatal(err)
	}
	for _, deg := range []float64{0, 15, 30, 45, 60, 75, 100, 145, 200, 285, 330} {
		got, report, err := DecodeWithRouteCapabilities(detect.RotateImage(img, deg), wire.BSI.Mask())
		if err != nil {
			t.Errorf("deg=%3.0f: %v", deg, err)
			continue
		}
		if string(got) != string(payload) {
			t.Errorf("deg=%3.0f: payload = %q, want %q", deg, got, payload)
		}
		// The route matters as much as the payload here. The ladder would also
		// decode these, so a payload-only assertion would pass with the
		// directional BSI scan deleted; requiring the upright route is what
		// pins that the scan found the symbol without a rotated canvas.
		if report.Kind != "upright" || report.Angle != 0 || report.Attempts != 1 {
			t.Errorf("deg=%3.0f: report = %v; want a single upright route", deg, report)
		}
	}
}
