//go:build jabcode_bsi && jabcode_non_iso_encode

package read

import (
	"image"
	"slices"
	"testing"

	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/testutil"
	"github.com/srlehn/jabcode/internal/wire"
)

// A docked BSI secondary carries no finder patterns at all. Per BSI TR-03137
// Part 2 its corners are the smaller joined alignment patterns, it is placed
// relative to the host's docking edge, and its edge metadata has to be decoded
// before the far alignment patterns can be located. So the directional finder
// scan does nothing for it and the whole read rests on the BSI-specific
// alignment-pattern route, which still walks image rows.
//
// This pins where that route works today. Both docking orientations are covered
// because the docking edge is what the locator anchors to, and they do not fail
// at the same angles: docking below also reaches 105 and 270. The decoded angles
// sit within about 15 degrees of a multiple of 90, which is the axis-aligned
// walk's own acceptance band, and the whole 30-to-60 quadrant interior is lost
// in both orientations.
//
// The sets are specific to this payload, not a geometric law. The same two
// configurations carrying a shorter payload lose 90, 270, 285 and 345 when
// docked below while docking right is unchanged, so the marginal angles turn on
// symbol content as well as on angle. Change the payload here and the expected
// sets have to be re-measured.
//
// The table is a boundary, not a target, and it fails in both directions on
// purpose. A decoded angle turning failed is a regression. A failed angle
// turning decoded is the repair this coverage exists to enable, and it fails
// here too so that the expectation is advanced by review rather than absorbed
// silently - the same rule the capture census runs under. Failures print the
// stage the axis-aligned reference chain reaches, which localizes the break
// without being the route the decoder actually took.
func TestBSIDockedSecondaryAngleBoundary(t *testing.T) {
	for _, tc := range []struct {
		name      string
		positions []int
		versions  []image.Point
		decoded   []float64
	}{
		{
			name:      "secondary docked right",
			positions: []int{0, 4},
			versions:  []image.Point{image.Pt(3, 2), image.Pt(5, 2)},
			decoded:   []float64{0, 15, 75, 90, 165, 180, 195, 255, 285, 345},
		},
		{
			name:      "secondary docked below",
			positions: []int{0, 2},
			versions:  []image.Point{image.Pt(3, 2), image.Pt(3, 4)},
			decoded:   []float64{0, 15, 75, 90, 105, 165, 180, 195, 255, 270, 285, 345},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			payload := []byte("BSI docked secondary at an angle")
			img, err := encode.Run(encode.Config{
				Colors: 8, ModuleSize: 12, Format: wire.EncodeBSI, SymbolNumber: 2,
				SymbolPositions: tc.positions,
				SymbolVersions:  tc.versions,
				SymbolECCLevels: []int{0, 0},
			}, payload)
			if err != nil {
				t.Fatal(err)
			}
			for deg := 0.0; deg < 360; deg += 15 {
				rotated := testutil.RotateImage(img, deg)
				got, report, err := DecodeWithRouteCapabilities(rotated, wire.BSI.Mask())
				if !slices.Contains(tc.decoded, deg) {
					// An angle expected to fail must fail by returning an
					// error. Returning a payload is either the repair, which
					// belongs in the expected set, or a wrong payload, which is
					// the fatal case: hard LDPC carries no integrity check, so
					// a confident wrong answer here would otherwise pass
					// unnoticed.
					switch {
					case err != nil:
					case string(got) == string(payload):
						t.Errorf("deg=%3.0f now decodes; fold it into the expected set", deg)
					default:
						t.Errorf("deg=%3.0f returned a wrong payload of %d bytes with no error",
							deg, len(got))
					}
					continue
				}
				if err != nil {
					t.Errorf("deg=%3.0f: %v | reference chain: %s", deg, err, bsiSecondaryStageSummary(rotated))
					continue
				}
				if string(got) != string(payload) {
					t.Errorf("deg=%3.0f: payload = %q, want %q", deg, got, payload)
				}
				// The route matters as much as the payload: full-frame rotation
				// would also reach these angles by turning the frame until the
				// patterns fall into the axis-aligned band, so a payload-only
				// assertion would pass with the source-frame work reverted.
				if report.Kind != "upright" || report.Attempts != 1 {
					t.Errorf("deg=%3.0f: report = %v; want a single upright route", deg, report)
				}
			}
		})
	}
}
