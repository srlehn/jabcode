//go:build jabcode_legacy

package read

import (
	"bytes"
	"image"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/wire"
)

func TestLegacyTagDecodesPreV2CReferenceJABCodes(t *testing.T) {
	// Both fixtures were generated and round-tripped with the pre-v2.0 C
	// reference implementation at commit 2ece74e. The multi-symbol fixture uses
	// positions 0,3,2 and side versions 3x2,4x2,3x2.
	tests := []struct {
		fixture string
		want    string
	}{
		{fixture: "legacy_c_reference_pre_v2_8c.png", want: "Legacy C-reference JAB Code primary fixture 0123456789"},
		{fixture: "legacy_c_reference_pre_v2_multi.png", want: "Legacy C-reference JAB Code multi-symbol fixture 0123456789"},
	}
	for _, tc := range tests {
		t.Run(tc.fixture, func(t *testing.T) {
			img := loadLegacyCReferenceFixture(t, tc.fixture)
			auto, err := Decode(img)
			if err != nil {
				t.Fatalf("additive Decode: %v", err)
			}
			if !bytes.Equal(auto, []byte(tc.want)) {
				t.Fatalf("additive Decode() = %q, want %q", auto, tc.want)
			}
			got, trace, err := DecodeWithTraceOnly(img, wire.PreV2C)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, []byte(tc.want)) {
				t.Fatalf("legacy DecodeOnly() = %q, want %q", got, tc.want)
			}
			if tc.fixture == "legacy_c_reference_pre_v2_multi.png" {
				assertPreV2CDockedTrace(t, trace, []byte(tc.want))
			}
			if _, err := DecodeOnly(img, wire.ISO23634); err == nil {
				t.Fatal("experimental ISO variant accepted a legacy JAB Code symbol from the pre-v2.0 C reference implementation")
			}

			frame := testNRGBA(img)
			var finding finding
			located, stage, _ := decodeBitmapFindingTracedOnly(core.BitmapFromImage(frame), func() bool { return false }, &finding, nil, wire.PreV2C)
			if stage != readDecoded || !bytes.Equal(located, []byte(tc.want)) {
				t.Fatalf("located pre-v2.0 decode = %q stage=%d, want %q", located, stage, tc.want)
			}
			if finding.family != detect.FinderFamilyBSI {
				t.Fatalf("finding family = %d, want BSI/pre-v2.0", finding.family)
			}
			seeded, _, ok := decodeSeededTracedOnly([]*image.NRGBA{frame, frame}, finding, func() bool { return false }, nil, wire.PreV2C)
			if !ok || !bytes.Equal(seeded, []byte(tc.want)) {
				t.Fatalf("seeded pre-v2.0 decode = %q ok=%v, want %q", seeded, ok, tc.want)
			}
		})
	}
}

func assertPreV2CDockedTrace(t *testing.T, trace *DiagnosticTrace, payload []byte) {
	t.Helper()
	var secondaries []DiagnosticSecondary
	for i := range trace.Attempts {
		if bytes.Equal(trace.Attempts[i].Payload, payload) {
			secondaries = trace.Attempts[i].Secondaries
			break
		}
	}
	if len(secondaries) == 0 {
		t.Fatal("successful pre-v2.0 traversal has no docked-secondary trace")
	}
	for i := range secondaries {
		secondary := &secondaries[i]
		if secondary.Symbol.WireVariant != wire.PreV2C {
			t.Fatalf("secondary %d variant = %d, want pre-v2.0", i, secondary.Symbol.WireVariant)
		}
		if secondary.Symbol.Index != i+1 || secondary.Symbol.HostIndex != secondary.HostIndex {
			t.Fatalf("secondary %d index/host = %d/%d, trace host %d", i,
				secondary.Symbol.Index, secondary.Symbol.HostIndex, secondary.HostIndex)
		}
		if secondary.Matrix == nil {
			t.Fatalf("secondary %d omitted its sampled matrix", i)
		}
		moduleCount := secondary.Matrix.Width * secondary.Matrix.Height
		if len(secondary.Classification.DataMap) != moduleCount ||
			len(secondary.Classification.Colors) != moduleCount {
			t.Fatalf("secondary %d classification = matrix %v map %d colors %d", i,
				secondary.Matrix != nil, len(secondary.Classification.DataMap), len(secondary.Classification.Colors))
		}
	}
}
