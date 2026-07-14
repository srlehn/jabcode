//go:build jabcode_legacy

package read

import (
	"bytes"
	"testing"

	"github.com/srlehn/jabcode/internal/wire"
)

func TestLegacyBuildDecodesPreV2CReferenceJABCodes(t *testing.T) {
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
			got, err := DecodeProfile(img, wire.Legacy)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, []byte(tc.want)) {
				t.Fatalf("legacy DecodeProfile() = %q, want %q", got, tc.want)
			}
			if _, err := DecodeProfile(img, wire.ISO23634); err == nil {
				t.Fatal("experimental ISO profile accepted a legacy JAB Code symbol from the pre-v2.0 C reference implementation")
			}
		})
	}
}
