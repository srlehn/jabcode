//go:build !jabcode_legacy

package read

import "testing"

func TestDefaultBuildRejectsPreV2CReferenceJABCodes(t *testing.T) {
	fixtures := []string{
		"legacy_c_reference_pre_v2_8c.png",
		"legacy_c_reference_pre_v2_multi.png",
	}
	for _, fixture := range fixtures {
		t.Run(fixture, func(t *testing.T) {
			img := loadLegacyCReferenceFixture(t, fixture)
			if _, err := Decode(img); err == nil {
				t.Fatal("default build accepted a legacy JAB Code symbol from the pre-v2.0 C reference implementation")
			}
		})
	}
}
