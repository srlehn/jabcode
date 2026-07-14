//go:build jabcode_non_iso_encode

package jabcode

import "testing"

func TestTaggedEncoderProfiles(t *testing.T) {
	if got := ProfileISO23634.String(); got != "iso" {
		t.Fatalf("ProfileISO23634.String() = %q", got)
	}
	if got := ProfileHighColor.String(); got != "hc" {
		t.Fatalf("ProfileHighColor.String() = %q", got)
	}
	if _, err := NewEncoder(
		WithProfile(ProfileHighColor),
		WithColors(16),
	).Encode([]byte("tagged high-color encoder")); err != nil {
		t.Fatalf("high-color encode: %v", err)
	}
	if _, err := NewEncoder(WithProfile(Profile(255))).Encode([]byte("invalid")); err == nil {
		t.Fatal("invalid encoder profile was accepted")
	}
}
