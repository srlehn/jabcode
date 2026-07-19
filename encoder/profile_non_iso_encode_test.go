//go:build jabcode_non_iso_encode

package encoder_test

import (
	"testing"

	"github.com/srlehn/jabcode/encoder"
)

func TestPublicEncoderTaggedProfile(t *testing.T) {
	_, err := encoder.New(
		encoder.WithProfile(encoder.ProfileHighColor),
		encoder.WithColors(16),
	).Encode([]byte("public high-color encoder"))
	if err != nil {
		t.Fatalf("high-color encode: %v", err)
	}
}
