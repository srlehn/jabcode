package read

import (
	"testing"

	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/encode"
)

// TestDecodeRotated checks that Decode's rotation ladder reads a clean code that
// has been rotated past the ~20-degree angle where upright finder detection
// collapses. The decoded bytes must match regardless of orientation.
func TestDecodeRotated(t *testing.T) {
	msg := []byte("rotation ladder round-trip")
	img, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 12, SymbolNumber: 1}, msg)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	for _, deg := range []float64{20, 30, 40, 45, -35, 60} {
		got, err := Decode(detect.RotateImage(img, deg))
		if err != nil {
			t.Errorf("Decode rotated %g deg: %v", deg, err)
			continue
		}
		if string(got) != string(msg) {
			t.Errorf("Decode rotated %g deg: got %q, want %q", deg, got, msg)
		}
	}
}
