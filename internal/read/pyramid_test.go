package read

import (
	"testing"

	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/encode"
)

// TestDecodePyramid checks that a capture large enough for the resolution
// pyramid decodes through it - upright and rotated past the upright detection
// limit - with the payload the single-level path would read. The rotated case
// drives the concurrent levels through their orientation rungs and the
// quit-on-coarser-success cancellation.
func TestDecodePyramid(t *testing.T) {
	msg := []byte("resolution pyramid round-trip")
	img, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 32, SymbolNumber: 1}, msg)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if levels := pyramidLevels(img); len(levels) < 2 {
		t.Fatalf("test image %v yields %d pyramid levels, want at least 2", img.Bounds(), len(levels))
	}
	got, err := Decode(img)
	if err != nil {
		t.Fatalf("Decode upright: %v", err)
	}
	if string(got) != string(msg) {
		t.Fatalf("Decode upright: got %q, want %q", got, msg)
	}
	got, err = Decode(detect.RotateImage(img, 30))
	if err != nil {
		t.Fatalf("Decode rotated 30 deg: %v", err)
	}
	if string(got) != string(msg) {
		t.Fatalf("Decode rotated 30 deg: got %q, want %q", got, msg)
	}
}
