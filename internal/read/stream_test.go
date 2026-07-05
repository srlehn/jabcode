package read

import (
	"testing"

	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/encode"
)

// TestStreamPrior checks the stream decoder's hypothesis reuse: the first
// rotated frame must record the winning level and rung, and the following
// near-identical frame must decode to the same payload (through the prior
// fast path when it holds, through the fallback search when it does not -
// the payload contract is the same either way).
func TestStreamPrior(t *testing.T) {
	msg := []byte("stream prior round-trip")
	img, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 12, SymbolNumber: 1}, msg)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	var s Stream
	got, err := s.Decode(detect.RotateImage(img, 30))
	if err != nil {
		t.Fatalf("Decode frame 1: %v", err)
	}
	if string(got) != string(msg) {
		t.Fatalf("Decode frame 1: got %q, want %q", got, msg)
	}
	if s.prior == nil {
		t.Fatal("no prior recorded after a successful rotated read")
	}
	if s.prior.deg == 0 {
		t.Fatal("rotated read recorded an upright prior")
	}
	got, err = s.Decode(detect.RotateImage(img, 31))
	if err != nil {
		t.Fatalf("Decode frame 2: %v", err)
	}
	if string(got) != string(msg) {
		t.Fatalf("Decode frame 2: got %q, want %q", got, msg)
	}
}
