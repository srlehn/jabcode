package read

import (
	"bytes"
	"image"
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
	if len(s.ring) == 0 {
		t.Fatal("no prior recorded after a successful rotated read")
	}
	if s.ring[0].deg == 0 {
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

// TestStreamQuota asserts the scheduler's per-frame work contract: every
// counter stays within its cap on decodable, rotated and hopeless frames
// alike, retained state stays bounded, and an identical frame sequence
// through a fresh Stream reproduces identical outputs. The exhaustive
// ladder's stages cannot exceed their zero budget by construction - the
// scheduler never calls the region-of-interest proposer or the
// alignment-pattern fallback.
func TestStreamQuota(t *testing.T) {
	msg := []byte("stream quota contract")
	img, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 12, SymbolNumber: 1}, msg)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	blank := image.NewNRGBA(image.Rect(0, 0, 480, 480))
	frames := []image.Image{
		detect.RotateImage(img, 30),
		blank,
		detect.RotateImage(img, 31),
		img,
		blank,
		detect.RotateImage(img, 29),
	}

	run := func() ([][]byte, []error) {
		var s Stream
		outs := make([][]byte, len(frames))
		errs := make([]error, len(frames))
		for i, f := range frames {
			outs[i], errs[i] = s.Decode(f)
			w := s.work
			if w.replayAttempts > 1 || w.uprightScans > 1 || w.rotatedAttempts > 1 || w.correctionChains > 1 {
				t.Errorf("frame %d: work over quota: %+v", i, w)
			}
			if len(s.ring) > streamRingCap || len(s.pending) > streamPendingCap {
				t.Errorf("frame %d: retained state over bounds: ring %d, pending %d", i, len(s.ring), len(s.pending))
			}
			if outs[i] != nil && !bytes.Equal(outs[i], msg) {
				t.Errorf("frame %d: wrong payload %q", i, outs[i])
			}
		}
		return outs, errs
	}

	outs1, errs1 := run()
	outs2, errs2 := run()
	for i := range frames {
		if !bytes.Equal(outs1[i], outs2[i]) || (errs1[i] == nil) != (errs2[i] == nil) {
			t.Errorf("frame %d: replayed sequence diverged", i)
		}
	}
}
