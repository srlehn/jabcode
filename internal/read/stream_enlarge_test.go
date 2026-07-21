package read

import (
	"image"
	"testing"
)

// TestStreamEnlargedHypothesis pins where the enlarged detection scale sits in
// the stream budget. A single-scale frame too small to verify a finder at its
// native pixels has no finer level to escalate to, so it escalates to the
// enlarged scale: a carried hypothesis spending its own budget slot, rate
// limited so the enlargement's four times the pixels stay off every failing
// frame. A frame that carries a pyramid escalates to its finer levels instead
// and must never enlarge.
func TestStreamEnlargedHypothesis(t *testing.T) {
	// Texture rather than a uniform frame: the escalation exists for a capture
	// whose structure the cross-checks cannot confirm, and a blank frame would
	// leave through the cheap uniform bailout instead.
	small := deterministicNoise(image.Rect(0, 0, 480, 640), 1)
	var s Stream
	enlarged, frames := 0, streamPendingCap+1
	for range frames {
		if _, err := s.Decode(small); err == nil {
			t.Fatal("noise frame decoded")
		}
		if s.work.enlargedAttempts > 1 {
			t.Errorf("enlarged attempts %d over the per-frame cap", s.work.enlargedAttempts)
		}
		enlarged += s.work.enlargedAttempts
	}
	if enlarged == 0 {
		t.Errorf("small single-scale frame never reached the enlarged scale in %d frames", frames)
	}
	// A frame whose orientation probe yields no rung leaves the enlarged entry
	// alone in the queue, so being carried bounds nothing on its own; the
	// cadence is what keeps the enlargement's squared pixel cost amortized to
	// one native-scale attempt per frame.
	period := enlargeFactor * enlargeFactor
	if limit := (frames + period - 1) / period; enlarged > limit {
		t.Errorf("enlarged %d times in %d frames, over the one-per-%d cadence (limit %d)",
			enlarged, frames, period, limit)
	}

	// A frame that carries a pyramid has finer levels to escalate to, and the
	// enlargement's cost must not be paid on top of them.
	large := deterministicNoise(image.Rect(0, 0, 2*minPyramidSide, 2*minPyramidSide), 2)
	var big Stream
	for range 3 {
		if _, err := big.Decode(large); err == nil {
			t.Fatal("noise frame decoded")
		}
		if big.work.enlargedAttempts != 0 {
			t.Fatalf("pyramided frame enlarged: %+v", big.work)
		}
	}
}
