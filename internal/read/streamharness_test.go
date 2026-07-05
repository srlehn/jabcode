//go:build jabharness

package read

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"math/rand"
	"testing"
	"time"

	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/encode"
)

// TestStreamHarness measures what the previous-frame prior is worth on
// synthetic hand-held preview streams: the same rotated symbol re-rendered
// over a sequence of frames with per-frame angle jitter, position drift and
// sensor noise. Every sequence is decoded twice - fresh through Decode, as
// if each frame were an isolated image, and through one Stream carrying its
// prior across frames - and the table reports per-frame wall time and the
// success count for both. Two regimes: a steady clearly-rotated hand, where
// the winning hypothesis is stable and the prior should carry every frame,
// and a sequence at the upright-detection boundary, where the winner
// flickers between upright and a rung and every flip costs a prior miss
// plus a full re-search - the prior's adversarial case.
//
//	go test -tags jabharness -run TestStreamHarness -v ./internal/read
func TestStreamHarness(t *testing.T) {
	payload := []byte("stream harness: previous-frame hypothesis reuse 0123456789")
	symbol, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 12, SymbolNumber: 1}, payload)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}

	const frameCount = 24
	sequences := []struct {
		name         string
		base, jitter float64
	}{
		{"steady-30deg", 30, 1.5},
		{"boundary-26deg", 26, 2},
	}
	for _, seq := range sequences {
		rng := rand.New(rand.NewSource(1))
		frames := make([]image.Image, frameCount)
		for i := range frames {
			angle := seq.base + seq.jitter*rng.Float64()
			rot := detect.RotateImage(symbol, angle)
			frame := image.NewNRGBA(image.Rect(0, 0, 1280, 960))
			draw.Draw(frame, frame.Bounds(), image.NewUniform(color.NRGBA{R: 212, G: 212, B: 212, A: 255}), image.Point{}, draw.Src)
			off := image.Pt(320+i*4+int(6*rng.Float64()), 200+i*3+int(6*rng.Float64()))
			draw.Draw(frame, rot.Bounds().Add(off), rot, image.Point{}, draw.Over)
			frames[i] = gaussianNoise(frame, 8, rng)
		}

		decodeAll := func(decode func(image.Image) ([]byte, error)) (times []time.Duration, okCount int) {
			times = make([]time.Duration, len(frames))
			for i, f := range frames {
				start := time.Now()
				out, err := decode(f)
				times[i] = time.Since(start)
				if err == nil && bytes.Equal(out, payload) {
					okCount++
				}
			}
			return times, okCount
		}

		freshTimes, freshOK := decodeAll(Decode)
		var s Stream
		streamTimes, streamOK := decodeAll(s.Decode)

		stats := func(times []time.Duration) (mean, worst time.Duration) {
			var sum time.Duration
			for _, d := range times {
				sum += d
				worst = max(worst, d)
			}
			return sum / time.Duration(len(times)), worst
		}
		freshMean, freshWorst := stats(freshTimes)
		streamMean, streamWorst := stats(streamTimes)

		var report bytes.Buffer
		fmt.Fprintf(&report, "%-8s %10s %10s %8s\n", "decoder", "mean", "worst", "ok")
		fmt.Fprintf(&report, "%-8s %10s %10s %5d/%2d\n", "fresh", freshMean.Round(time.Millisecond), freshWorst.Round(time.Millisecond), freshOK, frameCount)
		fmt.Fprintf(&report, "%-8s %10s %10s %5d/%2d\n", "stream", streamMean.Round(time.Millisecond), streamWorst.Round(time.Millisecond), streamOK, frameCount)
		t.Logf("stream harness %s (%d frames, 1280x960, noise sd 8):\n%s", seq.name, frameCount, report.String())

		if streamOK < freshOK {
			t.Errorf("%s: stream decoded %d frames, fresh decoded %d - the prior must never cost successes", seq.name, streamOK, freshOK)
		}
	}
}
