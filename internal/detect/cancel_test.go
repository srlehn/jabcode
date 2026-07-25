package detect

import (
	"bytes"
	"sync/atomic"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
)

// cancelTestBitmap builds a deterministic noisy frame large enough that the
// full-frame passes split into several parallel bands, which is what makes the
// hook fire from more than one goroutine.
func cancelTestBitmap(width, height int) *core.Bitmap {
	bm := core.NewBitmap(width, height, 4)
	state := uint32(0x9e3779b9)
	for index := range width * height {
		for c := range 3 {
			state = state*1664525 + 1013904223
			bm.Pix[index*4+c] = byte(state >> 24)
		}
		bm.Pix[index*4+3] = 255
	}
	return bm
}

// firesAfter returns a monotone quit hook that reports true from its (n+1)-th
// call onward, so a pass can be cancelled at a chosen depth. The counter is
// atomic because the pixel passes poll from every band goroutine.
func firesAfter(n int64) func() bool {
	var polls atomic.Int64
	return func() bool { return polls.Add(1) > n }
}

// TestBinarizerRGBUntilCancels pins both halves of the cancellable binarizer:
// uncancelled it is the plain binarizer byte for byte, and once cancelled it
// reports no channels at all. The second half is the load-bearing one - a
// partially binarized frame would sample as real data, and hard LDPC success
// is not payload integrity, so a half-written mask has no safe consumer.
func TestBinarizerRGBUntilCancels(t *testing.T) {
	bm := cancelTestBitmap(160, 160)
	want := BinarizerRGB(bm, nil)

	got, ok := BinarizerRGBUntil(bm, nil, nil)
	if !ok {
		t.Fatal("nil hook cancelled the binarizer")
	}
	for c := range got {
		if !bytes.Equal(got[c].Pix, want[c].Pix) {
			t.Fatalf("channel %d differs from BinarizerRGB with a nil hook", c)
		}
	}

	got, ok = BinarizerRGBUntil(bm, nil, func() bool { return false })
	if !ok {
		t.Fatal("hook that never fires cancelled the binarizer")
	}
	for c := range got {
		if !bytes.Equal(got[c].Pix, want[c].Pix) {
			t.Fatalf("channel %d differs from BinarizerRGB with an inert hook", c)
		}
	}

	// The deeper counts are only reachable if the pass polls per scanline, so
	// they pin the cancellation granularity as well as the return value: a
	// pass that polled once at the end would report success for all but the
	// first of them.
	for _, after := range []int64{0, 1, 17, 100} {
		got, ok := BinarizerRGBUntil(bm, nil, firesAfter(after))
		if ok {
			t.Fatalf("cancelled after %d polls but the binarizer reported success", after)
		}
		if got != ([3]*core.Bitmap{}) {
			t.Fatalf("cancelled after %d polls but channels were returned", after)
		}
	}
}

// TestDescreenCancels pins the same contract for the descreen filter, which
// runs ahead of the binarizer in the retry passes and is cancellable at its
// per-plane boundaries.
func TestDescreenCancels(t *testing.T) {
	bm := cancelTestBitmap(96, 96)
	want := descreen(bm, 2, 3, nil)

	got := descreen(bm, 2, 3, func() bool { return false })
	if got == nil {
		t.Fatal("hook that never fires cancelled the descreen")
	}
	if !bytes.Equal(got.Pix, want.Pix) {
		t.Fatal("descreen with an inert hook differs from the uncancelled filter")
	}

	// Nine poll points, three per colour plane, so this covers cancelling
	// before any plane and part-way through the middle one.
	for _, after := range []int64{0, 1, 4} {
		if got := descreen(bm, 2, 3, firesAfter(after)); got != nil {
			t.Fatalf("cancelled after %d polls but the descreen returned a frame", after)
		}
	}
}
