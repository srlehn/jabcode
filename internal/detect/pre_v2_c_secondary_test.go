//go:build jabcode_bsi || jabcode_legacy

package detect

import (
	"testing"

	"github.com/srlehn/jabcode/internal/core"
)

func TestFindBSIFamilyAlignmentPatternDeferredChannels(t *testing.T) {
	channels := [3]*core.Bitmap{}
	for i := range channels {
		ch := &core.Bitmap{Width: 64, Height: 64, Channels: 1}
		ch.SetPixelReader(func(int, int) byte { return 0 })
		channels[i] = ch
	}

	got := findBSIFamilyAlignmentPattern(channels, 32, 32, 2, ap0, [3]byte{0, 0, 0})
	if got.Typ != -1 {
		t.Fatalf("deferred alignment search found pattern %+v in blank channels", got)
	}
	for i, ch := range channels {
		if ch.Pix != nil {
			t.Fatalf("deferred channel %d was materialized", i)
		}
	}
}
