package detect

import (
	"bytes"
	"math/rand/v2"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
)

// filterBinaryDirect is the direct 5-tap majority form filterBinary replaces:
// every output re-reads its whole kernel. It is the reference the running-window
// implementation must reproduce byte for byte.
func filterBinaryDirect(binary *core.Bitmap) {
	w, h := binary.Width, binary.Height
	const halfSize = 2
	tmp := make([]byte, w*h)

	copy(tmp, binary.Pix)
	for i := halfSize; i < h-halfSize; i++ {
		for j := halfSize; j < w-halfSize; j++ {
			sum := b2i(tmp[i*w+j] > 0)
			for k := 1; k <= halfSize; k++ {
				sum += b2i(tmp[i*w+(j-k)] > 0) + b2i(tmp[i*w+(j+k)] > 0)
			}
			binary.Pix[i*w+j] = b2byte(sum > halfSize)
		}
	}
	copy(tmp, binary.Pix)
	for i := halfSize; i < h-halfSize; i++ {
		for j := halfSize; j < w-halfSize; j++ {
			sum := b2i(tmp[i*w+j] > 0)
			for k := 1; k <= halfSize; k++ {
				sum += b2i(tmp[(i-k)*w+j] > 0) + b2i(tmp[(i+k)*w+j] > 0)
			}
			binary.Pix[i*w+j] = b2byte(sum > halfSize)
		}
	}
}

// TestFilterBinaryMatchesDirectForm pins the running-window majority filter to
// the direct form it replaces. The filter feeds the finder scan and the device
// parity chain, so a single differing byte would move harness rows silently;
// byte identity here is what makes the rewrite safe without re-running them.
func TestFilterBinaryMatchesDirectForm(t *testing.T) {
	// Sizes straddle the kernel: below it nothing may be written, at it exactly
	// one interior pixel per axis, then ordinary and odd/even shapes.
	sizes := []struct{ w, h int }{
		{1, 1}, {3, 3}, {4, 4}, {5, 5}, {5, 9}, {9, 5},
		{6, 7}, {16, 16}, {17, 33}, {64, 40}, {97, 61},
	}
	rng := rand.New(rand.NewPCG(0x5eed, 0xf11e))
	for _, sz := range sizes {
		for density := range 4 {
			// Vary how often a module is set, so majority ties and runs of
			// both polarities are exercised rather than one noise regime.
			threshold := uint32(1 + density*84) // 1, 85, 169, 253 of 256
			got := core.NewBitmap(sz.w, sz.h, 1)
			for i := range got.Pix {
				if rng.Uint32N(256) < threshold {
					got.Pix[i] = 255
				}
			}
			want := core.NewBitmap(sz.w, sz.h, 1)
			copy(want.Pix, got.Pix)

			filterBinary(got)
			filterBinaryDirect(want)
			if !bytes.Equal(got.Pix, want.Pix) {
				t.Fatalf("%dx%d density %d: running-window filter differs from the direct form",
					sz.w, sz.h, density)
			}
		}
	}
}
