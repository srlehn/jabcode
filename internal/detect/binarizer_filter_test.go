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

// balanceRGBDirect is the per-pixel stretch BalanceRGB tabulates: the same
// branches and arithmetic evaluated once per pixel instead of once per possible
// input byte. It is the reference the tabulated form must reproduce exactly.
func balanceRGBDirect(bm *core.Bitmap) {
	bpp := bm.Channels
	bytesPerRow := bm.Width * bpp
	const countThs = 20

	minMax := [3][2]int{}
	for c := range 3 {
		lo, hi := histMaxMin(histogram(bm, c), countThs)
		minMax[c] = [2]int{lo, hi}
	}
	for i := range bm.Height {
		for j := 0; j < bm.Width; j++ {
			offset := i*bytesPerRow + j*bpp
			for c := range 3 {
				lo, hi := minMax[c][0], minMax[c][1]
				v := int(bm.Pix[offset+c])
				switch {
				case v < lo:
					bm.Pix[offset+c] = 0
				case v > hi:
					bm.Pix[offset+c] = 255
				default:
					bm.Pix[offset+c] = byte(float64(v-lo) / float64(hi-lo) * 255.0)
				}
			}
		}
	}
}

// TestBalanceRGBMatchesDirectForm pins the tabulated histogram stretch to the
// per-pixel form. It covers the degenerate empty range a uniform or near-empty
// frame produces, where the per-pixel expression divides by zero: the table must
// reproduce that result rather than sanitise it, or a blank frame would binarize
// differently than before.
func TestBalanceRGBMatchesDirectForm(t *testing.T) {
	rng := rand.New(rand.NewPCG(0xba1a, 0x9ce5))
	fill := func(bm *core.Bitmap, mode int) {
		for i := range bm.Pix {
			switch mode {
			case 0: // uniform: histogram collapses, hi == lo
				bm.Pix[i] = 128
			case 1: // two-level, wide range
				if rng.Uint32N(2) == 0 {
					bm.Pix[i] = 12
				} else {
					bm.Pix[i] = 243
				}
			case 2: // narrow band just above the count threshold
				bm.Pix[i] = byte(100 + rng.Uint32N(3))
			default: // full-range noise
				bm.Pix[i] = byte(rng.Uint32N(256))
			}
		}
	}
	for _, channels := range []int{3, 4} {
		for _, sz := range []struct{ w, h int }{{1, 1}, {7, 5}, {32, 24}, {61, 41}} {
			for mode := range 4 {
				got := core.NewBitmap(sz.w, sz.h, channels)
				fill(got, mode)
				want := core.NewBitmap(sz.w, sz.h, channels)
				copy(want.Pix, got.Pix)

				BalanceRGB(got)
				balanceRGBDirect(want)
				if !bytes.Equal(got.Pix, want.Pix) {
					t.Fatalf("%dx%d bpp %d mode %d: tabulated stretch differs from the per-pixel form",
						sz.w, sz.h, channels, mode)
				}
			}
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
