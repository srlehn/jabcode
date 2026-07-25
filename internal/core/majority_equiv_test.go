package core

import (
	"math/rand/v2"
	"testing"
)

// majority5RowNaive and majority5ColumnsNaive are the per-pixel forms the
// word-at-a-time passes replaced, kept purely as the comparison target. They
// state the contract directly: a centre is set when at least three of its five
// taps are, and pixels without a full kernel are left untouched.
func majority5RowNaive(src, dst []byte, width int) {
	if width < 5 {
		return
	}
	for center := 2; center < width-2; center++ {
		count := 0
		for k := -2; k <= 2; k++ {
			if src[center+k] != 0 {
				count++
			}
		}
		if count > 2 {
			dst[center] = 255
		} else {
			dst[center] = 0
		}
	}
}

func majority5ColumnsNaive(src, dst []byte, width, height int) {
	const radius = 2
	if width < 2*radius+1 || height < 2*radius+1 {
		return
	}
	for i := radius; i < height-radius; i++ {
		for j := radius; j < width-radius; j++ {
			count := 0
			for k := -radius; k <= radius; k++ {
				if src[(i+k)*width+j] != 0 {
					count++
				}
			}
			if count > 2 {
				dst[i*width+j] = 255
			} else {
				dst[i*width+j] = 0
			}
		}
	}
}

// TestMajority5RowMatchesNaive sweeps every width around the eight-pixel word
// the pass now votes with, which is where a batched form drops or doubles a
// centre, and checks that untouched edge pixels really are untouched.
func TestMajority5RowMatchesNaive(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	for width := 1; width <= 40; width++ {
		for trial := range 8 {
			src := make([]byte, width)
			for i := range src {
				switch trial % 4 {
				case 0:
					if rng.UintN(2) == 1 {
						src[i] = 255
					}
				case 1:
					src[i] = 255
				case 2:
					// alternating, the worst case for a majority window
					if i%2 == 0 {
						src[i] = 255
					}
				default:
					if rng.UintN(5) == 0 {
						src[i] = 255
					}
				}
			}
			const sentinel = 7
			got := make([]byte, width)
			want := make([]byte, width)
			for i := range got {
				got[i], want[i] = sentinel, sentinel
			}
			majority5Row(src, got, width)
			majority5RowNaive(src, want, width)
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("width %d trial %d index %d: got %d, want %d", width, trial, i, got[i], want[i])
				}
			}
		}
	}
}

// TestMajority5ColumnsMatchesNaive does the same for the vertical pass, over
// widths that straddle the word and heights that exercise the row chunking.
func TestMajority5ColumnsMatchesNaive(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 17))
	for _, width := range []int{1, 4, 5, 8, 9, 11, 12, 13, 16, 17, 20, 33, 64, 65} {
		for _, height := range []int{1, 5, 6, 9, 17, 32} {
			src := make([]byte, width*height)
			for i := range src {
				if rng.UintN(2) == 1 {
					src[i] = 255
				}
			}
			const sentinel = 3
			got := make([]byte, width*height)
			want := make([]byte, width*height)
			for i := range got {
				got[i], want[i] = sentinel, sentinel
			}
			Majority5Columns(src, got, width, height)
			majority5ColumnsNaive(src, want, width, height)
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("%dx%d index %d (row %d col %d): got %d, want %d",
						width, height, i, i/width, i%width, got[i], want[i])
				}
			}
		}
	}
}

// TestMajority5MaskVotes checks the carry-save vote against a direct count on
// every one of the 32 input combinations, in every byte lane.
func TestMajority5MaskVotes(t *testing.T) {
	for pattern := range 32 {
		var in [5]uint64
		count := 0
		for k := range 5 {
			if pattern&(1<<k) != 0 {
				in[k] = ^uint64(0)
				count++
			}
		}
		want := uint64(0)
		if count > 2 {
			want = ^uint64(0)
		}
		if got := majority5Mask(in[0], in[1], in[2], in[3], in[4]); got != want {
			t.Fatalf("pattern %05b (count %d): got %#x, want %#x", pattern, count, got, want)
		}
	}
}
