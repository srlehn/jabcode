package detect

import (
	"math/rand/v2"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
)

// crossCheckPatternHorizontalStepwise is the superseded per-pixel walk, kept
// purely as the comparison target for the run-batched form. It must stay a
// literal transcription: the batched walk is only allowed to reach the same
// decisions faster, and this is what proves it does.
func crossCheckPatternHorizontalStepwise(image *core.Bitmap, moduleSizeMax float64, centerx *float64, centery float64, moduleSize *float64, slack int) bool {
	const stateMiddle = 2
	var stateCount [5]int
	startx := int(*centerx)
	rowOffset := int(centery) * image.Width

	var i, stateIndex int
	stateCount[stateMiddle]++
	for i = 1; i <= startx && stateIndex <= stateMiddle; i++ {
		if image.Pix[rowOffset+(startx-i)] == image.Pix[rowOffset+(startx-(i-1))] {
			stateCount[stateMiddle-stateIndex]++
		} else if stateIndex > 0 && stateCount[stateMiddle-stateIndex] < slack {
			stateCount[stateMiddle-(stateIndex-1)] += stateCount[stateMiddle-stateIndex]
			stateCount[stateMiddle-stateIndex] = 0
			stateIndex--
			stateCount[stateMiddle-stateIndex]++
		} else {
			stateIndex++
			if stateIndex > stateMiddle {
				break
			}
			stateCount[stateMiddle-stateIndex]++
		}
	}
	if stateIndex < stateMiddle {
		return false
	}
	stateIndex = 0
	for i = 1; startx+i < image.Width && stateIndex <= stateMiddle; i++ {
		if image.Pix[rowOffset+(startx+i)] == image.Pix[rowOffset+(startx+(i-1))] {
			stateCount[stateMiddle+stateIndex]++
		} else if stateIndex > 0 && stateCount[stateMiddle+stateIndex] < slack {
			stateCount[stateMiddle+(stateIndex-1)] += stateCount[stateMiddle+stateIndex]
			stateCount[stateMiddle+stateIndex] = 0
			stateIndex--
			stateCount[stateMiddle+stateIndex]++
		} else {
			stateIndex++
			if stateIndex > stateMiddle {
				break
			}
			stateCount[stateMiddle+stateIndex]++
		}
	}
	if stateIndex < stateMiddle {
		return false
	}
	ms, ret := checkPatternCross(stateCount)
	*moduleSize = ms
	if ret && *moduleSize <= moduleSizeMax {
		*centerx = float64(startx+i-stateCount[4]-stateCount[3]) - float64(stateCount[2])/2.0
		return true
	}
	return false
}

// TestCrossCheckPatternHorizontalMatchesStepwise pins the run-batched walk to
// the per-pixel walk it replaced, on every output it produces: the verdict, the
// refined centre and the module size. The rows deliberately mix realistic
// finder-like run structure with adversarial patterns - alternating pixels,
// single-pixel runs and runs that straddle the eight-byte scan word - because
// the batching is exactly where run boundaries are decided. Arbitrary byte
// content is included as well, since the scans must not assume the channel
// bitmaps carry only two values.
func TestCrossCheckPatternHorizontalMatchesStepwise(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 5))
	for _, width := range []int{1, 2, 7, 8, 9, 15, 16, 17, 64, 129} {
		for trial := range 400 {
			bm := core.NewBitmap(width, 3, 1)
			switch trial % 4 {
			case 0: // finder-like: runs around a plausible module size
				x := 0
				v := byte(0)
				for x < len(bm.Pix) {
					n := 1 + int(rng.UintN(6))
					for k := 0; k < n && x < len(bm.Pix); k++ {
						bm.Pix[x] = v
						x++
					}
					v = 255 - v
				}
			case 1: // alternating, worst case for batching
				for x := range bm.Pix {
					bm.Pix[x] = byte(255 * (x % 2))
				}
			case 2: // random binary
				for x := range bm.Pix {
					bm.Pix[x] = []byte{0, 255}[rng.UintN(2)]
				}
			default: // arbitrary bytes
				for x := range bm.Pix {
					bm.Pix[x] = byte(rng.UintN(256))
				}
			}
			for startx := range width {
				for _, slack := range []int{0, 1, 3} {
					cxGot, cxWant := float64(startx), float64(startx)
					var msGot, msWant float64
					got := crossCheckPatternHorizontal(bm, 8, &cxGot, 1, &msGot, slack)
					want := crossCheckPatternHorizontalStepwise(bm, 8, &cxWant, 1, &msWant, slack)
					if got != want || cxGot != cxWant || msGot != msWant {
						t.Fatalf("width %d startx %d slack %d trial %d: got (%v, %v, %v), want (%v, %v, %v)\nrow %v",
							width, startx, slack, trial, got, cxGot, msGot, want, cxWant, msWant, bm.Pix[width:2*width])
					}
				}
			}
		}
	}
}
