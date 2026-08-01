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

// crossCheckPatternVerticalUnbounded is the superseded walk without the
// module-size bound, kept purely as the comparison target for the bounded
// form. It must stay a literal transcription: the bounded walk is only allowed
// to reach the same verdict sooner, and this is what proves it does.
func crossCheckPatternVerticalUnbounded(image *core.Bitmap, moduleSizeMax int, centerx float64, centery, moduleSize *float64, slack int) bool {
	const stateMiddle = 2
	var stateCount [5]int
	cx := int(centerx)
	cy := int(*centery)

	var i, stateIndex int
	stateCount[stateMiddle]++
	for i = 1; i <= cy && stateIndex <= stateMiddle; i++ {
		if image.Pix[(cy-i)*image.Width+cx] == image.Pix[(cy-(i-1))*image.Width+cx] {
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
	for i = 1; cy+i < image.Height && stateIndex <= stateMiddle; i++ {
		if image.Pix[(cy+i)*image.Width+cx] == image.Pix[(cy+(i-1))*image.Width+cx] {
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
	if ret && *moduleSize <= float64(moduleSizeMax) {
		*centery = float64(cy+i-stateCount[4]-stateCount[3]) - float64(stateCount[2])/2.0
		return true
	}
	return false
}

// TestCrossCheckPatternVerticalMatchesUnbounded pins the bounded vertical walk
// to the unbounded one it replaced. The columns run the same mix of
// finder-like, alternating, random binary and arbitrary byte content as the
// horizontal test, for the same reason: where the walk stops is decided by run
// structure.
func TestCrossCheckPatternVerticalMatchesUnbounded(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 13))
	for _, height := range []int{1, 2, 7, 9, 16, 33, 129} {
		for trial := range 200 {
			bm := core.NewBitmap(3, height, 1)
			switch trial % 4 {
			case 0:
				y := 0
				v := byte(0)
				for y < height {
					n := 1 + int(rng.UintN(6))
					for k := 0; k < n && y < height; k++ {
						bm.Pix[y*3+1] = v
						y++
					}
					v = 255 - v
				}
			case 1:
				for y := range height {
					bm.Pix[y*3+1] = byte(255 * (y % 2))
				}
			case 2:
				for y := range height {
					bm.Pix[y*3+1] = []byte{0, 255}[rng.UintN(2)]
				}
			default:
				for y := range height {
					bm.Pix[y*3+1] = byte(rng.UintN(256))
				}
			}
			for starty := range height {
				for _, slack := range []int{0, 1, 3} {
					for _, maxModule := range []int{1, 8} {
						cyGot, cyWant := float64(starty), float64(starty)
						var msGot, msWant float64
						got := crossCheckPatternVertical(bm, maxModule, 1, &cyGot, &msGot, slack)
						want := crossCheckPatternVerticalUnbounded(bm, maxModule, 1, &cyWant, &msWant, slack)
						if got != want || (want && (cyGot != cyWant || msGot != msWant)) {
							t.Fatalf("height %d starty %d slack %d max %d trial %d: got (%v, %v, %v), want (%v, %v, %v)",
								height, starty, slack, maxModule, trial, got, cyGot, msGot, want, cyWant, msWant)
						}
					}
				}
			}
		}
	}
}

// TestCrossCheckPatternHorizontalMatchesStepwise pins the run-batched walk to
// the per-pixel walk it replaced, on the outputs the walk actually promises:
// the verdict always, and the refined centre and module size wherever the
// verdict is positive. A rejected candidate leaves both untouched, since the
// module-size bound stops the walk before either is derived. The rows
// deliberately mix realistic finder-like run structure with adversarial
// patterns - alternating pixels, single-pixel runs and runs that straddle the
// eight-byte scan word - because the batching is exactly where run boundaries
// are decided. Arbitrary byte content is included as well, since the scans
// must not assume the channel bitmaps carry only two values. The two bounds
// bracket the row widths, so both the case where the module-size limit stops
// the walk and the case where it cannot are covered at every width.
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
					for _, maxModule := range []float64{1.5, 8} {
						cxGot, cxWant := float64(startx), float64(startx)
						var msGot, msWant float64
						got := crossCheckPatternHorizontal(bm, maxModule, &cxGot, 1, &msGot, slack)
						want := crossCheckPatternHorizontalStepwise(bm, maxModule, &cxWant, 1, &msWant, slack)
						if got != want || (want && (cxGot != cxWant || msGot != msWant)) {
							t.Fatalf("width %d startx %d slack %d max %v trial %d: got (%v, %v, %v), want (%v, %v, %v)\nrow %v",
								width, startx, slack, maxModule, trial, got, cxGot, msGot, want, cxWant, msWant, bm.Pix[width:2*width])
						}
					}
				}
			}
		}
	}
}
