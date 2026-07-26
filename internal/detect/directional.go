package detect

import (
	"math"

	"github.com/srlehn/jabcode/internal/core"
)

// Directional finder scanning: reading a rotated symbol by turning the scan
// line rather than the image.
//
// The axis-aligned walks in finderpattern.go can only see a finder whose
// symbol axis is within roughly ten degrees of an image axis, because beyond
// that a straight scan through the core drifts out of the core's own module row
// and starts crossing the two quadrants that are data rather than pattern (the
// finder is two square references joined at a core, not concentric rings; see
// the fp0..fp3 block). Pre-rotating the whole frame fixes that by moving the
// pixels. Scanning along the symbol's own direction fixes it without touching
// them, and one prepared frame then serves every direction.
//
// A scan line is undirected and the finder is half-turn symmetric, so
// directions fold modulo 180; the four finders' arrangement folds a quarter
// turn on top of that, leaving [0,90) to cover. scanDirections tiles it at the
// same 15-degree step coarseProbeAngles uses, for the same reason: a 7.5-degree
// worst-case residual sits inside the measured survival band, while the
// 11.25 degrees a four-direction set would leave does not.

// scanDirection is one probe direction. The walk advances its major axis by one
// pixel per sample, which keeps the layer-crossing ratios of the finder
// signature exact - every layer boundary scales by the same factor - at the
// cost of sampling the line more coarsely the further it tilts.
type scanDirection struct {
	dx, dy float64 // unit step along the major axis, one component being +/-1

	// pxPerSample is the physical distance one sample covers, 1/max(|cos|,|sin|).
	// Run lengths are measured in samples, so converting by this factor is what
	// makes a directional module size comparable with an axis-aligned one
	// rather than up to 41 percent short.
	pxPerSample float64
}

// newScanDirection builds the walk parameters for a scan at angleDeg.
func newScanDirection(angleDeg float64) scanDirection {
	a := angleDeg * math.Pi / 180
	c, s := math.Cos(a), math.Sin(a)
	major := math.Max(math.Abs(c), math.Abs(s))
	return scanDirection{dx: c / major, dy: s / major, pxPerSample: 1 / major}
}

// perpendicular returns the direction turned a quarter turn, which is the
// directional counterpart of the vertical cross-check.
func (d scanDirection) perpendicular() scanDirection {
	// Rotating (dx,dy) by 90 degrees and renormalizing to a unit major axis.
	x, y := -d.dy, d.dx
	major := math.Max(math.Abs(x), math.Abs(y))
	return scanDirection{dx: x / major, dy: y / major, pxPerSample: 1 / major}
}

// scanDirections are the probe directions covering [0,90). Kept at six for the
// residual-angle reason above; reducing the count is not where the saving is,
// since all of them read one prepared frame.
var scanDirections = []float64{0, 15, 30, 45, 60, 75}

// crossCheckPatternAlong validates a finder candidate along an arbitrary
// direction and refines its centre along that direction, reporting whether the
// five-run signature holds. It is the directional counterpart of
// crossCheckPatternHorizontal.
//
// Two differences from that function are forced rather than chosen. The word
// scans in runlength.go read eight contiguous bytes, which a strided walk
// cannot use, so runs are measured a sample at a time. And the module size is
// converted from samples to physical pixels before returning, because every
// consumer of ModuleSize expects the true module size.
//
// centre and moduleSize are outputs of a positive verdict only, matching the
// axis-aligned walks.
func crossCheckPatternAlong(img *core.Bitmap, dir scanDirection, moduleSizeMax float64, centre *core.PointF, moduleSize *float64, slack int) bool {
	const stateMiddle = 2
	var stateCount [5]int

	sx := *centre
	x0, y0 := int(sx.X), int(sx.Y)
	if x0 < 0 || x0 >= img.Width || y0 < 0 || y0 >= img.Height {
		return false
	}

	// The same bound the axis-aligned walks take: the module size is one third
	// of the three middle state counts, those counts never shrink, so once
	// their sum passes the limit no continuation recovers the candidate. In
	// samples rather than pixels, since that is what the counts are in.
	insideLimit := img.Width + img.Height + 1
	if lim := moduleSizeMax * 3 / dir.pxPerSample; lim >= 0 && lim < float64(insideLimit) {
		insideLimit = int(lim) + 2
	}
	inside := 1

	at := func(i int, sign float64) (byte, bool) {
		x := int(sx.X + sign*float64(i)*dir.dx)
		y := int(sx.Y + sign*float64(i)*dir.dy)
		if x < 0 || x >= img.Width || y < 0 || y >= img.Height {
			return 0, false
		}
		return img.Pix[y*img.Width+x], true
	}

	stateCount[stateMiddle]++
	prev := img.Pix[y0*img.Width+x0]
	stateIndex := 0
	for i := 1; stateIndex <= stateMiddle; i++ {
		cur, ok := at(i, -1)
		if !ok {
			break
		}
		if cur == prev {
			state := stateMiddle - stateIndex
			stateCount[state]++
			if state > 0 {
				if inside++; inside >= insideLimit {
					return false
				}
			}
			continue
		}
		if stateIndex > 0 && stateCount[stateMiddle-stateIndex] < slack {
			stateCount[stateMiddle-(stateIndex-1)] += stateCount[stateMiddle-stateIndex]
			stateCount[stateMiddle-stateIndex] = 0
			stateIndex--
			stateCount[stateMiddle-stateIndex]++
			if inside = stateCount[1] + stateCount[2] + stateCount[3]; inside >= insideLimit {
				return false
			}
		} else {
			stateIndex++
			if stateIndex > stateMiddle {
				break
			}
			stateCount[stateMiddle-stateIndex]++
			if stateMiddle-stateIndex > 0 {
				if inside++; inside >= insideLimit {
					return false
				}
			}
		}
		prev = cur
	}
	if stateIndex < stateMiddle {
		return false
	}

	stateIndex = 0
	prev = img.Pix[y0*img.Width+x0]
	last := 0
	for i := 1; stateIndex <= stateMiddle; i++ {
		cur, ok := at(i, +1)
		if !ok {
			break
		}
		last = i
		if cur == prev {
			state := stateMiddle + stateIndex
			stateCount[state]++
			if state < 4 {
				if inside++; inside >= insideLimit {
					return false
				}
			}
			continue
		}
		if stateIndex > 0 && stateCount[stateMiddle+stateIndex] < slack {
			stateCount[stateMiddle+(stateIndex-1)] += stateCount[stateMiddle+stateIndex]
			stateCount[stateMiddle+stateIndex] = 0
			stateIndex--
			stateCount[stateMiddle+stateIndex]++
			if inside = stateCount[1] + stateCount[2] + stateCount[3]; inside >= insideLimit {
				return false
			}
		} else {
			stateIndex++
			if stateIndex > stateMiddle {
				break
			}
			stateCount[stateMiddle+stateIndex]++
			if stateMiddle+stateIndex < 4 {
				if inside++; inside >= insideLimit {
					return false
				}
			}
		}
		prev = cur
	}
	if stateIndex < stateMiddle {
		return false
	}

	ms, ok := checkPatternCross(stateCount)
	if !ok {
		return false
	}
	ms *= dir.pxPerSample
	if ms > moduleSizeMax {
		return false
	}
	*moduleSize = ms
	// The refined centre is the midpoint of the middle run, expressed back in
	// image coordinates along the walk direction.
	offset := float64(last-stateCount[4]-stateCount[3]) - float64(stateCount[2])/2.0
	centre.X = sx.X + offset*dir.dx
	centre.Y = sx.Y + offset*dir.dy
	return true
}

// seekPatternAlong scans one line at direction dir, passing through (x0,y0), for
// the next finder run-length signature. It is the directional counterpart of
// seekPatternHorizontal: same five-state window and the same checkPatternCross
// acceptance, walking a line rather than a contiguous row slice.
//
// Sample indices run from start for at most limit steps. A line generally meets
// the frame partway along its span, so leading samples outside the frame are
// skipped rather than ending the scan; the walk ends at the first sample that
// leaves the frame again. That lets a caller sweep lines by perpendicular
// offset without solving each one's entry point.
//
// It reports the signature's centre in image coordinates, its module size in
// physical pixels, and the sample index to resume from.
func seekPatternAlong(img *core.Bitmap, dir scanDirection, x0, y0 float64, start, limit int) (centre core.PointF, moduleSize float64, skip, next int, ok bool) {
	const stateNumber = 5
	curState := 0
	var stateCount [5]int

	sample := func(i int) (byte, bool) {
		x := int(x0 + float64(i)*dir.dx)
		y := int(y0 + float64(i)*dir.dy)
		if x < 0 || x >= img.Width || y < 0 || y >= img.Height {
			return 0, false
		}
		return img.Pix[y*img.Width+x], true
	}

	end := start + limit
	first := start
	var prev byte
	entered := false
	for ; first < end; first++ {
		v, inFrame := sample(first)
		if inFrame {
			prev, entered = v, true
			break
		}
	}
	if !entered {
		return centre, 0, 0, end, false
	}
	stateCount[curState]++
	for i := first + 1; i < end; i++ {
		cur, inFrame := sample(i)
		if !inFrame {
			return centre, 0, 0, i, false
		}
		if cur == prev {
			stateCount[curState]++
			continue
		}
		prev = cur
		if curState < stateNumber-1 {
			if stateCount[curState] < 3 {
				if curState == 0 {
					stateCount[curState] = 1
				} else {
					stateCount[curState-1] += stateCount[curState]
					stateCount[curState] = 0
					curState--
					stateCount[curState]++
				}
			} else {
				curState++
				stateCount[curState]++
			}
			continue
		}
		if stateCount[curState] < 3 {
			stateCount[curState-1] += stateCount[curState]
			stateCount[curState] = 0
			curState--
			stateCount[curState]++
			continue
		}
		if ms, hit := checkPatternCross(stateCount); hit {
			offset := float64(i-stateCount[4]-stateCount[3]) - float64(stateCount[2])/2.0
			centre = core.PointF{X: x0 + offset*dir.dx, Y: y0 + offset*dir.dy}
			return centre, ms * dir.pxPerSample, stateCount[0], i, true
		}
		for k := range stateNumber - 1 {
			stateCount[k] = stateCount[k+1]
		}
		stateCount[stateNumber-1] = 1
		curState = stateNumber - 1
	}
	return centre, 0, 0, end, false
}
