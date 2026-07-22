package detect

import (
	"math"
	"sync"

	"github.com/srlehn/jabcode/internal/core"
)

// Block sizes for local binarization.
const (
	blockSizePower   = 5
	blockSize        = 1 << blockSizePower
	blockSizeMask    = blockSize - 1
	minimumDimension = blockSize * 5
)

func capInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// isBiTrimodal reports whether the smoothed histogram has exactly the expected
// number of peaks (3 for the green channel, else 2).
func isBiTrimodal(hist []float64, channel int) bool {
	// Ports isBiTrimodal in binarizer.c.
	modal := 2
	if channel == 1 {
		modal = 3
	}
	count := 0
	for i := 1; i < 255; i++ {
		if hist[i-1] < hist[i] && hist[i+1] < hist[i] {
			count++
			if count > modal {
				return false
			}
		}
	}
	return count == modal
}

// minimumThreshold smooths a histogram until it is bi/tri-modal, then returns
// the valley between peaks as the threshold.
func minimumThreshold(hist []int, channel int) int {
	// Ports getMinimumThreshold in binarizer.c.
	histC := make([]float64, 256)
	histS := make([]float64, 256)
	for i := range 256 {
		histC[i] = float64(hist[i])
		histS[i] = float64(hist[i])
	}

	for iter := 0; !isBiTrimodal(histS, channel); iter++ {
		histS[0] = (histC[0] + histC[0] + histC[1]) / 3
		for i := 1; i < 255; i++ {
			histS[i] = (histC[i-1] + histC[i] + histC[i+1]) / 3
		}
		histS[255] = (histC[254] + histC[255] + histC[255]) / 3
		copy(histC, histS)
		if iter >= 1000 {
			return -1
		}
	}

	peakNumber := 1
	if channel == 1 {
		peakNumber = 2
	}
	peakFound := 0
	for i := 1; i < 255; i++ {
		if histS[i-1] < histS[i] && histS[i+1] < histS[i] {
			peakFound++
		}
		if peakFound == peakNumber && histS[i-1] >= histS[i] && histS[i+1] >= histS[i] {
			return i - 1
		}
	}
	return -1
}

// newBinary allocates a single-channel binary bitmap of the same size as src.
func newBinary(src *core.Bitmap) *core.Bitmap { return core.NewBitmap(src.Width, src.Height, 1) }

// binarizerHist binarizes a channel using global histogram thresholding, with
// color-aware pixel skipping for the green/blue channels.
func binarizerHist(bm *core.Bitmap, channel int) *core.Bitmap {
	// Ports binarizerHist in binarizer.c.
	binary := newBinary(bm)
	bpp := bm.Channels

	hist := make([]int, 256)
	for i := 0; i < bm.Width*bm.Height; i++ {
		if channel > 0 {
			r := bm.Pix[i*bpp]
			g := bm.Pix[i*bpp+1]
			b := bm.Pix[i*bpp+2]
			mean := float32(int(r)+int(g)+int(b)) / 3
			pr := float32(r) / mean
			pg := float32(g) / mean
			pb := float32(b) / mean
			if channel == 1 { // green
				if (r > 200 && g > 200 && b > 200) || (r < 50 && g < 50 && b < 50) || (r > 200 && g > 200) {
					continue
				}
				if pr < 1.25 && pr > 0.8 && pg < 1.25 && pg > 0.8 && pb < 1.25 && pb > 0.8 {
					continue
				}
				if pb < 0.5 && pr/pg < 1.25 && pr/pg > 0.8 {
					continue
				}
			} else if channel == 2 { // blue
				if (r > 200 && g > 200 && b > 200) || (r < 50 && g < 50 && b < 50) {
					continue
				}
				if pr < 1.25 && pr > 0.8 && pg < 1.25 && pg > 0.8 && pb < 1.25 && pb > 0.8 {
					continue
				}
			}
		}
		hist[bm.Pix[i*bpp+channel]]++
	}

	ths := minimumThreshold(hist, channel)
	for i := 0; i < bm.Width*bm.Height; i++ {
		if int(bm.Pix[i*bpp+channel]) > ths {
			binary.Pix[i] = 255
		}
	}
	return binary
}

// binarizerHard binarizes a channel with a fixed threshold.
func binarizerHard(bm *core.Bitmap, channel, threshold int) *core.Bitmap {
	// Ports binarizerHard in binarizer.c.
	binary := newBinary(bm)
	bpp := bm.Channels
	for i := 0; i < bm.Width*bm.Height; i++ {
		if int(bm.Pix[i*bpp+channel]) > threshold {
			binary.Pix[i] = 255
		}
	}
	return binary
}

// calculateBlackPoints computes a per-block black-point estimate.
func calculateBlackPoints(bm *core.Bitmap, channel, subWidth, subHeight int, blackPoints []byte) {
	// Ports calculateBlackPoints in binarizer.c.
	const minDynamicRange = 24
	bpp := bm.Channels
	bytesPerRow := bm.Width * bpp

	for y := range subHeight {
		yoffset := y << blockSizePower
		if max := bm.Height - blockSize; yoffset > max {
			yoffset = max
		}
		for x := range subWidth {
			xoffset := x << blockSizePower
			if max := bm.Width - blockSize; xoffset > max {
				xoffset = max
			}
			sum, lo, hi := 0, 0xFF, 0
			for yy := 0; yy < blockSize; yy++ {
				for xx := range blockSize {
					p := int(bm.Pix[(yoffset+yy)*bytesPerRow+(xoffset+xx)*bpp+channel])
					sum += p
					if p < lo {
						lo = p
					}
					if p > hi {
						hi = p
					}
				}
				if hi-lo > minDynamicRange {
					for yy++; yy < blockSize; yy++ {
						for xx := range blockSize {
							sum += int(bm.Pix[(yoffset+yy)*bytesPerRow+(xoffset+xx)*bpp+channel])
						}
					}
				}
			}
			average := sum >> (blockSizePower * 2)
			if hi-lo <= minDynamicRange { // smooth block
				average = lo / 2
				if y > 0 && x > 0 {
					neighbor := (int(blackPoints[(y-1)*subWidth+x]) +
						2*int(blackPoints[y*subWidth+x-1]) +
						int(blackPoints[(y-1)*subWidth+x-1])) / 4
					if lo < neighbor {
						average = neighbor
					}
				}
			}
			blackPoints[y*subWidth+x] = byte(average)
		}
	}
}

// fillBinaryBitmap thresholds each block against a smoothed neighborhood of black
// points.
func fillBinaryBitmap(bm *core.Bitmap, channel, subWidth, subHeight int, blackPoints []byte, binary *core.Bitmap) {
	// Ports getBinaryBitmap in binarizer.c.
	bpp := bm.Channels
	bytesPerRow := bm.Width * bpp
	for y := range subHeight {
		yoffset := y << blockSizePower
		if max := bm.Height - blockSize; yoffset > max {
			yoffset = max
		}
		for x := range subWidth {
			xoffset := x << blockSizePower
			if max := bm.Width - blockSize; xoffset > max {
				xoffset = max
			}
			left := capInt(x, 2, subWidth-3)
			top := capInt(y, 2, subHeight-3)
			sum := 0
			for z := -2; z <= 2; z++ {
				row := blackPoints[(top+z)*subWidth:]
				sum += int(row[left-2]) + int(row[left-1]) + int(row[left]) + int(row[left+1]) + int(row[left+2])
			}
			average := sum / 25
			for yy := range blockSize {
				for xx := range blockSize {
					if int(bm.Pix[(yoffset+yy)*bytesPerRow+(xoffset+xx)*bpp+channel]) > average {
						binary.Pix[(yoffset+yy)*binary.Width+(xoffset+xx)] = 255
					}
				}
			}
		}
	}
}

// filterBinary removes salt-and-pepper noise with a separable 5-tap majority
// filter.
// filterBinary applies the two 5-tap majority passes that clean the binarized
// channel. Neighbouring outputs share all but one tap of the kernel, so each
// pass carries a running window count and advances it by one add and one
// subtract per pixel instead of re-reading the whole kernel; the vertical pass
// keeps that count per column so it walks whole rows instead of striding the
// image once per tap. The counts are integers, so the result is byte-identical
// to the direct form.
func filterBinary(binary *core.Bitmap) {
	// Ports filterBinary in binarizer.c.
	w, h := binary.Width, binary.Height
	const halfSize = 2
	const window = 2*halfSize + 1
	if w < window || h < window {
		return // no interior pixel has a full kernel
	}
	tmp := make([]byte, w*h)

	interior := h - 2*halfSize

	copy(tmp, binary.Pix)
	core.ParallelRows(interior, func(lo, hi int) {
		for i := lo + halfSize; i < hi+halfSize; i++ {
			base := i * w
			sum := 0
			for t := range window {
				sum += b2i(tmp[base+t] > 0)
			}
			binary.Pix[base+halfSize] = b2byte(sum > halfSize)
			for j := halfSize + 1; j < w-halfSize; j++ {
				sum += b2i(tmp[base+j+halfSize] > 0) - b2i(tmp[base+j-halfSize-1] > 0)
				binary.Pix[base+j] = b2byte(sum > halfSize)
			}
		}
	})
	copy(tmp, binary.Pix)
	core.ParallelRows(interior, func(lo, hi int) {
		// A column count never exceeds the window, so it fits a byte.
		colSum := make([]uint8, w)
		first := lo + halfSize
		for j := halfSize; j < w-halfSize; j++ {
			sum := 0
			for r := first - halfSize; r <= first+halfSize; r++ {
				sum += b2i(tmp[r*w+j] > 0)
			}
			colSum[j] = uint8(sum)
			binary.Pix[first*w+j] = b2byte(sum > halfSize)
		}
		for i := first + 1; i < hi+halfSize; i++ {
			add, sub := (i+halfSize)*w, (i-halfSize-1)*w
			for j := halfSize; j < w-halfSize; j++ {
				sum := int(colSum[j]) + b2i(tmp[add+j] > 0) - b2i(tmp[sub+j] > 0)
				colSum[j] = uint8(sum)
				binary.Pix[i*w+j] = b2byte(sum > halfSize)
			}
		}
	})
}

func b2byte(b bool) byte {
	if b {
		return 255
	}
	return 0
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

// binarizer binarizes a channel with local thresholding, falling back to global
// histogram thresholding for small images.
func binarizer(bm *core.Bitmap, channel int) *core.Bitmap {
	// Ports binarizer in binarizer.c.
	if bm.Width < minimumDimension || bm.Height < minimumDimension {
		return binarizerHist(bm, channel)
	}
	subWidth := bm.Width >> blockSizePower
	if subWidth&blockSizeMask != 0 {
		subWidth++
	}
	subHeight := bm.Height >> blockSizePower
	if subHeight&blockSizeMask != 0 {
		subHeight++
	}
	blackPoints := make([]byte, subWidth*subHeight)
	calculateBlackPoints(bm, channel, subWidth, subHeight, blackPoints)
	binary := newBinary(bm)
	fillBinaryBitmap(bm, channel, subWidth, subHeight, blackPoints, binary)
	filterBinary(binary)
	return binary
}

// histogram fills a 256-bin histogram of a channel. Chunks count locally and
// merge under a lock; integer sums commute, so the result is order-independent.
func histogram(bm *core.Bitmap, channel int) []int {
	hist := make([]int, 256)
	bpp := bm.Channels
	var mu sync.Mutex
	core.ParallelChunks(bm.Width*bm.Height, 1<<16, func(lo, hi int) {
		var local [256]int
		for i := lo; i < hi; i++ {
			local[bm.Pix[i*bpp+channel]]++
		}
		mu.Lock()
		for b, v := range local {
			hist[b] += v
		}
		mu.Unlock()
	})
	return hist
}

// histMaxMin returns the smallest and largest bin indices whose count exceeds
// ths.
func histMaxMin(hist []int, ths int) (min, max int) {
	// Ports getHistMaxMin in binarizer.c.
	for i := range 256 {
		if hist[i] > ths {
			min = i
			break
		}
	}
	max = 255
	for i := 255; i >= 0; i-- {
		if hist[i] > ths {
			max = i
			break
		}
	}
	return min, max
}

// BalanceRGB stretches each channel's histogram to the full 0..255 range, in
// place.
func BalanceRGB(bm *core.Bitmap) {
	// Ports balanceRGB in binarizer.c.
	bpp := bm.Channels
	bytesPerRow := bm.Width * bpp
	const countThs = 20

	minMax := [3][2]int{}
	for c := range 3 {
		lo, hi := histMaxMin(histogram(bm, c), countThs)
		minMax[c] = [2]int{lo, hi}
	}
	// The stretch is a pure function of one input byte per channel, so the 256
	// possible results are tabulated once with the same expression instead of
	// recomputed per pixel. The table is built from the identical branches and
	// arithmetic, so every output byte - including the degenerate empty-range
	// case a uniform frame produces - matches the per-pixel form exactly.
	var lut [3][256]byte
	for c := range 3 {
		lo, hi := minMax[c][0], minMax[c][1]
		for v := range 256 {
			switch {
			case v < lo:
				lut[c][v] = 0
			case v > hi:
				lut[c][v] = 255
			default:
				lut[c][v] = byte(float64(v-lo) / float64(hi-lo) * 255.0)
			}
		}
	}
	core.ParallelRows(bm.Height, func(rlo, rhi int) {
		for i := rlo; i < rhi; i++ {
			row := bm.Pix[i*bytesPerRow : i*bytesPerRow+bm.Width*bpp]
			for j := 0; j < len(row); j += bpp {
				row[j+0] = lut[0][row[j+0]]
				row[j+1] = lut[1][row[j+1]]
				row[j+2] = lut[2][row[j+2]]
			}
		}
	})
}

// Local-threshold block grid for BinarizerRGB. Each block is about
// min(width,height)/binThresholdDivisor pixels, clamped to [binMinBlock,
// binMaxBlock], so the grid scales with the image rather than using a fixed pixel
// size. The floor keeps blocks well wider than a module on small clean encodes (so
// a block average is a meaningful black/colour threshold, not a single module's
// value); on a large photographed screen the grid is fine enough to track
// vignetting and colour cast.
const (
	binThresholdDivisor = 24
	binMinBlock         = 64
	binMaxBlock         = 512
)

// blackAnchorFrac positions the black/colour decision level within each
// block's per-channel dynamic range. The anchor must sit between the block's
// black level and its darkest ink level: saturated print colours are dark
// (subtractive gamut - a printed blue's own channel can fall below the block
// mean), so a mean-based threshold classifies whole colour modules as black.
// minBlockDynamicRange matches calculateBlackPoints: a block without that
// much range is flat (paper, margins) and anchors at half its floor so its
// pixels never classify as black.
const (
	blackAnchorFrac      = 0.25
	minBlockDynamicRange = 24
)

// blockThresholds returns two per-channel grids over the bs-sized blocks of
// bm (row-major, three values per cell): the black anchor of each block (the
// block minimum plus blackAnchorFrac of its dynamic range, or half the
// minimum for flat blocks) and the block mean. The print retry's black gate
// compares against the anchor; the default black gate and the white gate use
// the mean.
func blockThresholds(bm *core.Bitmap, bs int) (anchors, means [][3]float64, nbx, nby int) {
	w, h, bpp := bm.Width, bm.Height, bm.Channels
	nbx = (w + bs - 1) / bs
	nby = (h + bs - 1) / bs
	anchors = make([][3]float64, nbx*nby)
	means = make([][3]float64, nbx*nby)
	core.ParallelChunks(nby, 1, func(blo, bhi int) {
		for by := blo; by < bhi; by++ {
			sy, ey := by*bs, min((by+1)*bs, h)
			for bx := range nbx {
				sx, ex := bx*bs, min((bx+1)*bs, w)
				lo, hi, sum, n := core.RGBBlockStats(bm.Pix, w, bpp, sx, ex, sy, ey)
				var anchor, mean [3]float64
				for c := range 3 {
					if hi[c]-lo[c] < minBlockDynamicRange {
						anchor[c] = float64(lo[c]) / 2
					} else {
						anchor[c] = float64(lo[c]) + blackAnchorFrac*float64(hi[c]-lo[c])
					}
					mean[c] = sum[c] / float64(n)
				}
				anchors[by*nbx+bx] = anchor
				means[by*nbx+bx] = mean
			}
		}
	})
	return anchors, means, nbx, nby
}

// BinarizerRGB binarizes the image into three channel bitmaps using per-pixel
// color analysis. When blkThs is nil, a scale-adaptive grid of
// bilinearly-interpolated per-channel block means is used as the local black
// threshold; otherwise blkThs is a flat per-channel threshold.
func BinarizerRGB(bm *core.Bitmap, blkThs []float32) [3]*core.Bitmap {
	return binarizeRGB(bm, blkThs, false)
}

// BinarizerRGBPrint binarizes at print levels for the detector's print
// retry: the black gate tests against each block's black anchor instead of
// its mean, so dark saturated print colours (subtractive gamut) classify as
// colour rather than black. Not the default because the two level choices
// genuinely conflict: heavy blur lifts true black into the same value band
// where print inks live, and a half-blurred black pixel next to a yellow
// module is ratio-identical to a dark printed yellow - only the retry
// ladder's evidence separates the two regimes image-wide.
func BinarizerRGBPrint(bm *core.Bitmap) [3]*core.Bitmap {
	return binarizeRGB(bm, nil, true)
}

func binarizeRGB(bm *core.Bitmap, blkThs []float32, printLevels bool) [3]*core.Bitmap {
	// Ports binarizerRGB in binarizer.c.
	var rgb [3]*core.Bitmap
	for i := range rgb {
		rgb[i] = newBinary(bm)
	}
	bpp := bm.Channels
	bytesPerRow := bm.Width * bpp

	// The per-pixel local threshold interpolates the block grids bilinearly,
	// so it varies smoothly instead of jumping at block boundaries (block
	// centres sit at integer block indices). The x-axis floor, division and
	// caps are hoisted into one per-column table - identical values, so the
	// interpolated thresholds stay bit-identical to the per-pixel form.
	var anchors, means [][3]float64
	var nbx, nby, bs int
	var colX0, colX1 []int
	var colTx []float64
	if blkThs == nil {
		bs = capInt(min(bm.Width, bm.Height)/binThresholdDivisor, binMinBlock, binMaxBlock)
		anchors, means, nbx, nby = blockThresholds(bm, bs)
		colX0 = make([]int, bm.Width)
		colX1 = make([]int, bm.Width)
		colTx = make([]float64, bm.Width)
		for j := range colTx {
			fx := (float64(j)+0.5)/float64(bs) - 0.5
			x0 := int(math.Floor(fx))
			colTx[j] = fx - float64(x0)
			colX0[j] = capInt(x0, 0, nbx-1)
			colX1[j] = capInt(x0+1, 0, nbx-1)
		}
	}

	const thsStd = 0.08
	core.ParallelRows(bm.Height, func(rlo, rhi int) {
		for i := rlo; i < rhi; i++ {
			var rowTopM, rowBotM, rowTopA, rowBotA [][3]float64
			var ty float64
			if blkThs == nil {
				fy := (float64(i)+0.5)/float64(bs) - 0.5
				y0 := int(math.Floor(fy))
				ty = fy - float64(y0)
				y0c := capInt(y0, 0, nby-1)
				y1c := capInt(y0+1, 0, nby-1)
				rowTopM = means[y0c*nbx : y0c*nbx+nbx]
				rowBotM = means[y1c*nbx : y1c*nbx+nbx]
				if printLevels {
					rowTopA = anchors[y0c*nbx : y0c*nbx+nbx]
					rowBotA = anchors[y1c*nbx : y1c*nbx+nbx]
				}
			}
			for j := 0; j < bm.Width; j++ {
				offset := i*bytesPerRow + j*bpp
				var thsBlack, thsWhite [3]float64
				if blkThs == nil {
					x0, x1, tx := colX0[j], colX1[j], colTx[j]
					for c := range 3 {
						top := rowTopM[x0][c] + (rowTopM[x1][c]-rowTopM[x0][c])*tx
						bot := rowBotM[x0][c] + (rowBotM[x1][c]-rowBotM[x0][c])*tx
						thsWhite[c] = top + (bot-top)*ty
					}
					if printLevels {
						for c := range 3 {
							top := rowTopA[x0][c] + (rowTopA[x1][c]-rowTopA[x0][c])*tx
							bot := rowBotA[x0][c] + (rowBotA[x1][c]-rowBotA[x0][c])*tx
							thsBlack[c] = top + (bot-top)*ty
						}
					} else {
						thsBlack = thsWhite
					}
				} else {
					thsBlack = [3]float64{float64(blkThs[0]), float64(blkThs[1]), float64(blkThs[2])}
					thsWhite = thsBlack
				}
				pix := bm.Pix[offset : offset+3]
				if float64(pix[0]) < thsBlack[0] && float64(pix[1]) < thsBlack[1] && float64(pix[2]) < thsBlack[2] {
					continue // black pixel: all channels 0
				}
				_, variance := core.AvgVar(pix)
				std := math.Sqrt(variance)
				_, _, mx, iMin, iMid, iMax := core.MinMax(pix)
				std /= float64(mx)

				idx := i*bm.Width + j
				if std < thsStd && float64(pix[0]) > thsWhite[0] && float64(pix[1]) > thsWhite[1] && float64(pix[2]) > thsWhite[2] {
					rgb[0].Pix[idx] = 255
					rgb[1].Pix[idx] = 255
					rgb[2].Pix[idx] = 255
				} else {
					rgb[iMax].Pix[idx] = 255
					rgb[iMin].Pix[idx] = 0
					r1 := float64(pix[iMid]) / float64(pix[iMin])
					r2 := float64(pix[iMax]) / float64(pix[iMid])
					if r1 > r2 {
						rgb[iMid].Pix[idx] = 255
					} else {
						rgb[iMid].Pix[idx] = 0
					}
				}
			}
		}
	})
	filterBinary(rgb[0])
	filterBinary(rgb[1])
	filterBinary(rgb[2])
	return rgb
}
