package decode

import "math"

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

// getMinimumThreshold smooths a histogram until it is bi/tri-modal, then returns
// the valley between peaks as the threshold.
func getMinimumThreshold(hist []int, channel int) int {
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
func newBinary(src *Bitmap) *Bitmap { return NewBitmap(src.Width, src.Height, 1) }

// binarizerHist binarizes a channel using global histogram thresholding, with
// color-aware pixel skipping for the green/blue channels.
func binarizerHist(bm *Bitmap, channel int) *Bitmap {
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

	ths := getMinimumThreshold(hist, channel)
	for i := 0; i < bm.Width*bm.Height; i++ {
		if int(bm.Pix[i*bpp+channel]) > ths {
			binary.Pix[i] = 255
		}
	}
	return binary
}

// binarizerHard binarizes a channel with a fixed threshold.
func binarizerHard(bm *Bitmap, channel, threshold int) *Bitmap {
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
func calculateBlackPoints(bm *Bitmap, channel, subWidth, subHeight int, blackPoints []byte) {
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

// getBinaryBitmap thresholds each block against a smoothed neighborhood of black
// points.
func getBinaryBitmap(bm *Bitmap, channel, subWidth, subHeight int, blackPoints []byte, binary *Bitmap) {
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
func filterBinary(binary *Bitmap) {
	// Ports filterBinary in binarizer.c.
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
func binarizer(bm *Bitmap, channel int) *Bitmap {
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
	getBinaryBitmap(bm, channel, subWidth, subHeight, blackPoints, binary)
	filterBinary(binary)
	return binary
}

// getHistogram fills a 256-bin histogram of a channel.
func getHistogram(bm *Bitmap, channel int) []int {
	hist := make([]int, 256)
	bpp := bm.Channels
	for i := 0; i < bm.Width*bm.Height; i++ {
		hist[bm.Pix[i*bpp+channel]]++
	}
	return hist
}

// getHistMaxMin returns the smallest and largest bin indices whose count exceeds
// ths.
func getHistMaxMin(hist []int, ths int) (min, max int) {
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
func BalanceRGB(bm *Bitmap) {
	// Ports BalanceRGB in binarizer.c.
	bpp := bm.Channels
	bytesPerRow := bm.Width * bpp
	const countThs = 20

	minMax := [3][2]int{}
	for c := range 3 {
		lo, hi := getHistMaxMin(getHistogram(bm, c), countThs)
		minMax[c] = [2]int{lo, hi}
	}
	for i := 0; i < bm.Height; i++ {
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

// getAvgVar returns the mean and variance of a pixel's RGB values.
func getAvgVar(rgb []byte) (avg, variance float64) {
	avg = float64(int(rgb[0])+int(rgb[1])+int(rgb[2])) / 3
	sum := 0.0
	for i := range 3 {
		d := float64(rgb[i]) - avg
		sum += d * d
	}
	return avg, sum / 3
}

// getMinMax orders a pixel's three channels, returning the values and their
// original channel indices.
func getMinMax(rgb []byte) (min, mid, max byte, iMin, iMid, iMax int) {
	// Ports getMinMax in binarizer.c.
	iMin, iMid, iMax = 0, 1, 2
	if rgb[iMin] > rgb[iMax] {
		iMin, iMax = iMax, iMin
	}
	if rgb[iMin] > rgb[iMid] {
		iMin, iMid = iMid, iMin
	}
	if rgb[iMid] > rgb[iMax] {
		iMid, iMax = iMax, iMid
	}
	return rgb[iMin], rgb[iMid], rgb[iMax], iMin, iMid, iMax
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

// blockMeans returns the per-channel mean of each bs-sized block of bm as an
// nbx*nby grid (row-major, three means per cell), used as a local black threshold.
func blockMeans(bm *Bitmap, bs int) (grid [][3]float64, nbx, nby int) {
	w, h, bpp := bm.Width, bm.Height, bm.Channels
	bytesPerRow := w * bpp
	nbx = (w + bs - 1) / bs
	nby = (h + bs - 1) / bs
	grid = make([][3]float64, nbx*nby)
	for by := range nby {
		sy, ey := by*bs, min((by+1)*bs, h)
		for bx := range nbx {
			sx, ex := bx*bs, min((bx+1)*bs, w)
			var sum [3]float64
			n := 0
			for y := sy; y < ey; y++ {
				row := y * bytesPerRow
				for x := sx; x < ex; x++ {
					o := row + x*bpp
					sum[0] += float64(bm.Pix[o+0])
					sum[1] += float64(bm.Pix[o+1])
					sum[2] += float64(bm.Pix[o+2])
					n++
				}
			}
			grid[by*nbx+bx] = [3]float64{sum[0] / float64(n), sum[1] / float64(n), sum[2] / float64(n)}
		}
	}
	return grid, nbx, nby
}

// sampleGrid bilinearly interpolates the block-mean grid at pixel (x,y), so the
// local threshold varies smoothly instead of jumping at block boundaries (block
// centres sit at integer block indices).
func sampleGrid(grid [][3]float64, nbx, nby, bs, x, y int) [3]float64 {
	fx := (float64(x)+0.5)/float64(bs) - 0.5
	fy := (float64(y)+0.5)/float64(bs) - 0.5
	x0 := int(math.Floor(fx))
	y0 := int(math.Floor(fy))
	tx, ty := fx-float64(x0), fy-float64(y0)
	x0c, x1c := capInt(x0, 0, nbx-1), capInt(x0+1, 0, nbx-1)
	y0c, y1c := capInt(y0, 0, nby-1), capInt(y0+1, 0, nby-1)
	var out [3]float64
	for c := range 3 {
		top := grid[y0c*nbx+x0c][c] + (grid[y0c*nbx+x1c][c]-grid[y0c*nbx+x0c][c])*tx
		bot := grid[y1c*nbx+x0c][c] + (grid[y1c*nbx+x1c][c]-grid[y1c*nbx+x0c][c])*tx
		out[c] = top + (bot-top)*ty
	}
	return out
}

// BinarizerRGB binarizes the image into three channel bitmaps using per-pixel
// color analysis. When blkThs is nil, a scale-adaptive grid of
// bilinearly-interpolated per-channel block means is used as the local black
// threshold; otherwise blkThs is a flat per-channel threshold.
func BinarizerRGB(bm *Bitmap, blkThs []float32) [3]*Bitmap {
	// Ports BinarizerRGB in binarizer.c.
	var rgb [3]*Bitmap
	for i := range rgb {
		rgb[i] = newBinary(bm)
	}
	bpp := bm.Channels
	bytesPerRow := bm.Width * bpp

	var grid [][3]float64
	var nbx, nby, bs int
	if blkThs == nil {
		bs = capInt(min(bm.Width, bm.Height)/binThresholdDivisor, binMinBlock, binMaxBlock)
		grid, nbx, nby = blockMeans(bm, bs)
	}

	const thsStd = 0.08
	for i := 0; i < bm.Height; i++ {
		for j := 0; j < bm.Width; j++ {
			offset := i*bytesPerRow + j*bpp
			var ths [3]float64
			if blkThs == nil {
				ths = sampleGrid(grid, nbx, nby, bs, j, i)
			} else {
				ths = [3]float64{float64(blkThs[0]), float64(blkThs[1]), float64(blkThs[2])}
			}
			pix := bm.Pix[offset : offset+3]
			if float64(pix[0]) < ths[0] && float64(pix[1]) < ths[1] && float64(pix[2]) < ths[2] {
				continue // black pixel: all channels 0
			}
			_, variance := getAvgVar(pix)
			std := math.Sqrt(variance)
			_, _, mx, iMin, iMid, iMax := getMinMax(pix)
			std /= float64(mx)

			idx := i*bm.Width + j
			if std < thsStd && float64(pix[0]) > ths[0] && float64(pix[1]) > ths[1] && float64(pix[2]) > ths[2] {
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
	filterBinary(rgb[0])
	filterBinary(rgb[1])
	filterBinary(rgb[2])
	return rgb
}
