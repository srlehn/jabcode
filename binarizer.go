package jabcode

import "math"

// Block sizes for local binarization (binarizer.c).
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
// number of peaks (3 for the green channel, else 2) — isBiTrimodal in binarizer.c.
func isBiTrimodal(hist []float64, channel int) bool {
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
// the valley between peaks as the threshold (getMinimumThreshold in binarizer.c).
func getMinimumThreshold(hist []int, channel int) int {
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
func newBinary(src *bitmap) *bitmap { return newBitmap(src.width, src.height, 1) }

// binarizerHist binarizes a channel using global histogram thresholding, with
// color-aware pixel skipping for the green/blue channels (binarizerHist).
func binarizerHist(bm *bitmap, channel int) *bitmap {
	binary := newBinary(bm)
	bpp := bm.channels

	hist := make([]int, 256)
	for i := 0; i < bm.width*bm.height; i++ {
		if channel > 0 {
			r := bm.pix[i*bpp]
			g := bm.pix[i*bpp+1]
			b := bm.pix[i*bpp+2]
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
		hist[bm.pix[i*bpp+channel]]++
	}

	ths := getMinimumThreshold(hist, channel)
	for i := 0; i < bm.width*bm.height; i++ {
		if int(bm.pix[i*bpp+channel]) > ths {
			binary.pix[i] = 255
		}
	}
	return binary
}

// binarizerHard binarizes a channel with a fixed threshold (binarizerHard).
func binarizerHard(bm *bitmap, channel, threshold int) *bitmap {
	binary := newBinary(bm)
	bpp := bm.channels
	for i := 0; i < bm.width*bm.height; i++ {
		if int(bm.pix[i*bpp+channel]) > threshold {
			binary.pix[i] = 255
		}
	}
	return binary
}

// calculateBlackPoints computes a per-block black-point estimate (binarizer.c).
func calculateBlackPoints(bm *bitmap, channel, subWidth, subHeight int, blackPoints []byte) {
	const minDynamicRange = 24
	bpp := bm.channels
	bytesPerRow := bm.width * bpp

	for y := range subHeight {
		yoffset := y << blockSizePower
		if max := bm.height - blockSize; yoffset > max {
			yoffset = max
		}
		for x := range subWidth {
			xoffset := x << blockSizePower
			if max := bm.width - blockSize; xoffset > max {
				xoffset = max
			}
			sum, lo, hi := 0, 0xFF, 0
			for yy := 0; yy < blockSize; yy++ {
				for xx := range blockSize {
					p := int(bm.pix[(yoffset+yy)*bytesPerRow+(xoffset+xx)*bpp+channel])
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
							sum += int(bm.pix[(yoffset+yy)*bytesPerRow+(xoffset+xx)*bpp+channel])
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
// points (binarizer.c).
func getBinaryBitmap(bm *bitmap, channel, subWidth, subHeight int, blackPoints []byte, binary *bitmap) {
	bpp := bm.channels
	bytesPerRow := bm.width * bpp
	for y := range subHeight {
		yoffset := y << blockSizePower
		if max := bm.height - blockSize; yoffset > max {
			yoffset = max
		}
		for x := range subWidth {
			xoffset := x << blockSizePower
			if max := bm.width - blockSize; xoffset > max {
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
					if int(bm.pix[(yoffset+yy)*bytesPerRow+(xoffset+xx)*bpp+channel]) > average {
						binary.pix[(yoffset+yy)*binary.width+(xoffset+xx)] = 255
					}
				}
			}
		}
	}
}

// filterBinary removes salt-and-pepper noise with a separable 5-tap majority
// filter (filterBinary in binarizer.c).
func filterBinary(binary *bitmap) {
	w, h := binary.width, binary.height
	const halfSize = 2
	tmp := make([]byte, w*h)

	copy(tmp, binary.pix)
	for i := halfSize; i < h-halfSize; i++ {
		for j := halfSize; j < w-halfSize; j++ {
			sum := b2i(tmp[i*w+j] > 0)
			for k := 1; k <= halfSize; k++ {
				sum += b2i(tmp[i*w+(j-k)] > 0) + b2i(tmp[i*w+(j+k)] > 0)
			}
			binary.pix[i*w+j] = b2byte(sum > halfSize)
		}
	}
	copy(tmp, binary.pix)
	for i := halfSize; i < h-halfSize; i++ {
		for j := halfSize; j < w-halfSize; j++ {
			sum := b2i(tmp[i*w+j] > 0)
			for k := 1; k <= halfSize; k++ {
				sum += b2i(tmp[(i-k)*w+j] > 0) + b2i(tmp[(i+k)*w+j] > 0)
			}
			binary.pix[i*w+j] = b2byte(sum > halfSize)
		}
	}
}

func b2byte(b bool) byte {
	if b {
		return 255
	}
	return 0
}

// binarizer binarizes a channel with local thresholding, falling back to global
// histogram thresholding for small images (binarizer in binarizer.c).
func binarizer(bm *bitmap, channel int) *bitmap {
	if bm.width < minimumDimension || bm.height < minimumDimension {
		return binarizerHist(bm, channel)
	}
	subWidth := bm.width >> blockSizePower
	if subWidth&blockSizeMask != 0 {
		subWidth++
	}
	subHeight := bm.height >> blockSizePower
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
func getHistogram(bm *bitmap, channel int) []int {
	hist := make([]int, 256)
	bpp := bm.channels
	for i := 0; i < bm.width*bm.height; i++ {
		hist[bm.pix[i*bpp+channel]]++
	}
	return hist
}

// getHistMaxMin returns the smallest and largest bin indices whose count exceeds
// ths (getHistMaxMin in binarizer.c).
func getHistMaxMin(hist []int, ths int) (min, max int) {
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

// balanceRGB stretches each channel's histogram to the full 0..255 range,
// in place (balanceRGB in binarizer.c).
func balanceRGB(bm *bitmap) {
	bpp := bm.channels
	bytesPerRow := bm.width * bpp
	const countThs = 20

	minMax := [3][2]int{}
	for c := range 3 {
		lo, hi := getHistMaxMin(getHistogram(bm, c), countThs)
		minMax[c] = [2]int{lo, hi}
	}
	for i := 0; i < bm.height; i++ {
		for j := 0; j < bm.width; j++ {
			offset := i*bytesPerRow + j*bpp
			for c := range 3 {
				lo, hi := minMax[c][0], minMax[c][1]
				v := int(bm.pix[offset+c])
				switch {
				case v < lo:
					bm.pix[offset+c] = 0
				case v > hi:
					bm.pix[offset+c] = 255
				default:
					bm.pix[offset+c] = byte(float64(v-lo) / float64(hi-lo) * 255.0)
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
// original channel indices (getMinMax in binarizer.c).
func getMinMax(rgb []byte) (min, mid, max byte, iMin, iMid, iMax int) {
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

// binarizerRGB binarizes the image into three channel bitmaps using per-pixel
// color analysis (binarizerRGB in binarizer.c). When blkThs is nil, per-block
// average values are used as the black thresholds.
func binarizerRGB(bm *bitmap, blkThs []float32) [3]*bitmap {
	var rgb [3]*bitmap
	for i := range rgb {
		rgb[i] = newBinary(bm)
	}
	bpp := bm.channels
	bytesPerRow := bm.width * bpp

	maxBlockSize := max(bm.width, bm.height) / 2
	blockNumX := bm.width / maxBlockSize
	if bm.width%maxBlockSize != 0 {
		blockNumX++
	}
	blockNumY := bm.height / maxBlockSize
	if bm.height%maxBlockSize != 0 {
		blockNumY++
	}
	blockSizeX := bm.width / blockNumX
	blockSizeY := bm.height / blockNumY
	pixelAvg := make([][3]float64, blockNumX*blockNumY)

	if blkThs == nil {
		for i := 0; i < blockNumY; i++ {
			for j := 0; j < blockNumX; j++ {
				bi := i*blockNumX + j
				sx := j * blockSizeX
				ex := sx + blockSizeX
				if j == blockNumX-1 {
					ex = bm.width
				}
				sy := i * blockSizeY
				ey := sy + blockSizeY
				if i == blockNumY-1 {
					ey = bm.height
				}
				counter := 0
				for y := sy; y < ey; y++ {
					for x := sx; x < ex; x++ {
						offset := y*bytesPerRow + x*bpp
						pixelAvg[bi][0] += float64(bm.pix[offset+0])
						pixelAvg[bi][1] += float64(bm.pix[offset+1])
						pixelAvg[bi][2] += float64(bm.pix[offset+2])
						counter++
					}
				}
				for c := range 3 {
					pixelAvg[bi][c] /= float64(counter)
				}
			}
		}
	}

	const thsStd = 0.08
	for i := 0; i < bm.height; i++ {
		for j := 0; j < bm.width; j++ {
			offset := i*bytesPerRow + j*bpp
			var ths [3]float64
			if blkThs == nil {
				bi := min(i/blockSizeY, blockNumY-1)*blockNumX + min(j/blockSizeX, blockNumX-1)
				ths = pixelAvg[bi]
			} else {
				ths = [3]float64{float64(blkThs[0]), float64(blkThs[1]), float64(blkThs[2])}
			}
			pix := bm.pix[offset : offset+3]
			if float64(pix[0]) < ths[0] && float64(pix[1]) < ths[1] && float64(pix[2]) < ths[2] {
				continue // black pixel: all channels 0
			}
			_, variance := getAvgVar(pix)
			std := math.Sqrt(variance)
			_, _, mx, iMin, iMid, iMax := getMinMax(pix)
			std /= float64(mx)

			idx := i*bm.width + j
			if std < thsStd && float64(pix[0]) > ths[0] && float64(pix[1]) > ths[1] && float64(pix[2]) > ths[2] {
				rgb[0].pix[idx] = 255
				rgb[1].pix[idx] = 255
				rgb[2].pix[idx] = 255
			} else {
				rgb[iMax].pix[idx] = 255
				rgb[iMin].pix[idx] = 0
				r1 := float64(pix[iMid]) / float64(pix[iMin])
				r2 := float64(pix[iMax]) / float64(pix[iMid])
				if r1 > r2 {
					rgb[iMid].pix[idx] = 255
				} else {
					rgb[iMid].pix[idx] = 0
				}
			}
		}
	}
	filterBinary(rgb[0])
	filterBinary(rgb[1])
	filterBinary(rgb[2])
	return rgb
}
