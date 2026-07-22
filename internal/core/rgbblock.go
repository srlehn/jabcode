package core

// RGBBlockStats returns per-channel extrema and sums for a rectangular RGB
// region. The separate sum helper lets SIMD-capable builds amortize vector
// setup over a block while keeping the threshold contract in one place.
func RGBBlockStats(pix []byte, width, channels, sx, ex, sy, ey int) (lo, hi [3]int, sum [3]float64, n int) {
	lo = [3]int{255, 255, 255}
	for y := sy; y < ey; y++ {
		row := y * width * channels
		for x := sx; x < ex; x++ {
			o := row + x*channels
			for c := range 3 {
				v := int(pix[o+c])
				lo[c] = min(lo[c], v)
				hi[c] = max(hi[c], v)
			}
			n++
		}
	}
	sum = rgbBlockSums(pix, width, channels, sx, ex, sy, ey)
	return lo, hi, sum, n
}
