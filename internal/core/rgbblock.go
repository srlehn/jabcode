package core

// RGBBlockStats returns per-channel extrema and sums for a rectangular RGB
// region.
func RGBBlockStats(pix []byte, width, channels, sx, ex, sy, ey int) (lo, hi [3]int, sum [3]float64, n int) {
	return rgbBlockStats(pix, width, channels, sx, ex, sy, ey)
}
