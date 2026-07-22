//go:build !goexperiment.simd || (!go1.27 && !amd64) || (go1.27 && !(amd64 || arm64 || wasm))

package core

func rgbBlockStats(pix []byte, width, channels, sx, ex, sy, ey int) (lo, hi [3]int, sum [3]float64, n int) {
	lo = [3]int{255, 255, 255}
	for y := sy; y < ey; y++ {
		row := y * width * channels
		for x := sx; x < ex; x++ {
			o := row + x*channels
			for c := range 3 {
				v := int(pix[o+c])
				lo[c] = min(lo[c], v)
				hi[c] = max(hi[c], v)
				sum[c] += float64(v)
			}
			n++
		}
	}
	return lo, hi, sum, n
}
