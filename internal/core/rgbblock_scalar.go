//go:build !goexperiment.simd || (!amd64 && !(go1.27 && (arm64 || wasm)))

package core

func rgbBlockSums(pix []byte, width, channels, sx, ex, sy, ey int) (sum [3]float64) {
	for y := sy; y < ey; y++ {
		row := y * width * channels
		for x := sx; x < ex; x++ {
			o := row + x*channels
			for c := range 3 {
				sum[c] += float64(pix[o+c])
			}
		}
	}
	return sum
}
