//go:build !goexperiment.simd || !go1.27 || !(amd64 || arm64 || wasm)

package core

func majority5Row(src, dst []byte, width int) {
	for j := 2; j < width-2; j++ {
		sum := 0
		for k := -2; k <= 2; k++ {
			if src[j+k] != 0 {
				sum++
			}
		}
		if sum > 2 {
			dst[j] = 255
		} else {
			dst[j] = 0
		}
	}
}
