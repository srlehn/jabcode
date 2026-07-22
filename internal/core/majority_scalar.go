//go:build !goexperiment.simd || !go1.27 || !(amd64 || arm64 || wasm)

package core

func majority5Row(src, dst []byte, width int) {
	if width < 5 {
		return
	}
	count := 0
	for _, v := range src[:5] {
		if v != 0 {
			count++
		}
	}
	for center := 2; center < width-2; center++ {
		if count > 2 {
			dst[center] = 255
		} else {
			dst[center] = 0
		}
		if center+3 < width {
			if src[center-2] != 0 {
				count--
			}
			if src[center+3] != 0 {
				count++
			}
		}
	}
}
