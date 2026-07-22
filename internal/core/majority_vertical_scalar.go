//go:build !goexperiment.simd || !go1.27 || !(amd64 || arm64 || wasm)

package core

func majority5VerticalRow(rows [5][]byte, dst []byte, width int) {
	for j := 2; j < width-2; j++ {
		sum := 0
		for _, row := range rows {
			if row[j] != 0 {
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
