//go:build goexperiment.simd && go1.27 && (amd64 || arm64 || wasm)

package core

import "simd/archsimd"

func majority5VerticalRow(rows [5][]byte, dst []byte, width int) {
	const lanes = 16
	const radius = 2
	limit := width - radius
	j := radius
	for ; j+lanes <= limit; j += lanes {
		a := archsimd.LoadUint8x16(rows[0][j:])
		b := archsimd.LoadUint8x16(rows[1][j:])
		c := archsimd.LoadUint8x16(rows[2][j:])
		d := archsimd.LoadUint8x16(rows[3][j:])
		e := archsimd.LoadUint8x16(rows[4][j:])
		majority := majority5(a, b, c, d, e)
		majority.Store(dst[j:])
	}
	for ; j < limit; j++ {
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
