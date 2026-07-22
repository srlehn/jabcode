//go:build goexperiment.simd && go1.27 && (amd64 || arm64 || wasm)

package core

import "simd/archsimd"

func majority5Row(src, dst []byte, width int) {
	const lanes = 16
	const radius = 2
	var j int
	limit := width - 2*radius
	for ; j+lanes <= limit; j += lanes {
		a := archsimd.LoadUint8x16(src[j:])
		b := archsimd.LoadUint8x16(src[j+1:])
		c := archsimd.LoadUint8x16(src[j+2:])
		d := archsimd.LoadUint8x16(src[j+3:])
		e := archsimd.LoadUint8x16(src[j+4:])
		// Inputs are 0 or 255, so a three-way conjunction is a boolean
		// vote. The ten terms cover every three-of-five combination.
		majority := majority5(a, b, c, d, e)
		center := j + radius
		majority.Store(dst[center:])
	}
	for ; j < limit; j++ {
		center := j + radius
		sum := 0
		for k := -radius; k <= radius; k++ {
			if src[center+k] != 0 {
				sum++
			}
		}
		if sum > 2 {
			dst[center] = 255
		} else {
			dst[center] = 0
		}
	}
}

func majority5(a, b, c, d, e archsimd.Uint8x16) archsimd.Uint8x16 {
	m := a.And(b).And(c)
	m = m.Or(a.And(b).And(d))
	m = m.Or(a.And(b).And(e))
	m = m.Or(a.And(c).And(d))
	m = m.Or(a.And(c).And(e))
	m = m.Or(a.And(d).And(e))
	m = m.Or(b.And(c).And(d))
	m = m.Or(b.And(c).And(e))
	m = m.Or(b.And(d).And(e))
	return m.Or(c.And(d).And(e))
}
