//go:build goexperiment.simd && go1.27 && (amd64 || arm64 || wasm)

package core

import "simd"

// Majority5Columns writes the interior pixels of a vertical five-pixel
// majority pass. Edge pixels remain untouched.
func Majority5Columns(src, dst []byte, width, height int) {
	const radius = 2
	if width < 2*radius+1 || height < 2*radius+1 {
		return
	}
	lanes := simd.VectorBitSize() / 8
	limit := width - radius
	ParallelRows(height-2*radius, func(lo, hi int) {
		for i := lo + radius; i < hi+radius; i++ {
			j := radius
			for ; j+lanes <= limit; j += lanes {
				a := simd.LoadUint8s(src[(i-radius)*width+j:])
				b := simd.LoadUint8s(src[(i-1)*width+j:])
				c := simd.LoadUint8s(src[i*width+j:])
				d := simd.LoadUint8s(src[(i+1)*width+j:])
				e := simd.LoadUint8s(src[(i+radius)*width+j:])
				majority := a.And(b).And(c).
					Or(a.And(b).And(d)).
					Or(a.And(b).And(e)).
					Or(a.And(c).And(d)).
					Or(a.And(c).And(e)).
					Or(a.And(d).And(e)).
					Or(b.And(c).And(d)).
					Or(b.And(c).And(e)).
					Or(b.And(d).And(e)).
					Or(c.And(d).And(e))
				majority.Store(dst[i*width+j:])
			}
			for ; j < limit; j++ {
				sum := 0
				for r := i - radius; r <= i+radius; r++ {
					if src[r*width+j] != 0 {
						sum++
					}
				}
				dst[i*width+j] = byte(255 * boolByte(sum > radius))
			}
		}
	})
}

func boolByte(value bool) int {
	if value {
		return 1
	}
	return 0
}
