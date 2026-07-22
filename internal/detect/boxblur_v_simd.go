//go:build goexperiment.simd && go1.27 && (amd64 || arm64 || wasm)

package detect

import (
	"simd"

	"github.com/srlehn/jabcode/internal/core"
)

func boxBlurV(src, dst []float64, w, h, radius int) {
	lanes := simd.VectorBitSize() / 64
	if lanes < 1 {
		boxBlurVScalar(src, dst, w, h, radius)
		return
	}
	win := float64(2*radius + 1)
	core.ParallelChunks(w, lanes, func(xlo, xhi int) {
		values := make([]float64, lanes)
		x := xlo
		for ; x+lanes <= xhi; x += lanes {
			sum := simd.BroadcastFloat64s(0)
			for k := -radius; k <= radius; k++ {
				sum = sum.Add(simd.LoadFloat64s(src[min(max(k, 0), h-1)*w+x:]))
			}
			sum.Store(values[:lanes])
			for lane := range lanes {
				dst[x+lane] = values[lane] / win
			}
			for y := 1; y < h; y++ {
				sum = sum.Add(simd.LoadFloat64s(src[min(max(y+radius, 0), h-1)*w+x:]))
				sum = sum.Sub(simd.LoadFloat64s(src[min(max(y-1-radius, 0), h-1)*w+x:]))
				sum.Store(values[:lanes])
				for lane := range lanes {
					dst[y*w+x+lane] = values[lane] / win
				}
			}
		}
		for ; x < xhi; x++ {
			var sum float64
			for k := -radius; k <= radius; k++ {
				sum += src[min(max(k, 0), h-1)*w+x]
			}
			dst[x] = sum / win
			for y := 1; y < h; y++ {
				sum += src[min(max(y+radius, 0), h-1)*w+x]
				sum -= src[min(max(y-1-radius, 0), h-1)*w+x]
				dst[y*w+x] = sum / win
			}
		}
	})
}
