//go:build goexperiment.simd && amd64 && !go1.27

package detect

import (
	"simd/archsimd"

	"github.com/srlehn/jabcode/internal/core"
)

func boxBlurV(src, dst []float64, w, h, radius int) {
	const lanes = 2
	win := float64(2*radius + 1)
	core.ParallelChunks(w, lanes, func(xlo, xhi int) {
		x := xlo
		for ; x+lanes <= xhi; x += lanes {
			sum := archsimd.Float64x2{}
			for k := -radius; k <= radius; k++ {
				sum = sum.Add(archsimd.LoadFloat64x2((*[2]float64)(src[min(max(k, 0), h-1)*w+x:])))
			}
			var values [2]float64
			sum.Store(&values)
			for lane := range lanes {
				dst[x+lane] = values[lane] / win
			}
			for y := 1; y < h; y++ {
				sum = sum.Add(archsimd.LoadFloat64x2((*[2]float64)(src[min(max(y+radius, 0), h-1)*w+x:])))
				sum = sum.Sub(archsimd.LoadFloat64x2((*[2]float64)(src[min(max(y-1-radius, 0), h-1)*w+x:])))
				sum.Store(&values)
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
