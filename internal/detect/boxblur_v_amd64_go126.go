//go:build goexperiment.simd && amd64 && !go1.27

package detect

import (
	"simd/archsimd"

	"golang.org/x/sys/cpu"

	"github.com/srlehn/jabcode/internal/core"
)

// boxBlurV vectorizes boxBlurVScalar's row-major sweep. The running sums live
// in a per-chunk column buffer, so lanes cover neighbouring columns and every
// load, store and divide streams contiguously.
//
// This build picks its width at compile time: archsimd exposes fixed-width
// types with no runtime query, so the widest safely assumable vector is the
// 256-bit one behind an AVX2 guard, and machines without it take the scalar
// sweep rather than faulting on an unsupported encoding.
func boxBlurV(src, dst []float64, w, h, radius int) {
	if !cpu.X86.HasAVX2 {
		boxBlurVScalar(src, dst, w, h, radius)
		return
	}
	const lanes = 4
	window := float64(2*radius + 1)
	windowLanes := [lanes]float64{window, window, window, window}
	win := archsimd.LoadFloat64x4(&windowLanes)
	core.ParallelChunks(w, 64, func(xlo, xhi int) {
		n := xhi - xlo
		sums := make([]float64, n)
		seed := func(row int) {
			i := 0
			for ; i+lanes <= n; i += lanes {
				archsimd.LoadFloat64x4((*[lanes]float64)(sums[i:])).
					Add(archsimd.LoadFloat64x4((*[lanes]float64)(src[row+i:]))).
					Store((*[lanes]float64)(sums[i:]))
			}
			for ; i < n; i++ {
				sums[i] += src[row+i]
			}
		}
		// The window slides by one combined delta rather than an add and a
		// separate subtract, so each column sees the same rounding the scalar
		// sweep applies and the two backends cannot drift apart.
		slide := func(addRow, subRow int) {
			i := 0
			for ; i+lanes <= n; i += lanes {
				delta := archsimd.LoadFloat64x4((*[lanes]float64)(src[addRow+i:])).
					Sub(archsimd.LoadFloat64x4((*[lanes]float64)(src[subRow+i:])))
				archsimd.LoadFloat64x4((*[lanes]float64)(sums[i:])).
					Add(delta).Store((*[lanes]float64)(sums[i:]))
			}
			for ; i < n; i++ {
				sums[i] += src[addRow+i] - src[subRow+i]
			}
		}
		for k := -radius; k <= radius; k++ {
			seed(min(max(k, 0), h-1)*w + xlo)
		}
		for y := range h {
			out := y*w + xlo
			i := 0
			for ; i+lanes <= n; i += lanes {
				archsimd.LoadFloat64x4((*[lanes]float64)(sums[i:])).
					Div(win).Store((*[lanes]float64)(dst[out+i:]))
			}
			for ; i < n; i++ {
				dst[out+i] = sums[i] / window
			}
			if y+1 == h {
				break
			}
			slide(min(max(y+1+radius, 0), h-1)*w+xlo, min(max(y-radius, 0), h-1)*w+xlo)
		}
	})
}
