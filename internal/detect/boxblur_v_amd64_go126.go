//go:build goexperiment.simd && amd64 && !go1.27

package detect

import (
	"simd/archsimd"

	"golang.org/x/sys/cpu"

	"github.com/srlehn/jabcode/internal/core"
)

// boxBlurV picks the widest vector the running CPU supports. archsimd exposes
// fixed-width types with no runtime width query, so each width is its own
// function selected behind its own feature bit, and a machine with neither
// extension takes the scalar sweep rather than faulting on an unsupported
// encoding.
//
// The per-width bodies are deliberately line-for-line parallel. They cannot be
// one generic function: instantiating a type parameter with an archsimd vector
// makes the Go 1.26 compiler fail with "cannot represent parameters of type
// [N]float64 in registers", at every width, so the duplication is forced by
// the toolchain rather than chosen.
func boxBlurV(src, dst []float64, w, h, radius int) {
	switch {
	case cpu.X86.HasAVX512F:
		boxBlurVAVX512(src, dst, w, h, radius)
	case cpu.X86.HasAVX2:
		boxBlurVAVX2(src, dst, w, h, radius)
	default:
		boxBlurVScalar(src, dst, w, h, radius)
	}
}

// boxBlurVAVX2 vectorizes boxBlurVScalar's row-major sweep. The running sums
// live in a per-chunk column buffer, so lanes cover neighbouring columns and
// every load, store and divide streams contiguously. Columns past the last
// whole vector finish on the scalar tail.
//
// The window slides by one combined delta rather than an add and a separate
// subtract, so each column sees the same rounding the scalar sweep applies and
// the two cannot drift apart.
func boxBlurVAVX2(src, dst []float64, w, h, radius int) {
	const lanes = 4
	window := float64(2*radius + 1)
	win := archsimd.LoadFloat64x4Slice([]float64{window, window, window, window})
	core.ParallelChunks(w, 64, func(xlo, xhi int) {
		n := xhi - xlo
		sums := make([]float64, n)
		seed := func(row int) {
			i := 0
			for ; i+lanes <= n; i += lanes {
				archsimd.LoadFloat64x4Slice(sums[i:]).
					Add(archsimd.LoadFloat64x4Slice(src[row+i:])).StoreSlice(sums[i:])
			}
			for ; i < n; i++ {
				sums[i] += src[row+i]
			}
		}
		slide := func(addRow, subRow int) {
			i := 0
			for ; i+lanes <= n; i += lanes {
				delta := archsimd.LoadFloat64x4Slice(src[addRow+i:]).
					Sub(archsimd.LoadFloat64x4Slice(src[subRow+i:]))
				archsimd.LoadFloat64x4Slice(sums[i:]).Add(delta).StoreSlice(sums[i:])
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
				archsimd.LoadFloat64x4Slice(sums[i:]).Div(win).StoreSlice(dst[out+i:])
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

// boxBlurVAVX512 is boxBlurVAVX2 at eight lanes. Keep the two bodies in step.
func boxBlurVAVX512(src, dst []float64, w, h, radius int) {
	const lanes = 8
	window := float64(2*radius + 1)
	win := archsimd.LoadFloat64x8Slice([]float64{
		window, window, window, window, window, window, window, window,
	})
	core.ParallelChunks(w, 64, func(xlo, xhi int) {
		n := xhi - xlo
		sums := make([]float64, n)
		seed := func(row int) {
			i := 0
			for ; i+lanes <= n; i += lanes {
				archsimd.LoadFloat64x8Slice(sums[i:]).
					Add(archsimd.LoadFloat64x8Slice(src[row+i:])).StoreSlice(sums[i:])
			}
			for ; i < n; i++ {
				sums[i] += src[row+i]
			}
		}
		slide := func(addRow, subRow int) {
			i := 0
			for ; i+lanes <= n; i += lanes {
				delta := archsimd.LoadFloat64x8Slice(src[addRow+i:]).
					Sub(archsimd.LoadFloat64x8Slice(src[subRow+i:]))
				archsimd.LoadFloat64x8Slice(sums[i:]).Add(delta).StoreSlice(sums[i:])
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
				archsimd.LoadFloat64x8Slice(sums[i:]).Div(win).StoreSlice(dst[out+i:])
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
