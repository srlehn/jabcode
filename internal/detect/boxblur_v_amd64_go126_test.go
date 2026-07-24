//go:build goexperiment.simd && amd64 && !go1.27

package detect

import (
	"math/rand"
	"testing"

	"golang.org/x/sys/cpu"
)

// TestBoxBlurVWidthsMatchColumnwise checks each vector width against the
// columnwise oracle directly, rather than only whichever width the dispatch
// happens to select on this machine. A width whose feature bit is clear is
// skipped: its instructions would fault here, so it can only be covered on a
// CPU that supports it.
func TestBoxBlurVWidthsMatchColumnwise(t *testing.T) {
	widths := []struct {
		name      string
		supported bool
		fn        func(src, dst []float64, w, h, radius int)
	}{
		{"avx2", cpu.X86.HasAVX2, boxBlurVAVX2},
		{"avx512", cpu.X86.HasAVX512F, boxBlurVAVX512},
	}
	rng := rand.New(rand.NewSource(1))
	// Sizes straddle the 64-column chunking and every lane remainder, so the
	// scalar tail past the last whole vector is exercised at each width.
	dims := []struct{ w, h, radius int }{
		{1, 1, 1}, {3, 5, 2}, {7, 4, 1}, {64, 8, 1}, {65, 9, 3}, {130, 17, 4}, {200, 6, 9},
	}
	for _, width := range widths {
		t.Run(width.name, func(t *testing.T) {
			if !width.supported {
				t.Skipf("%s is not supported by this CPU", width.name)
			}
			for _, dim := range dims {
				src := make([]float64, dim.w*dim.h)
				for i := range src {
					src[i] = rng.Float64()
				}
				got := make([]float64, len(src))
				want := make([]float64, len(src))
				width.fn(src, got, dim.w, dim.h, dim.radius)
				boxBlurVColumnwise(src, want, dim.w, dim.h, dim.radius)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("%dx%d radius %d: pixel %d = %v, columnwise gives %v",
							dim.w, dim.h, dim.radius, i, got[i], want[i])
					}
				}
			}
		})
	}
}
