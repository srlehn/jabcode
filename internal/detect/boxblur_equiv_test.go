package detect

import (
	"math/rand"
	"testing"
)

// boxBlurVColumnwise is the column-at-a-time form boxBlurVScalar replaced. It
// stays here as the equivalence oracle: the swept form only reorders memory
// access, never the arithmetic, so the two must agree bit for bit.
func boxBlurVColumnwise(src, dst []float64, w, h, radius int) {
	win := float64(2*radius + 1)
	for x := range w {
		var sum float64
		for k := -radius; k <= radius; k++ {
			sum += src[min(max(k, 0), h-1)*w+x]
		}
		dst[x] = sum / win
		for y := 1; y < h; y++ {
			sum += src[min(max(y+radius, 0), h-1)*w+x] - src[min(max(y-1-radius, 0), h-1)*w+x]
			dst[y*w+x] = sum / win
		}
	}
}

func TestBoxBlurVMatchesColumnwise(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	// Sizes straddle the 64-column chunking and the radius-vs-height clamp, so
	// both the interior and the fully clamped short-image case are covered.
	for _, dim := range []struct{ w, h, radius int }{
		{1, 1, 1}, {3, 5, 2}, {64, 8, 1}, {65, 9, 3}, {130, 17, 4}, {200, 6, 9},
	} {
		src := make([]float64, dim.w*dim.h)
		for i := range src {
			src[i] = rng.Float64()
		}
		want := make([]float64, len(src))
		boxBlurVColumnwise(src, want, dim.w, dim.h, dim.radius)
		// boxBlurV is the build-selected entry point, so on a vector build this
		// covers the SIMD kernel rather than only its scalar twin.
		for _, impl := range []struct {
			name string
			fn   func(src, dst []float64, w, h, radius int)
		}{{"scalar", boxBlurVScalar}, {"selected", boxBlurV}} {
			got := make([]float64, len(src))
			impl.fn(src, got, dim.w, dim.h, dim.radius)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s %dx%d radius %d: pixel %d = %v, columnwise gives %v",
						impl.name, dim.w, dim.h, dim.radius, i, got[i], want[i])
				}
			}
		}
	}
}
