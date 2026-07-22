//go:build !goexperiment.simd || (!go1.27 && !amd64) || (go1.27 && !amd64 && !arm64 && !wasm)

package detect

func boxBlurV(src, dst []float64, w, h, radius int) {
	boxBlurVScalar(src, dst, w, h, radius)
}
