//go:build !goexperiment.simd || (!go1.27 && !amd64) || (go1.27 && !(amd64 || arm64 || wasm))

package core

// AvgVar returns the mean and variance of a pixel's RGB values.
func AvgVar(rgb []byte) (avg, variance float64) {
	avg = float64(int(rgb[0])+int(rgb[1])+int(rgb[2])) / 3
	sum := 0.0
	for i := range 3 {
		d := float64(rgb[i]) - avg
		sum += d * d
	}
	return avg, sum / 3
}
