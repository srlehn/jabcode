//go:build goexperiment.simd && amd64 && !go1.27

package core

import "simd/archsimd"

// AvgVar uses the Go 1.26 archsimd load/store API while retaining the same
// three-channel contract as the scalar and Go 1.27 implementations.
func AvgVar(rgb []byte) (avg, variance float64) {
	avg = float64(int(rgb[0])+int(rgb[1])+int(rgb[2])) / 3
	delta := [4]float64{
		float64(rgb[0]) - avg,
		float64(rgb[1]) - avg,
		float64(rgb[2]) - avg,
		0,
	}
	loInput := [2]float64{delta[0], delta[1]}
	hiInput := [2]float64{delta[2], delta[3]}
	lo := archsimd.LoadFloat64x2(&loInput)
	hi := archsimd.LoadFloat64x2(&hiInput)
	squared := lo.Mul(lo).Add(hi.Mul(hi))
	var sum [2]float64
	squared.Store(&sum)
	return avg, (sum[0] + sum[1]) / 3
}
