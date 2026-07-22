//go:build goexperiment.simd && (amd64 || (go1.27 && (arm64 || wasm)))

package core

import "testing"

var benchmarkAvgVarResult float64

func BenchmarkAvgVarSIMD(b *testing.B) {
	rgb := []byte{17, 91, 233}
	b.ResetTimer()
	for range b.N {
		_, benchmarkAvgVarResult = avgVarSIMD(rgb)
	}
}

func BenchmarkAvgVarScalarReference(b *testing.B) {
	rgb := []byte{17, 91, 233}
	b.ResetTimer()
	for range b.N {
		avg := float64(int(rgb[0])+int(rgb[1])+int(rgb[2])) / 3
		variance := 0.0
		for _, value := range rgb {
			delta := float64(value) - avg
			variance += delta * delta
		}
		benchmarkAvgVarResult = variance / 3
	}
}
