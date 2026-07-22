//go:build goexperiment.simd && (amd64 || (go1.27 && (arm64 || wasm)))

package core

import (
	"math"
	"testing"
)

var benchmarkAvgVarResult float64

func TestAvgVarSIMDParity(t *testing.T) {
	for _, rgb := range [][]byte{{0, 0, 0}, {255, 128, 1}, {17, 91, 233}, {255, 255, 255}} {
		avg := float64(int(rgb[0])+int(rgb[1])+int(rgb[2])) / 3
		var sum float64
		for _, value := range rgb {
			delta := float64(value) - avg
			sum += delta * delta
		}
		gotAvg, gotVariance := AvgVar(rgb)
		if gotAvg != avg || math.Abs(gotVariance-sum/3) > 1e-11 {
			t.Fatalf("rgb=%v got=(%v,%v) want=(%v,%v)", rgb, gotAvg, gotVariance, avg, sum/3)
		}
	}
}

func BenchmarkAvgVarSIMD(b *testing.B) {
	rgb := []byte{17, 91, 233}
	b.ResetTimer()
	for range b.N {
		_, benchmarkAvgVarResult = AvgVar(rgb)
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
