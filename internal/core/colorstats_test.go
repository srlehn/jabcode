package core

import "testing"

func TestAvgVarExact(t *testing.T) {
	for _, rgb := range [][]byte{{0, 0, 0}, {255, 128, 1}, {17, 91, 233}, {255, 255, 255}} {
		avg := float64(int(rgb[0])+int(rgb[1])+int(rgb[2])) / 3
		variance := 0.0
		for _, value := range rgb {
			delta := float64(value) - avg
			variance += delta * delta
		}
		gotAvg, gotVariance := AvgVar(rgb)
		if gotAvg != avg || gotVariance != variance/3 {
			t.Fatalf("rgb=%v got=(%v,%v) want=(%v,%v)", rgb, gotAvg, gotVariance, avg, variance/3)
		}
	}
}
