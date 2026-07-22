package core

import "testing"

func TestRGBBlockStats(t *testing.T) {
	pix := []byte{
		10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255,
		15, 25, 35, 255, 45, 55, 65, 255, 75, 85, 95, 255,
	}
	lo, hi, sum, n := RGBBlockStats(pix, 3, 4, 1, 3, 0, 2)
	if lo != [3]int{40, 50, 60} || hi != [3]int{75, 85, 95} || n != 4 {
		t.Fatalf("stats=(%v,%v,%d), want extrema=(%v,%v), n=4", lo, hi, n, [3]int{40, 50, 60}, [3]int{75, 85, 95})
	}
	if sum != [3]float64{230, 270, 310} {
		t.Fatalf("sum=%v, want [230 270 310]", sum)
	}
}
