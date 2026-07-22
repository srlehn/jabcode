package core

import (
	"math/rand/v2"
	"testing"
)

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

func TestRGBBlockStatsCropsAndStrides(t *testing.T) {
	rng := rand.New(rand.NewPCG(0x726762, 0x73746174))
	for _, channels := range []int{3, 4} {
		for _, width := range []int{1, 2, 5, 7, 16} {
			height := 6
			pix := make([]byte, width*height*channels)
			for i := range pix {
				pix[i] = byte(rng.Uint32N(256))
			}
			for _, region := range []struct{ sx, ex, sy, ey int }{
				{0, width, 0, height},
				{0, min(3, width), 1, 5},
				{max(0, width-3), width, 0, 2},
			} {
				lo, hi, sum, n := RGBBlockStats(pix, width, channels,
					region.sx, region.ex, region.sy, region.ey)
				wantLo := [3]int{255, 255, 255}
				var wantHi [3]int
				var wantSum [3]float64
				wantN := 0
				for y := region.sy; y < region.ey; y++ {
					for x := region.sx; x < region.ex; x++ {
						o := (y*width + x) * channels
						for c := range 3 {
							v := int(pix[o+c])
							wantLo[c] = min(wantLo[c], v)
							wantHi[c] = max(wantHi[c], v)
							wantSum[c] += float64(v)
						}
						wantN++
					}
				}
				if lo != wantLo || hi != wantHi || sum != wantSum || n != wantN {
					t.Fatalf("channels=%d width=%d region=%+v: got (%v, %v, %v, %d), want (%v, %v, %v, %d)",
						channels, width, region, lo, hi, sum, n, wantLo, wantHi, wantSum, wantN)
				}
			}
		}
	}
}
