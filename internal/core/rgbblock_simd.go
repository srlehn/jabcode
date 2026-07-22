//go:build goexperiment.simd && go1.27 && (amd64 || arm64 || wasm)

package core

import "simd/archsimd"

func rgbBlockSums(pix []byte, width, channels, sx, ex, sy, ey int) (sum [3]float64) {
	var r, g, b [2]float64
	var rSum, gSum, bSum archsimd.Float64x2
	for y := sy; y < ey; y++ {
		row := y * width * channels
		for x := sx; x < ex; x += 2 {
			o := row + x*channels
			r[0] = float64(pix[o])
			g[0] = float64(pix[o+1])
			b[0] = float64(pix[o+2])
			if x+1 < ex {
				n := o + channels
				r[1] = float64(pix[n])
				g[1] = float64(pix[n+1])
				b[1] = float64(pix[n+2])
			} else {
				r[1], g[1], b[1] = 0, 0, 0
			}
			rSum = rSum.Add(archsimd.LoadFloat64x2(r[:]))
			gSum = gSum.Add(archsimd.LoadFloat64x2(g[:]))
			bSum = bSum.Add(archsimd.LoadFloat64x2(b[:]))
		}
	}
	var rOut, gOut, bOut [2]float64
	rSum.Store(rOut[:])
	gSum.Store(gOut[:])
	bSum.Store(bOut[:])
	return [3]float64{rOut[0] + rOut[1], gOut[0] + gOut[1], bOut[0] + bOut[1]}
}
