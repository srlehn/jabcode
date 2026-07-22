//go:build !goexperiment.simd || (!go1.27 && !amd64) || (go1.27 && !amd64 && !arm64 && !wasm)

package core

// Majority5Columns writes the interior pixels of a vertical five-pixel
// majority pass. Edge pixels remain untouched.
func Majority5Columns(src, dst []byte, width, height int) {
	const radius = 2
	if width < 2*radius+1 || height < 2*radius+1 {
		return
	}
	ParallelRows(height-2*radius, func(lo, hi int) {
		first := lo + radius
		colSum := make([]uint8, width)
		for j := radius; j < width-radius; j++ {
			sum := 0
			for r := first - radius; r <= first+radius; r++ {
				sum += boolByte(src[r*width+j] != 0)
			}
			colSum[j] = uint8(sum)
			dst[first*width+j] = byte(255 * boolByte(sum > radius))
		}
		for i := first + 1; i < hi+radius; i++ {
			add, sub := (i+radius)*width, (i-radius-1)*width
			for j := radius; j < width-radius; j++ {
				sum := int(colSum[j]) + boolByte(src[add+j] != 0) - boolByte(src[sub+j] != 0)
				colSum[j] = uint8(sum)
				dst[i*width+j] = byte(255 * boolByte(sum > radius))
			}
		}
	})
}

func boolByte(value bool) int {
	if value {
		return 1
	}
	return 0
}
