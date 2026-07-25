//go:build !goexperiment.simd || (!go1.27 && !amd64) || (go1.27 && !amd64 && !arm64 && !wasm)

package core

import "encoding/binary"

// Majority5Columns writes the interior pixels of a vertical five-pixel
// majority pass. Edge pixels remain untouched.
func Majority5Columns(src, dst []byte, width, height int) {
	const radius = 2
	if width < 2*radius+1 || height < 2*radius+1 {
		return
	}
	ParallelRows(height-2*radius, func(lo, hi int) {
		for i := lo + radius; i < hi+radius; i++ {
			// The five taps of a column are the same offset in five
			// consecutive rows, so a whole word of columns votes at once and
			// the per-column running sum this replaces - a read-modify-write
			// per output pixel - is not needed at all.
			r0 := (i - 2) * width
			r1 := r0 + width
			r2 := r1 + width
			r3 := r2 + width
			r4 := r3 + width
			out := r2
			j := radius
			for ; j+majorityWord <= width-radius; j += majorityWord {
				a := binary.LittleEndian.Uint64(src[r0+j:])
				b := binary.LittleEndian.Uint64(src[r1+j:])
				c := binary.LittleEndian.Uint64(src[r2+j:])
				d := binary.LittleEndian.Uint64(src[r3+j:])
				e := binary.LittleEndian.Uint64(src[r4+j:])
				binary.LittleEndian.PutUint64(dst[out+j:], majority5Mask(a, b, c, d, e))
			}
			for ; j < width-radius; j++ {
				count := boolByte(src[r0+j] != 0) + boolByte(src[r1+j] != 0) +
					boolByte(src[r2+j] != 0) + boolByte(src[r3+j] != 0) +
					boolByte(src[r4+j] != 0)
				dst[out+j] = byte(255 * boolByte(count > radius))
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
