//go:build !goexperiment.simd || !go1.27 || !(amd64 || arm64 || wasm)

package core

import "encoding/binary"

func majority5Row(src, dst []byte, width int) {
	if width < 5 {
		return
	}
	center := 2
	// The five taps are five overlapping loads of the same row, so a word of
	// output carries no state into the next - which is the point, since the
	// running counter this replaces made every centre depend on the one before
	// it.
	for ; center+majorityWord <= width-2; center += majorityWord {
		a := binary.LittleEndian.Uint64(src[center-2:])
		b := binary.LittleEndian.Uint64(src[center-1:])
		c := binary.LittleEndian.Uint64(src[center:])
		d := binary.LittleEndian.Uint64(src[center+1:])
		e := binary.LittleEndian.Uint64(src[center+2:])
		binary.LittleEndian.PutUint64(dst[center:], majority5Mask(a, b, c, d, e))
	}
	for ; center < width-2; center++ {
		count := 0
		for k := -2; k <= 2; k++ {
			if src[center+k] != 0 {
				count++
			}
		}
		if count > 2 {
			dst[center] = 255
		} else {
			dst[center] = 0
		}
	}
}
