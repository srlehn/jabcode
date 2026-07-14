//go:build jabcode_bsi || jabcode_non_iso_encode

package tables

import "image"

// BSIPrimaryPalettePositions gives the eight fixed primary-palette module
// positions shared by the BSI encoder and decoder.
var BSIPrimaryPalettePositions = [8]image.Point{
	image.Pt(4, 1), image.Pt(4, 2), image.Pt(5, 1), image.Pt(5, 2),
	image.Pt(2, 4), image.Pt(2, 5), image.Pt(1, 4), image.Pt(1, 5),
}

// BSIPalettePlacementIndex returns the physical order of the palette colors
// carried by a BSI symbol. The returned slice is independently mutable.
func BSIPalettePlacementIndex(size, colorNumber int) []int {
	index := make([]int, size)
	for i := range index {
		index[i] = i
	}
	switch colorNumber {
	case 16:
		for i := range 4 {
			index[4+i] = 12 + i
		}
		for i := range 8 {
			index[8+i] = 4 + i
		}
	case 32:
		copy(index[2:8], []int{6, 7, 24, 25, 30, 31})
		for i := range 4 {
			index[8+i] = 2 + i
		}
		for i := range 16 {
			index[12+i] = 8 + i
		}
		for i := range 4 {
			index[28+i] = 26 + i
		}
	case 64:
		copy(index[1:8], []int{3, 12, 15, 48, 51, 60, 63})
		for i := range 2 {
			index[8+i] = 1 + i
		}
		for i := range 8 {
			index[10+i] = 4 + i
		}
		for i := range 2 {
			index[18+i] = 13 + i
		}
		for i := range 32 {
			index[20+i] = 16 + i
		}
		for i := range 2 {
			index[52+i] = 49 + i
		}
		for i := range 8 {
			index[54+i] = 52 + i
		}
		for i := range 2 {
			index[62+i] = 61 + i
		}
	case 128:
		copy(index[1:8], []int{3, 12, 15, 112, 115, 124, 127})
		for i := range 2 {
			index[8+i] = 1 + i
		}
		for i := range 8 {
			index[10+i] = 4 + i
		}
		for i := range 2 {
			index[18+i] = 13 + i
		}
		for i := range 16 {
			index[20+i] = 32 + i
			index[36+i] = 80 + i
		}
		for i := range 2 {
			index[52+i] = 113 + i
		}
		for i := range 8 {
			index[54+i] = 116 + i
		}
		for i := range 2 {
			index[62+i] = 125 + i
		}
	case 256:
		copy(index[1:8], []int{3, 28, 31, 224, 227, 252, 255})
		for i := range 2 {
			index[8+i] = 1 + i
			index[18+i] = 29 + i
			index[52+i] = 225 + i
			index[62+i] = 253 + i
		}
		for i := range 4 {
			index[10+i] = 8 + i
			index[14+i] = 20 + i
			index[20+i] = 64 + i
			index[24+i] = 72 + i
			index[28+i] = 84 + i
			index[32+i] = 92 + i
			index[36+i] = 160 + i
			index[40+i] = 168 + i
			index[44+i] = 180 + i
			index[48+i] = 188 + i
			index[54+i] = 232 + i
			index[58+i] = 244 + i
		}
	}
	return index
}
