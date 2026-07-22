package core

// Majority5Row writes the interior pixels of a horizontal five-pixel
// majority pass. src must contain only 0 or 255 bytes: the SIMD
// implementation relies on those values being boolean masks. Edge pixels are
// deliberately untouched because the filter's full-kernel contract leaves
// them unchanged.
func Majority5Row(src, dst []byte, width int) {
	majority5Row(src, dst, width)
}
