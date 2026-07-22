package core

// Majority5Row writes the interior pixels of a horizontal five-pixel
// majority pass. Edge pixels are deliberately untouched because the filter's
// full-kernel contract leaves them unchanged.
func Majority5Row(src, dst []byte, width int) {
	majority5Row(src, dst, width)
}

// Majority5VerticalRow writes one output row from five neighboring rows.
func Majority5VerticalRow(rows [5][]byte, dst []byte, width int) {
	majority5VerticalRow(rows, dst, width)
}
