package core

// MinMax orders a pixel's three channels, returning the values and their
// original channel indices.
func MinMax(rgb []byte) (min, mid, max byte, iMin, iMid, iMax int) {
	// Ports getMinMax in binarizer.c.
	iMin, iMid, iMax = 0, 1, 2
	if rgb[iMin] > rgb[iMax] {
		iMin, iMax = iMax, iMin
	}
	if rgb[iMin] > rgb[iMid] {
		iMin, iMid = iMid, iMin
	}
	if rgb[iMid] > rgb[iMax] {
		iMid, iMax = iMax, iMid
	}
	return rgb[iMin], rgb[iMid], rgb[iMax], iMin, iMid, iMax
}

// BoolColor maps a binary channel test to the 0/255 color value used for type
// classification.
func BoolColor(b bool) int {
	if b {
		return 255
	}
	return 0
}
