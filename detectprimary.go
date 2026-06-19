package jabcode

import (
	"errors"
	"image"
)

// maxSymbolNumber is the maximum number of symbols in a JAB Code (MAX_SYMBOL_NUMBER).
const maxSymbolNumber = 61

// Decode decodes the data of a JAB Code from img: the primary symbol and any
// docked secondary symbols (decodeJABCode/decodeJABCodeEx in detector.c, in
// NORMAL_DECODE mode). Reading a JAB Code from a file is stdlib decoding
// (e.g. png.Decode) followed by Decode.
func Decode(img image.Image) ([]byte, error) {
	bm := bitmapFromImage(img)
	balanceRGB(bm)
	ch := binarizerRGB(bm, nil)

	symbols := make([]decodedSymbol, maxSymbolNumber)
	total := 0
	if detectPrimary(bm, ch, &symbols[0]) {
		total++
	}

	// Detect and decode docked secondary symbols recursively.
	res := true
	for i := 0; i < total && total < maxSymbolNumber; i++ {
		if !decodeDockedSecondaries(bm, ch, symbols, i, &total) {
			res = false
			break
		}
	}
	if total == 0 || !res {
		return nil, errors.New("jabcode: detecting or decoding the JAB Code failed")
	}

	// Concatenate the decoded bits of all symbols, then interpret them.
	var bits []byte
	for i := 0; i < total; i++ {
		bits = append(bits, symbols[i].data...)
	}
	return decodeData(bits), nil
}

// detectPrimary locates the primary symbol's finder patterns, rectifies and
// samples the symbol, and decodes it, falling back to alignment-pattern
// resampling if the finder-pattern sample fails (detectMaster in detector.c).
func detectPrimary(bm *bitmap, ch [3]*bitmap, symbol *decodedSymbol) bool {
	fps, status := findPrimarySymbol(bm, ch, intensiveDetect)
	if status == fatalError {
		return false
	}
	if status == jabFailure {
		// Re-binarize using adaptive thresholds from around the found patterns.
		rgbAve := getAveragePixelValue(bm, fps)
		ch2 := binarizerRGB(bm, rgbAve[:])
		ch[0], ch[1], ch[2] = ch2[0], ch2[1], ch2[2]
		fps, status = findPrimarySymbol(bm, ch, intensiveDetect)
		if status != jabSuccess {
			return false
		}
	}

	sideSize := calculateSideSize(fps)
	if sideSize.X == -1 || sideSize.Y == -1 {
		return false
	}

	pt := getPerspectiveTransform(fps[0].center, fps[1].center, fps[2].center, fps[3].center, sideSize)
	matrix := sampleSymbol(bm, pt, sideSize)
	if matrix == nil {
		return false
	}

	symbol.index = 0
	symbol.hostIndex = 0
	symbol.sideSize = sideSize
	symbol.moduleSize = (fps[0].moduleSize + fps[1].moduleSize + fps[2].moduleSize + fps[3].moduleSize) / 4.0
	for i := range 4 {
		symbol.patternPositions[i] = fps[i].center
	}

	switch res := decodePrimary(matrix, symbol); {
	case res == jabSuccess:
		return true
	case res < 0: // fatal error occurred
		return false
	}

	// if decoding using only finder patterns failed, try decoding using alignment patterns
	symbol.sideSize = image.Pt(version2size(symbol.meta.sideVersion.X), version2size(symbol.meta.sideVersion.Y))
	apMatrix := sampleSymbolByAlignmentPattern(bm, ch, symbol, fps)
	if apMatrix == nil {
		return false
	}
	return decodePrimary(apMatrix, symbol) == jabSuccess
}
