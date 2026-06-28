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

// finderPassStats records the per-pass finder-detection counters that the
// jabdiag-tagged diagnostic reads off the detector. They are observation only
// and never influence detection.
type finderPassStats struct {
	rawHits        int    // n-1-1-1-m run-length hits (horizontal + conditional vertical scan)
	crossSurvivors [4]int // candidates passing crossCheckPattern, by finder type
	preprune       [4]int // selectBestPatterns group sizes before the 0.5*maxFound prune
	selected       [4]int // foundCount of the selected pattern per type after the prune (0 = absent)
	missing        int    // types absent after selection
	status         int    // findPrimarySymbol status for the pass
	interpolated   bool   // whether the single-missing-finder estimate fired
}

// detectorStats aggregates finder-detection instrumentation across the up-to-two
// binarization passes locateFinders runs.
type detectorStats struct {
	passes []finderPassStats // one entry per findPrimarySymbol pass
	rgbAve [3]float32        // retry thresholds from getAveragePixelValue, between passes
}

// primaryDetector orchestrates primary-symbol finder detection over the three
// binarized channels. Its findPrimarySymbol/selectBestPatterns/scanPatternVertical
// methods populate stats, the single source of truth for the diagnostic. The ch
// field is a by-value [3]*bitmap: the retry's re-binarization (locateFinders) is
// scoped to this detector and never leaks into secondary decoding.
type primaryDetector struct {
	bm    *bitmap
	ch    [3]*bitmap
	mode  int
	fps   []finderPattern
	stats detectorStats
}

// pass returns the current (last-appended) finder pass's stats.
func (d *primaryDetector) pass() *finderPassStats {
	return &d.stats.passes[len(d.stats.passes)-1]
}

// detectPrimary locates the primary symbol's finder patterns, rectifies and
// samples the symbol, and decodes it, falling back to alignment-pattern
// resampling if the finder-pattern sample fails (detectMaster in detector.c).
func detectPrimary(bm *bitmap, ch [3]*bitmap, symbol *decodedSymbol) bool {
	d := &primaryDetector{bm: bm, ch: ch, mode: intensiveDetect}
	if !d.locateFinders() {
		return false
	}
	fps := d.fps

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
	apMatrix := sampleSymbolByAlignmentPattern(bm, d.ch, symbol, fps)
	if apMatrix == nil {
		return false
	}
	return decodePrimary(apMatrix, symbol) == jabSuccess
}

// locateFinders runs the finder search, falling back to a finder-seeded second
// binarization pass on failure (the retry orchestration of detectMaster in
// detector.c). The retry re-binarizes d.ch in place; because the channel array
// is held by value, that swap is scoped to this detector and does not propagate
// to secondary detection.
func (d *primaryDetector) locateFinders() bool {
	status := d.findPrimarySymbol()
	if status == fatalError {
		return false
	}
	if status == jabSuccess {
		return true
	}

	// Retry 1: re-binarize using adaptive thresholds from around the found patterns.
	rgbAve := getAveragePixelValue(d.bm, d.fps)
	d.stats.rgbAve = rgbAve
	ch2 := binarizerRGB(d.bm, rgbAve[:])
	d.ch[0], d.ch[1], d.ch[2] = ch2[0], ch2[1], ch2[2]
	if d.findPrimarySymbol() == jabSuccess {
		return true
	}

	// Retry 2 (descreen ladder): screen captures inject the display's subpixel/diode
	// lattice and moiré, which can leave the raw and avg-RGB passes without enough
	// surviving finders. Low-pass the source before binarizing, walking scales since
	// the module size is unknown. bm is left untouched so colour sampling still reads
	// the original pixels; the d.ch swap stays primary-scoped.
	for _, r := range descreenRadii {
		chN := binarizerRGB(descreen(d.bm, r), nil)
		d.ch[0], d.ch[1], d.ch[2] = chN[0], chN[1], chN[2]
		if d.findPrimarySymbol() == jabSuccess {
			return true
		}
	}
	return false
}
