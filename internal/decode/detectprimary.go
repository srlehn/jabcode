package decode

import (
	"errors"
	"image"

	"github.com/srlehn/jabcode/internal/spec"
)

// maxSymbolNumber is the maximum number of symbols in a JAB Code.
const maxSymbolNumber = 61

// errDecodeFailed is returned when no orientation of img yields a readable symbol.
var errDecodeFailed = errors.New("jabcode: detecting or decoding the JAB Code failed")

// maxDecodeROIs bounds how many proposed regions the region-of-interest retry
// probes. The proposer ranks regions by score and a symbol's dense colourful
// texture dominates that ranking, so the true region is expected at the front;
// the cap keeps a failed read's cost bounded on cluttered images.
const maxDecodeROIs = 2

// Decode decodes the data of a JAB Code from img: the primary symbol and any docked
// secondary symbols. Reading a JAB Code from a file is stdlib decoding (e.g. png.Decode)
// followed by Decode.
//
// Finder detection collapses beyond ~20 degrees of rotation, so an upright read alone
// misses a rotated capture. Decode recovers it coarse-to-fine: try upright at full
// resolution first (clean captures resolve here and stay byte-identical), and on failure
// find the promising orientations on a downscaled copy before spending a full-resolution
// decode only on those few rungs. The decoded bytes are orientation-independent, so the
// first orientation that reads wins. The downscaled orientation search bounds the cost of
// a failed read by the probe resolution rather than the capture's megapixels - which also
// means a symbol small within a large frame can vanish in the probe downscale, so as the
// last resort the same orientation search runs per proposed region of interest, spending
// the bounded probe resolution on the region instead of the whole frame.
func Decode(img image.Image) ([]byte, error) {
	data, ok, evidence := decodeImage(img)
	if ok {
		return data, nil
	}
	// A blank or near-uniform image has no finder structure at any orientation, so skip
	// the rotation search entirely - the cheap uniform bailout.
	if !evidence {
		return nil, errDecodeFailed
	}
	// Spend a full-resolution decode only on the orientations the coarse search found
	// promising; counter-rotating a strongly-rotated code to near upright restores the
	// integer run-lengths its single-module finders need.
	for _, deg := range coarseOrientationRungs(img) {
		if data, ok, _ := decodeImage(rotateImage(img, deg)); ok {
			return data, nil
		}
	}
	// Region-of-interest retry: probe orientation per proposed region at the
	// region's own scale, restoring the module resolution a small symbol loses
	// in the whole-frame probe downscale. A region spanning the full frame
	// would repeat the search above at the same scale, so it is skipped.
	for _, roi := range proposeROIs(img, maxDecodeROIs) {
		if roi.bounds == img.Bounds() {
			continue
		}
		crop := cropImage(img, roi.bounds)
		for _, deg := range coarseOrientationRungs(crop) {
			if data, ok, _ := decodeImage(rotateImage(crop, deg)); ok {
				return data, nil
			}
		}
	}
	return nil, errDecodeFailed
}

// decodeImage attempts one full read of img as given: binarize, locate and decode
// the primary symbol, then its docked secondaries, then assemble the message. It
// runs the entire session on one image so the primary, the alignment-pattern
// fallback and the secondaries share a single coherent coordinate frame. evidence
// reports whether the finder search saw any finder structure at all, so Decode can
// skip the rotation search outright on blank or near-uniform input.
func decodeImage(img image.Image) (data []byte, ok, evidence bool) {
	// Ports decodeJABCode/decodeJABCodeEx (NORMAL_DECODE mode) in detector.c.
	bm := bitmapFromImage(img)
	balanceRGB(bm)
	ch := binarizerRGB(bm, nil)

	symbols := make([]decodedSymbol, maxSymbolNumber)
	d := &primaryDetector{bm: bm, ch: ch, mode: intensiveDetect}
	total := 0
	if detectPrimary(d, &symbols[0]) {
		total++
	}
	evidence = finderEvidence(d)

	// Detect and decode docked secondary symbols recursively.
	for i := 0; i < total && total < maxSymbolNumber; i++ {
		if !decodeDockedSecondaries(bm, ch, symbols, i, &total) {
			return nil, false, evidence
		}
	}
	if total == 0 {
		return nil, false, evidence
	}

	// Concatenate the decoded bits of all symbols, then interpret them.
	var bits []byte
	for i := 0; i < total; i++ {
		bits = append(bits, symbols[i].data...)
	}
	return decodeData(bits), true, evidence
}

// finderEvidence reports whether the upright finder search saw any finder structure at
// all - the cheap uniform bailout that lets Decode skip the rotation search on blank or
// near-uniform input. It gates on raw run-length hits (the n-1-1-1-m seed scan), which
// are rotation-robust: a code produces hundreds at every angle (the rotation gating
// measurement) even when the cross-check survivors collapse, whereas a blank image
// produces almost none. It deliberately does not try to judge orientation - that is the
// coarse search's job; a structured non-code image clears this gate and is then rejected
// by the coarse search finding no orientation with aligned finders.
func finderEvidence(d *primaryDetector) bool {
	const minRawHits = 100
	for _, p := range d.stats.passes {
		if p.rawHits >= minRawHits {
			return true
		}
	}
	return false
}

// finderPassStats records the per-pass finder-detection counters that the
// jabdiag-tagged diagnostic reads off the detector. They are observation only
// and never influence detection.
type finderPassStats struct {
	rawHits        int             // n-1-1-1-m run-length hits (horizontal + conditional vertical scan)
	branchBlue     int             // green seeds where the blue cross-check fired (-> {FP0,FP3} path)
	branchRed      int             // green seeds where blue failed and the red cross-check fired (-> {FP1,FP2} path)
	redColor       int             // red-path candidates passing the inner core-colour check (fp2found)
	redClassified  int             // red-path candidates matched to fp1/fp2 by classifyFinderPattern
	crossSurvivors [4]int          // candidates passing crossCheckPattern, by finder type
	preprune       [4]int          // selectBestPatterns group sizes before the 0.5*maxFound prune
	selected       [4]int          // foundCount of the selected pattern per type after the prune (0 = absent)
	missing        int             // types absent after selection
	status         int             // findPrimarySymbol status for the pass
	interpolated   bool            // whether the single-missing-finder estimate fired
	candidates     []finderPattern // merged finder candidates this pass (pre-prune)
}

// detectorStats aggregates finder-detection instrumentation across the up-to-two
// binarization passes locateFinders runs.
type detectorStats struct {
	passes []finderPassStats // one entry per findPrimarySymbol pass
	rgbAvg [3]float32        // retry thresholds from getAveragePixelValue, between passes
}

// primaryDetector orchestrates primary-symbol finder detection over the three
// binarized channels. Its findPrimarySymbol/selectBestPatterns/scanPatternVertical
// methods populate stats, the single source of truth for the diagnostic. The ch
// field is a by-value [3]*bitmap: the retry's re-binarization (locateFinders) is
// scoped to this detector and never leaks into secondary decoding.
type primaryDetector struct {
	bm         *bitmap
	ch         [3]*bitmap
	mode       int
	fps        []finderPattern
	candidates []finderPattern // last pass's pre-prune candidates, for the geometric quad fallback
	stats      detectorStats
}

// pass returns the current (last-appended) finder pass's stats.
func (d *primaryDetector) pass() *finderPassStats {
	return &d.stats.passes[len(d.stats.passes)-1]
}

// detectPrimary locates the primary symbol's finder patterns, rectifies and
// samples the symbol, and decodes it, falling back to alignment-pattern
// resampling if the finder-pattern sample fails.
func detectPrimary(d *primaryDetector, symbol *decodedSymbol) bool {
	// Ports detectMaster in detector.c.
	if !d.locateFinders() {
		return false
	}
	fps := d.fps

	sideSize := calculateSideSize(fps)
	if sideSize.X == -1 || sideSize.Y == -1 {
		// Per-type selection scores each finder type's best by foundCount, not
		// geometry, so on a noisy capture it can choose four candidates that do not
		// form a symbol quad. Retry once with a geometric consensus over all
		// candidates before giving up.
		if quad, ok := d.selectFinderQuadByGeometry(); ok {
			copy(fps, quad[:])
			sideSize = calculateSideSize(fps)
		}
		if sideSize.X == -1 || sideSize.Y == -1 {
			return false
		}
	}

	pt := getPerspectiveTransform(fps[0].center, fps[1].center, fps[2].center, fps[3].center, sideSize)
	matrix := sampleSymbol(d.bm, pt, sideSize)
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
	sv := symbol.meta.sideVersion
	if sv.X < 1 || sv.X > 32 || sv.Y < 1 || sv.Y > 32 {
		// The metadata was not fully read (decodePrimary failed before the version
		// was known), so the alignment-pattern geometry would be derived from an
		// unset version and the resample would read out of bounds. Give up instead.
		return false
	}
	symbol.sideSize = image.Pt(spec.VersionToSize(sv.X), spec.VersionToSize(sv.Y))
	apMatrix := sampleSymbolByAlignmentPattern(d.bm, d.ch, symbol, fps)
	if apMatrix == nil {
		return false
	}
	return decodePrimary(apMatrix, symbol) == jabSuccess
}

// locateFinders runs the finder search, falling back to a finder-seeded second
// binarization pass on failure. The retry re-binarizes d.ch in place; because the
// channel array is held by value, that swap is scoped to this detector and does
// not propagate to secondary detection.
func (d *primaryDetector) locateFinders() bool {
	// Ports the retry orchestration of detectMaster in detector.c.
	status := d.findPrimarySymbol()
	if status == fatalError {
		return false
	}
	if status == jabSuccess {
		return true
	}

	// Retry 1: re-binarize using adaptive thresholds from around the found patterns.
	rgbAvg := getAveragePixelValue(d.bm, d.fps)
	d.stats.rgbAvg = rgbAvg
	ch2 := binarizerRGB(d.bm, rgbAvg[:])
	d.ch[0], d.ch[1], d.ch[2] = ch2[0], ch2[1], ch2[2]
	if d.findPrimarySymbol() == jabSuccess {
		return true
	}

	// Retry 2 (descreen): screen captures inject the display's subpixel/diode lattice
	// and moiré, which can leave the raw and avg-RGB passes without enough surviving
	// finders. Estimate the lattice pitch per image and low-pass ≈ one grid cell (then
	// a coarser pass) before binarizing — the kernel is derived, not a fixed radius.
	// bm is left untouched so colour sampling still reads the original pixels; the
	// d.ch swap stays primary-scoped.
	px, py := estimatePitch(d.bm)
	for _, r := range descreenSchedule(px, py) {
		chN := binarizerRGB(descreen(d.bm, r[0], r[1]), nil)
		d.ch[0], d.ch[1], d.ch[2] = chN[0], chN[1], chN[2]
		if d.findPrimarySymbol() == jabSuccess {
			return true
		}
	}
	return false
}
