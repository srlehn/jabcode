// Package read coordinates detection and decoding into the full JAB Code
// reading pipeline: it owns the orientation and region-of-interest retries,
// the detect-then-decode handoff for the primary symbol (including the
// alignment-pattern fallback that needs the decoded side version), and the
// docked-secondary walk that derives each secondary's geometry from its
// decoded host metadata.
package read

import (
	"errors"
	"image"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/detect"
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
// A large capture rarely needs its full resolution - only small-module symbols do - so
// Decode searches a resolution pyramid: box-halved levels of the frame decode
// concurrently and the coarsest success wins (see decodePyramid). Small images run the
// single full-resolution search directly and behave exactly as before.
//
// Within one level, finder detection collapses beyond ~20 degrees of rotation, so an
// upright read alone misses a rotated capture. The search recovers it coarse-to-fine:
// try upright first (clean captures resolve here and stay byte-identical), and on
// failure find the promising orientations on a downscaled copy before spending a
// full-resolution decode only on those few rungs. The decoded bytes are
// orientation-independent, so the first orientation that reads wins. The downscaled
// orientation search bounds the cost of a failed read by the probe resolution rather
// than the capture's megapixels - which also means a symbol small within a large frame
// can vanish in the probe downscale, so as the last resort the same orientation search
// runs per proposed region of interest, spending the bounded probe resolution on the
// region instead of the whole frame.
func Decode(img image.Image) ([]byte, error) {
	if levels := pyramidLevels(img); levels != nil {
		if data, _, _, ok := decodePyramid(levels); ok {
			return data, nil
		}
		return nil, errDecodeFailed
	}
	if data, _, ok := decodeSearch(img, nil); ok {
		return data, nil
	}
	return nil, errDecodeFailed
}

// decodeSearch runs the full single-resolution read ladder on img: upright,
// then the coarse orientation rungs, then per-region orientation retries. On
// success deg reports the pre-rotation that read (0 for upright) - the
// hypothesis a Stream reuses on its next frame. A non-nil quit is polled
// between ladder stages; once it reports true the search returns early with
// ok=false (the pyramid cancels levels that can no longer win this way,
// bounding their wasted work to one stage).
func decodeSearch(img image.Image, quit func() bool) (data []byte, deg float64, ok bool) {
	data, ok, evidence := decodeBitmap(core.BitmapFromImage(img), quit)
	if ok {
		return data, 0, true
	}
	// A blank or near-uniform image has no finder structure at any orientation, so skip
	// the rotation search entirely - the cheap uniform bailout.
	if !evidence || (quit != nil && quit()) {
		return nil, 0, false
	}
	return decodeRetries(img, quit)
}

// decodeRetries is decodeSearch after a failed upright read: the orientation
// rungs, then the per-region retries. The pyramid runs it as its second phase,
// only once every level's upright attempt has failed. A region win reports the
// rung angle like a whole-frame win - the orientation holds for the frame even
// though the read happened on a crop.
func decodeRetries(img image.Image, quit func() bool) (data []byte, deg float64, ok bool) {
	// Spend a full-resolution decode only on the orientations the coarse search found
	// promising; counter-rotating a strongly-rotated code to near upright restores the
	// integer run-lengths its single-module finders need.
	for _, deg := range detect.CoarseOrientationRungs(img) {
		if quit != nil && quit() {
			return nil, 0, false
		}
		if data, ok, _ := decodeBitmap(detect.RotateToBitmap(img, deg), quit); ok {
			return data, deg, true
		}
	}
	if quit != nil && quit() {
		return nil, 0, false
	}
	// Region-of-interest retry: probe orientation per proposed region at the
	// region's own scale, restoring the module resolution a small symbol loses
	// in the whole-frame probe downscale. A region spanning the full frame
	// would repeat the search above at the same scale, so it is skipped.
	for _, roi := range detect.ProposeROIs(img, maxDecodeROIs) {
		if roi.Bounds == img.Bounds() {
			continue
		}
		crop := detect.CropImage(img, roi.Bounds)
		for _, deg := range detect.CoarseOrientationRungs(crop) {
			if quit != nil && quit() {
				return nil, 0, false
			}
			if data, ok, _ := decodeBitmap(detect.RotateToBitmap(crop, deg), quit); ok {
				return data, deg, true
			}
		}
	}
	return nil, 0, false
}

// DecodeImage attempts one full read of img as given: binarize, locate and decode
// the primary symbol, then its docked secondaries, then assemble the message. It
// runs the entire session on one image so the primary, the alignment-pattern
// fallback and the secondaries share a single coherent coordinate frame. evidence
// reports whether the finder search saw any finder structure at all, so Decode can
// skip the rotation search outright on blank or near-uniform input.
func DecodeImage(img image.Image) (data []byte, ok, evidence bool) {
	return decodeBitmap(core.BitmapFromImage(img), nil)
}

// decodeBitmap is DecodeImage on an already-converted bitmap, so the rotation
// rungs can resample straight into decoder layout without an image in between.
// A non-nil quit is handed to the finder search, which polls it between its
// binarization passes and abandons the remaining retries once it reports true.
func decodeBitmap(bm *core.Bitmap, quit func() bool) (data []byte, ok, evidence bool) {
	// Ports decodeJABCode/decodeJABCodeEx (NORMAL_DECODE mode) in detector.c.
	detect.BalanceRGB(bm)
	if quit != nil && quit() {
		return nil, false, false
	}
	ch := detect.BinarizerRGB(bm, nil)
	if quit != nil && quit() {
		return nil, false, false
	}

	symbols := make([]core.DecodedSymbol, maxSymbolNumber)
	d := &detect.PrimaryDetector{BM: bm, Ch: ch, Mode: detect.IntensiveDetect, Quit: quit}
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
	n := 0
	for i := 0; i < total; i++ {
		n += len(symbols[i].Data)
	}
	bits := make([]byte, 0, n)
	for i := 0; i < total; i++ {
		bits = append(bits, symbols[i].Data...)
	}
	return decode.DecodeData(bits), true, evidence
}

// finderEvidence reports whether the upright finder search saw any finder structure at
// all - the cheap uniform bailout that lets Decode skip the rotation search on blank or
// near-uniform input. It gates on raw run-length hits (the n-1-1-1-m seed scan), which
// are rotation-robust: a code produces hundreds at every angle (the rotation gating
// measurement) even when the cross-check survivors collapse, whereas a blank image
// produces almost none. It deliberately does not try to judge orientation - that is the
// coarse search's job; a structured non-code image clears this gate and is then rejected
// by the coarse search finding no orientation with aligned finders.
func finderEvidence(d *detect.PrimaryDetector) bool {
	const minRawHits = 100
	for _, p := range d.Stats.Passes {
		if p.RawHits >= minRawHits {
			return true
		}
	}
	return false
}

// detectPrimary locates the primary symbol's finder patterns, rectifies and
// samples the symbol, and decodes it, falling back to alignment-pattern
// resampling if the finder-pattern sample fails.
func detectPrimary(d *detect.PrimaryDetector, symbol *core.DecodedSymbol) bool {
	// Ports detectMaster in detector.c.
	if !d.LocateFinders() {
		return false
	}
	fps := d.FPs

	sideSize := detect.CalculateSideSize(d.BM, fps)
	if sideSize.X == -1 || sideSize.Y == -1 {
		// Per-type selection scores each finder type's best by foundCount, not
		// geometry, so on a noisy capture it can choose four candidates that do not
		// form a symbol quad. Retry once with a geometric consensus over all
		// candidates before giving up.
		if quad, ok := d.SelectFinderQuadByGeometry(); ok {
			copy(fps, quad[:])
			sideSize = detect.CalculateSideSize(d.BM, fps)
		}
		if sideSize.X == -1 || sideSize.Y == -1 {
			return false
		}
	}

	pt := core.PerspectiveTransform(fps[0].Center, fps[1].Center, fps[2].Center, fps[3].Center, sideSize)
	// A print-level detection samples each channel where its colorant plane
	// actually landed: misregistered planes displace every channel's content
	// from the finder grid, and the offset search recovers the displacement.
	var matrix *core.Bitmap
	if d.PrintDetected() {
		matrix = detect.SampleSymbolOffset(d.BM, pt, sideSize, detect.SearchChannelOffsets(d.BM, pt, sideSize))
	} else {
		matrix = detect.SampleSymbol(d.BM, pt, sideSize)
	}
	if matrix == nil {
		return false
	}

	symbol.Index = 0
	symbol.HostIndex = 0
	symbol.SideSize = sideSize
	symbol.ModuleSize = (fps[0].ModuleSize + fps[1].ModuleSize + fps[2].ModuleSize + fps[3].ModuleSize) / 4.0
	for i := range 4 {
		symbol.PatternPositions[i] = fps[i].Center
	}

	switch res := decode.DecodePrimary(matrix, symbol); {
	case res == core.Success:
		return true
	case res < 0: // fatal error occurred
		return false
	}

	// if decoding using only finder patterns failed, try decoding using alignment patterns
	sv := symbol.Meta.SideVersion
	if sv.X < 1 || sv.X > 32 || sv.Y < 1 || sv.Y > 32 {
		// The metadata was not fully read (DecodePrimary failed before the version
		// was known), so the alignment-pattern geometry would be derived from an
		// unset version and the resample would read out of bounds. Give up instead.
		return false
	}
	symbol.SideSize = image.Pt(spec.VersionToSize(sv.X), spec.VersionToSize(sv.Y))
	apMatrix := detect.SampleSymbolByAlignmentPattern(d.BM, d.Ch, symbol, fps)
	if apMatrix == nil {
		return false
	}
	return decode.DecodePrimary(apMatrix, symbol) == core.Success
}

// decodeDockedSecondaries detects and decodes every secondary symbol docked to a
// host symbol.
func decodeDockedSecondaries(bm *core.Bitmap, ch [3]*core.Bitmap, symbols []core.DecodedSymbol, hostIndex int, total *int) bool {
	// Ports decodeDockedSlaves in detector.c.
	dp := symbols[hostIndex].Meta.DockedPosition
	docked := [4]int{dp & 0x08, dp & 0x04, dp & 0x02, dp & 0x01}
	for j := range 4 {
		if docked[j] > 0 && *total < maxSymbolNumber {
			symbols[*total].Index = *total
			symbols[*total].HostIndex = hostIndex
			symbols[*total].Meta = symbols[hostIndex].SecondaryMeta[j]
			matrix := detect.DetectSecondary(bm, ch, &symbols[hostIndex], &symbols[*total], j)
			if matrix == nil {
				return false
			}
			if decode.DecodeSecondary(matrix, &symbols[*total]) > 0 {
				*total++
			} else {
				return false
			}
		}
	}
	return true
}
