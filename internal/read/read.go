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
	"math"

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

// finding is the detection geometry a read route publishes instead of dropping
// it on a failure exit: where the primary symbol's finder quad sits, at which
// module side size, under which pre-rotation. Another route can re-enter the
// decode directly at this geometry on a different pyramid level - scaling the
// quad instead of re-running the finder search (see decodeSeeded). The quad and
// module sizes are stored in the coordinates of the image the route searched
// (unrotated, uncropped), so they transfer across scales by plain scaling.
type finding struct {
	quad    [4]core.PointF // finder centers, image coordinates
	sizes   [4]float64     // per-corner module sizes, image scale
	side    image.Point    // module side size from the locate
	deg     float64        // pre-rotation of the canvas the quad was located on
	payload []byte         // full decoded bytes when the route also decoded
	located bool
}

// toImage converts a finding located on a rotated (and possibly cropped)
// canvas back into image coordinates: the rotation canvas is centred on its
// source (rotateInto's inverse mapping), and a region crop is offset by its
// origin. srcW/srcH are the dimensions of what was rotated (the crop when off
// is set, the image otherwise).
func (f *finding) toImage(deg float64, canvasW, canvasH, srcW, srcH int, off image.Point) {
	rad := deg * math.Pi / 180
	cs, sn := math.Cos(rad), math.Sin(rad)
	ccx, ccy := float64(canvasW)/2, float64(canvasH)/2
	for i := range 4 {
		dx, dy := f.quad[i].X-ccx, f.quad[i].Y-ccy
		f.quad[i] = core.PointF{
			X: cs*dx + sn*dy + float64(srcW)/2 + float64(off.X),
			Y: -sn*dx + cs*dy + float64(srcH)/2 + float64(off.Y),
		}
	}
	f.deg = deg
}

// scale maps a finding between resolutions of the same frame: quad positions
// scale per axis, module sizes by the mean factor.
func (f *finding) scale(sx, sy float64) {
	for i := range 4 {
		f.quad[i].X *= sx
		f.quad[i].Y *= sy
		f.sizes[i] *= (sx + sy) / 2
	}
}

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
	return decodeRoutes(img, nil)
}

// decodeTraced is Decode with the per-route observation trace enabled - the
// diagnostic entry the capture harness reads failure attribution from. The
// trace is complete for failed reads; a successful read may return early with
// a partial one.
func decodeTraced(img image.Image) ([]byte, *routeTrace, error) {
	tr := &routeTrace{level: -1}
	data, err := decodeRoutes(img, tr)
	return data, tr, err
}

// decodeRoutes dispatches a read to the pyramid search or, for small images,
// the single full-resolution search, collecting route attempts into tr (nil
// to skip).
func decodeRoutes(img image.Image, tr *routeTrace) ([]byte, error) {
	if levels := pyramidLevels(img); levels != nil {
		if data, _, _, ok := decodePyramid(levels, tr); ok {
			return data, nil
		}
		return nil, errDecodeFailed
	}
	if data, _, ok := decodeSearch(img, nil, tr); ok {
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
// bounding their wasted work to one stage). Route attempts are collected into
// tr (nil to skip).
func decodeSearch(img image.Image, quit func() bool, tr *routeTrace) (data []byte, deg float64, ok bool) {
	var f finding
	data, stage, evidence := decodeBitmapFinding(core.BitmapFromImage(img), quit, &f)
	tr.add(routeAttempt{deg: 0, roi: -1, stage: stage, side: f.side})
	if stage == readDecoded {
		return data, 0, true
	}
	// A blank or near-uniform image has no finder structure at any orientation, so skip
	// the rotation search entirely - the cheap uniform bailout.
	if !evidence || (quit != nil && quit()) {
		return nil, 0, false
	}
	return decodeRetriesFinding(img, quit, nil, nil, tr)
}

// decodeRetriesFinding runs the ladder after a failed upright read - the
// orientation rungs, then the per-region retries - publishing detection
// findings into f (nil to skip). The pyramid runs it as its second phase,
// only once every level's upright attempt has failed. A region win reports
// the rung angle like a whole-frame win - the orientation holds for the frame
// even though the read happened on a crop. The winning rung's finding always
// wins; among rungs that only located, the first in ladder order is kept -
// the ladder is sequential, so the choice is deterministic. A non-nil rungs
// list replaces the whole-frame orientation probe: the promising angles are
// scale-invariant, so the pyramid probes once on its coarsest level and
// shares the result instead of paying one probe downscale per level (region
// probes stay per crop - a region's content differs from the frame's). Route
// attempts are collected into tr (nil to skip).
func decodeRetriesFinding(img image.Image, quit func() bool, f *finding, rungs []float64, tr *routeTrace) (data []byte, deg float64, ok bool) {
	b := img.Bounds()
	if rungs == nil {
		rungs = detect.CoarseOrientationRungs(img)
	}
	// Spend a full-resolution decode only on the orientations the coarse search found
	// promising; counter-rotating a strongly-rotated code to near upright restores the
	// integer run-lengths its single-module finders need.
	for _, deg := range rungs {
		if deg == 0 {
			// The upright attempt already ran (this ladder only starts after it
			// failed), so a zero rung would repeat the same canvas and
			// binarizations. Region rungs below keep their zero rung - no
			// upright ran on a crop.
			continue
		}
		if quit != nil && quit() {
			return nil, 0, false
		}
		bm := detect.RotateToBitmap(img, deg)
		var rf finding
		data, stage, _ := decodeBitmapFinding(bm, quit, &rf)
		tr.add(routeAttempt{deg: deg, roi: -1, stage: stage, side: rf.side})
		ok := stage == readDecoded
		if rf.located && f != nil && (ok || !f.located) {
			rf.toImage(deg, bm.Width, bm.Height, b.Dx(), b.Dy(), image.Point{})
			rf.payload = data
			*f = rf
		}
		if ok {
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
	for r, roi := range detect.ProposeROIs(img, maxDecodeROIs) {
		if roi.Bounds == img.Bounds() {
			continue
		}
		crop := detect.CropImage(img, roi.Bounds)
		off := roi.Bounds.Intersect(img.Bounds()).Min.Sub(b.Min)
		for _, deg := range roiRungs(crop) {
			if quit != nil && quit() {
				return nil, 0, false
			}
			bm := detect.RotateToBitmap(crop, deg)
			var rf finding
			data, stage, _ := decodeBitmapFinding(bm, quit, &rf)
			tr.add(routeAttempt{deg: deg, roi: r, stage: stage, side: rf.side})
			ok := stage == readDecoded
			if rf.located && f != nil && (ok || !f.located) {
				rf.toImage(deg, bm.Width, bm.Height, crop.Rect.Dx(), crop.Rect.Dy(), off)
				rf.payload = data
				*f = rf
			}
			if ok {
				return data, deg, true
			}
		}
	}
	return nil, 0, false
}

// roiRungs returns the orientation rungs for a region crop: the flat bounded
// probe first (unchanged behaviour whenever it retains anything), then - when
// that probe starves and the crop is large enough to hold a pyramid - the
// same finer-level escalation the frame search uses (doubled resolution
// bound per level, uncapped family retention; see decodePyramid's shared
// probe). A dense multi-code print region can hold a symbol at 3-4 px per
// module under the flat probe bound, below the cross-check floor, even
// though the crop decodes at full resolution once its orientation is known -
// the same starvation the frame-level escalation closed. The coarsest
// pyramid level is skipped: it sits at the flat probe's own scale.
func roiRungs(crop *image.NRGBA) []float64 {
	if rungs := detect.CoarseOrientationRungs(crop); len(rungs) > 0 {
		return rungs
	}
	levels := pyramidLevels(crop)
	if levels == nil {
		return nil
	}
	for k, lvl := range levels[1:] {
		fams := detect.CoarseProbeFamiliesWithin(lvl, detect.CoarseMaxDim<<(k+1))
		if rungs := detect.FamiliesToRungsUncapped(fams); len(rungs) > 0 {
			return rungs
		}
	}
	return nil
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
	data, stage, evidence := decodeBitmapFinding(bm, quit, nil)
	return data, stage == readDecoded, evidence
}

// decodeBitmapFinding is decodeBitmap publishing the primary locate geometry
// into f (nil to skip) and reporting the furthest stage the attempt reached
// (readDecoded on success). The quad is recorded in bm's own coordinates; the
// caller converts it to image coordinates when bm is a rotated or cropped
// canvas (finding.toImage).
func decodeBitmapFinding(bm *core.Bitmap, quit func() bool, f *finding) (data []byte, stage readStage, evidence bool) {
	// Ports decodeJABCode/decodeJABCodeEx (NORMAL_DECODE mode) in detector.c.
	detect.BalanceRGB(bm)
	if quit != nil && quit() {
		return nil, readAborted, false
	}
	ch := detect.BinarizerRGB(bm, nil)
	if quit != nil && quit() {
		return nil, readAborted, false
	}

	symbols := make([]core.DecodedSymbol, maxSymbolNumber)
	d := &detect.PrimaryDetector{BM: bm, Ch: ch, Mode: detect.IntensiveDetect, Quit: quit}
	stage = detectPrimary(d, &symbols[0], f)
	evidence = finderEvidence(d)
	if stage != readDecoded {
		return nil, stage, evidence
	}
	data, ok := decodeSymbols(bm, ch, symbols, 1)
	if !ok {
		// The primary decoded but a docked secondary or the message assembly
		// failed; a sampled matrix existed, no payload came out.
		return nil, readSampled, evidence
	}
	if f != nil && f.located {
		f.payload = data
	}
	return data, readDecoded, evidence
}

// decodeSymbols finishes a read whose primary symbol is decoded in symbols[0]:
// it detects and decodes every docked secondary recursively, then assembles
// and interprets the concatenated bit stream.
func decodeSymbols(bm *core.Bitmap, ch [3]*core.Bitmap, symbols []core.DecodedSymbol, total int) (data []byte, ok bool) {
	for i := 0; i < total && total < maxSymbolNumber; i++ {
		if !decodeDockedSecondaries(bm, ch, symbols, i, &total) {
			return nil, false
		}
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
	return decode.DecodeData(bits), true
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

// observePrimary locates the primary symbol's finder patterns, rectifies and
// samples the symbol, and interprets its metadata - everything up to but
// excluding payload correction. It reports the furthest stage reached:
// readNoFinders, readNoSideSize or readNoSample when the respective step
// failed, readSampled once a matrix was sampled. On readSampled the returned
// observation is non-nil when the metadata interpreted cleanly and nil when
// it did not (symbol.Meta then holds the partial interpretation - the
// alignment-pattern fallback may still use a plausible side version from
// it). A successful locate is published into f (nil to skip) even when a
// later step fails - that geometry is what another pyramid level can resume
// from.
func observePrimary(d *detect.PrimaryDetector, symbol *core.DecodedSymbol, f *finding) (*decode.PrimaryObservation, readStage) {
	// Ports the detection phase of detectMaster in detector.c.
	if !d.LocateFinders() {
		return nil, readNoFinders
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
			return nil, readNoSideSize
		}
	}
	if f != nil {
		for i := range 4 {
			f.quad[i] = fps[i].Center
			f.sizes[i] = fps[i].ModuleSize
		}
		f.side = sideSize
		f.located = true
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
		return nil, readNoSample
	}

	symbol.Index = 0
	symbol.HostIndex = 0
	symbol.SideSize = sideSize
	symbol.ModuleSize = (fps[0].ModuleSize + fps[1].ModuleSize + fps[2].ModuleSize + fps[3].ModuleSize) / 4.0
	for i := range 4 {
		symbol.PatternPositions[i] = fps[i].Center
	}

	obs, _ := decode.ObservePrimary(matrix, symbol)
	return obs, readSampled
}

// detectPrimary runs a full primary read: the observation (locate, sample,
// metadata), payload correction on it, and the alignment-pattern fallback
// when the finder-pattern read fails. It reports the furthest stage reached
// (readDecoded on success).
func detectPrimary(d *detect.PrimaryDetector, symbol *core.DecodedSymbol, f *finding) readStage {
	// Ports detectMaster in detector.c.
	obs, stage := observePrimary(d, symbol, f)
	if stage != readSampled {
		return stage
	}
	if obs != nil && obs.AdmitPayloadCorrection() && obs.CorrectPayload() == core.Success {
		return readDecoded
	}

	// if decoding using only finder patterns failed, try decoding using alignment patterns
	sv := symbol.Meta.SideVersion
	if sv.X < 1 || sv.X > 32 || sv.Y < 1 || sv.Y > 32 {
		// The metadata was not fully read (the observation failed before the
		// version was known), so the alignment-pattern geometry would be derived
		// from an unset version and the resample would read out of bounds. Give
		// up instead.
		return readSampled
	}
	symbol.SideSize = image.Pt(spec.VersionToSize(sv.X), spec.VersionToSize(sv.Y))
	apMatrix := detect.SampleSymbolByAlignmentPattern(d.BM, d.Ch, symbol, d.FPs)
	if apMatrix == nil {
		return readSampled
	}
	if apObs, ret := decode.ObservePrimary(apMatrix, symbol); ret == core.Success && apObs.AdmitPayloadCorrection() && apObs.CorrectPayload() == core.Success {
		return readDecoded
	}
	return readSampled
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
