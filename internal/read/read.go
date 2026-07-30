// Package read coordinates detection and decoding into the full JAB Code
// reading pipeline: it owns the orientation and region-of-interest retries,
// the detect-then-decode handoff for the primary symbol (including the
// alignment-pattern fallback that needs the decoded side version), and the
// docked-secondary walk that derives each secondary's geometry from its
// decoded host metadata.
package read

import (
	"errors"
	"fmt"
	"image"
	"math"
	"reflect"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/wire"
)

// maxSymbolNumber is the maximum number of symbols in a JAB Code.
const maxSymbolNumber = 61

// maxDerivedPixels accounts for the largest image-derived allocation: the
// enlarged retry can expand a frame by enlargeFactor in both dimensions.
const maxDerivedPixels = 128 * 1024 * 1024

// maxImagePixels limits caller-controlled allocations and the corresponding
// pyramid work before the reader has had a chance to inspect the image.
const maxImagePixels = maxDerivedPixels / (enlargeFactor * enlargeFactor)

// errDecodeFailed is returned when no orientation of img yields a readable symbol.
var errDecodeFailed = errors.New("jabcode: detecting or decoding the JAB Code failed")

var errInvalidImage = errors.New("jabcode: invalid image")

func validateImage(img image.Image) error {
	if img == nil {
		return errInvalidImage
	}
	v := reflect.ValueOf(img)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		if v.IsNil() {
			return errInvalidImage
		}
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 {
		return errInvalidImage
	}
	if h > math.MaxInt/4 || w > math.MaxInt/(h*4) {
		return errInvalidImage
	}
	if w > maxImagePixels/h {
		return errInvalidImage
	}
	if err := validateRasterStorage(img, b); err != nil {
		return errInvalidImage
	}
	return nil
}

// planeOffset returns y*stride + x when every step stays within int, reporting
// false on overflow or a negative input. Stride is caller-controlled and only
// the pixel count is bounded above, so a large stride can wrap the product; a
// wrapped offset compares as in range while the rows between the endpoints are
// not, which is why this is computed rather than asserted.
func planeOffset(y, stride, x int) (int, bool) {
	if y < 0 || stride < 0 || x < 0 {
		return 0, false
	}
	if y != 0 && stride != 0 && y > math.MaxInt/stride {
		return 0, false
	}
	p := y * stride
	if p > math.MaxInt-x {
		return 0, false
	}
	return p + x, true
}

// planeFits reports whether rows of rowBytes each, laid out at stride from
// base, stay inside length. Every intermediate is overflow-checked, so the span
// is monotone in the row index and checking the last row covers all of them.
func planeFits(base, stride, rows, rowBytes, length int) bool {
	if base < 0 || rows < 0 || rowBytes < 0 || stride < rowBytes {
		return false
	}
	if rows == 0 {
		return true
	}
	span, ok := planeOffset(rows-1, stride, rowBytes)
	if !ok {
		return false
	}
	if base > math.MaxInt-span {
		return false
	}
	return base+span <= length
}

func validateRasterStorage(img image.Image, b image.Rectangle) error {
	validRows := func(base, stride, pixelBytes, length int) error {
		if stride < 0 || pixelBytes <= 0 {
			return errInvalidImage
		}
		w := b.Dx()
		if w < 0 || w > math.MaxInt/pixelBytes {
			return errInvalidImage
		}
		if !planeFits(base, stride, b.Dy(), w*pixelBytes, length) {
			return errInvalidImage
		}
		return nil
	}
	// The chroma and alpha planes carry their own strides and lengths, so each
	// is bounded against the subsampled extent it is actually indexed by.
	validPlane := func(minX, minY, maxX, maxY, stride, length int) error {
		last, ok := planeOffset(maxY-minY, stride, maxX-minX)
		if !ok || stride < maxX-minX+1 || last >= length {
			return errInvalidImage
		}
		return nil
	}
	switch src := img.(type) {
	case *image.NRGBA:
		return validRows(src.PixOffset(b.Min.X, b.Min.Y), src.Stride, 4, len(src.Pix))
	case *image.RGBA:
		return validRows(src.PixOffset(b.Min.X, b.Min.Y), src.Stride, 4, len(src.Pix))
	case *image.NRGBA64:
		return validRows(src.PixOffset(b.Min.X, b.Min.Y), src.Stride, 8, len(src.Pix))
	case *image.RGBA64:
		return validRows(src.PixOffset(b.Min.X, b.Min.Y), src.Stride, 8, len(src.Pix))
	case *image.Gray:
		return validRows(src.PixOffset(b.Min.X, b.Min.Y), src.Stride, 1, len(src.Pix))
	case *image.Gray16:
		return validRows(src.PixOffset(b.Min.X, b.Min.Y), src.Stride, 2, len(src.Pix))
	case *image.CMYK:
		return validRows(src.PixOffset(b.Min.X, b.Min.Y), src.Stride, 4, len(src.Pix))
	case *image.Paletted:
		if len(src.Palette) > 256 {
			return errInvalidImage
		}
		if err := validRows(src.PixOffset(b.Min.X, b.Min.Y), src.Stride, 1, len(src.Pix)); err != nil {
			return err
		}
		for y := b.Min.Y; y < b.Max.Y; y++ {
			row := src.Pix[src.PixOffset(b.Min.X, y):]
			for x := 0; x < b.Dx(); x++ {
				if int(row[x]) >= len(src.Palette) {
					return errInvalidImage
				}
			}
		}
		return nil
	case *image.Alpha:
		return validRows(src.PixOffset(b.Min.X, b.Min.Y), src.Stride, 1, len(src.Pix))
	case *image.Alpha16:
		return validRows(src.PixOffset(b.Min.X, b.Min.Y), src.Stride, 2, len(src.Pix))
	case *image.YCbCr:
		return validYCbCrPlanes(src, b, validPlane)
	case *image.NYCbCrA:
		// NYCbCrA is classified safe for concurrent reads, so its alpha plane is
		// indexed from parallel workers through AOffset, which bounds-checks
		// nothing. Validating only the embedded YCbCr would leave that plane
		// free to be inconsistent with the rectangle.
		if err := validYCbCrPlanes(&src.YCbCr, b, validPlane); err != nil {
			return err
		}
		return validPlane(src.Rect.Min.X, src.Rect.Min.Y, b.Max.X-1, b.Max.Y-1, src.AStride, len(src.A))
	}
	return nil
}

// validYCbCrPlanes bounds the luma and both chroma planes of src against b.
// Chroma is indexed at the subsampled resolution, so its extent is taken from
// COffset's own divisors rather than from the full rectangle.
func validYCbCrPlanes(src *image.YCbCr, b image.Rectangle, validPlane func(minX, minY, maxX, maxY, stride, length int) error) error {
	if src.YStride <= 0 || src.CStride <= 0 {
		return errInvalidImage
	}
	if b.Max.X-1 < b.Min.X || b.Max.Y-1 < b.Min.Y {
		return errInvalidImage
	}
	if err := validPlane(src.Rect.Min.X, src.Rect.Min.Y, b.Max.X-1, b.Max.Y-1, src.YStride, len(src.Y)); err != nil {
		return err
	}
	c0 := src.COffset(b.Min.X, b.Min.Y)
	c1 := src.COffset(b.Max.X-1, b.Max.Y-1)
	if c0 < 0 || c1 < c0 || c1 >= len(src.Cb) || c1 >= len(src.Cr) {
		return errInvalidImage
	}
	return nil
}

// compiledCapabilities is the additive decoder capability mask. ISO is always
// present; build tags only add readers and never replace or reprioritize it.
func compiledCapabilities() wire.Capabilities {
	capabilities := wire.ISO23634.Mask()
	if highColorReadEnabled {
		capabilities |= wire.ISOHighColor.Mask()
	}
	if bsiReadEnabled {
		capabilities |= wire.BSI.Mask()
	}
	if currentCReadEnabled {
		capabilities |= wire.CurrentC.Mask()
	}
	if preV2CReadEnabled {
		capabilities |= wire.PreV2C.Mask()
	}
	return capabilities
}

// CompiledCapabilities reports the decoder variants included in this build.
// It is internal API for the CLI's oracle-only selector and capability tests;
// normal callers use Decode and automatically receive the whole set.
func CompiledCapabilities() wire.Capabilities { return compiledCapabilities() }

// maxDecodeROIs bounds how many proposed regions the region-of-interest retry
// probes. The proposer ranks regions by score and a symbol's dense colourful
// texture dominates that ranking, so the true region is expected at the front;
// the cap keeps a failed read's cost bounded on cluttered images.
const maxDecodeROIs = 2

const (
	// enlargeFactor is the one step of the enlarged detection scale
	// (decodeEnlarged). One doubling halves the quantization step the
	// run-length checks round against, which is the whole mechanism; cost
	// grows with the square of the factor, and wider steps add none of it.
	enlargeFactor = 2
	// enlargedTraceLevel stamps that ladder's attempts, which are otherwise
	// indistinguishable from the frame-scale ones (levels count from 0 up,
	// and the single-scale path uses -1).
	enlargedTraceLevel = -2
)

// finding is the detection geometry a read route publishes instead of dropping
// it on a failure exit: where the primary symbol's finder quad sits, at which
// module side size, and under which physical finder signature.
// Another route can re-enter the decode directly at this geometry on a
// different pyramid level - scaling the quad instead of re-running the finder
// search (see decodeSeeded). The quad and module sizes are stored in the
// coordinates of the image the route searched (unrotated, uncropped), so they
// transfer across scales by plain scaling.
type finding struct {
	quad    [4]core.PointF      // finder centers, image coordinates
	sizes   [4]float64          // per-corner module sizes, image scale
	side    image.Point         // module side size from the locate
	family  detect.FinderFamily // physical signature that produced the geometry
	payload *Message            // full decoded message when the route also decoded
	located bool
}

// toImage converts a finding located on a region crop back into image
// coordinates. Only the crop origin separates the two frames now; nothing
// resamples a route's pixels, so there is no canvas mapping left to invert.
func (f *finding) toImage(off image.Point) {
	for i := range 4 {
		f.quad[i].X += float64(off.X)
		f.quad[i].Y += float64(off.Y)
	}
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
// Orientation is not searched. The finder scan turns its own scan lines rather than
// the frame, so one upright read of a level covers every orientation that level can
// present and no route resamples pixels to try an angle.
//
// What a level's upright read can still miss is a symbol competing with the rest of a
// large cluttered frame, so as a last resort the read repeats per proposed region of
// interest. A crop carries no more resolution than the level it came from; what it
// changes is that binarization and the finder scan work against the region's own
// statistics.
func Decode(img image.Image) ([]byte, error) {
	return DecodeCapabilities(img, compiledCapabilities())
}

// DecodeMessage decodes img once and returns both raw application data and the
// standards-facing reader transmission produced from the same corrected bits.
func DecodeMessage(img image.Image) (*Message, error) {
	return DecodeMessageCapabilities(img, compiledCapabilities())
}

// DecodeOnly is Decode under one selected internal wire variant.
func DecodeOnly(img image.Image, variant wire.Variant) ([]byte, error) {
	return DecodeCapabilities(img, variant.Mask())
}

// DecodeCapabilities is Decode with every wire format enabled by capabilities. The
// mask is additive: one physical locate and sample can be interpreted by each
// compatible wire decoder before the route escalates.
func DecodeCapabilities(img image.Image, capabilities wire.Capabilities) ([]byte, error) {
	message, err := DecodeMessageCapabilities(img, capabilities)
	return messageTransmission(message), err
}

// DecodeMessageCapabilities is DecodeMessage with an explicit additive
// capability set for internal oracle and CLI use.
func DecodeMessageCapabilities(img image.Image, capabilities wire.Capabilities) (*Message, error) {
	if err := validateCapabilities(capabilities); err != nil {
		return nil, err
	}
	if err := validateImage(img); err != nil {
		return nil, err
	}
	return decodeRoutesCapabilities(img, nil, capabilities)
}

func validateCapabilities(capabilities wire.Capabilities) error {
	if !capabilities.Valid() {
		return fmt.Errorf("jabcode: invalid decoder capability set %#x", capabilities)
	}
	if unavailable := capabilities &^ compiledCapabilities(); unavailable != 0 {
		return fmt.Errorf("jabcode: decoder capabilities %#x were not compiled into this build", unavailable)
	}
	return nil
}

// decodeTraced is Decode with the per-route observation trace enabled - the
// diagnostic entry the capture harness reads failure attribution from. The
// trace is complete for failed reads; a successful read may return early with
// a partial one.
func decodeTraced(img image.Image) ([]byte, *routeTrace, error) {
	tr := &routeTrace{level: -1}
	message, err := decodeRoutesCapabilities(img, tr, compiledCapabilities())
	return messageTransmission(message), tr, err
}

// decodeRoutes dispatches a read to the pyramid search or, for small images,
// the single full-resolution search, collecting route attempts into tr (nil
// to skip).
func decodeRoutes(img image.Image, tr *routeTrace) ([]byte, error) {
	message, err := decodeRoutesCapabilities(img, tr, compiledCapabilities())
	return messageTransmission(message), err
}

func decodeRoutesOnly(img image.Image, tr *routeTrace, variant wire.Variant) ([]byte, error) {
	message, err := decodeRoutesCapabilities(img, tr, variant.Mask())
	return messageTransmission(message), err
}

func decodeRoutesCapabilities(img image.Image, tr *routeTrace, capabilities wire.Capabilities) (*Message, error) {
	if p := newPyramid(img); p != nil {
		tr.setLevels(p.count())
		if data, _, ok := decodePyramidCapabilities(p, tr, capabilities); ok {
			return data, nil
		}
		return nil, errDecodeFailed
	}
	tr.setLevels(1)
	// The search abandons losing route slots without joining them, so every
	// slot input must be decoder-owned memory: convert the caller's image
	// exactly once here (the pyramid path gets the same guarantee from its
	// level copies). A straggler may then briefly keep reading this base
	// after the decode returns without racing the caller's buffer reuse.
	if data, ok := decodeSearchCapabilities(pyramidBase(img), nil, tr, capabilities); ok {
		return data, nil
	}
	return nil, errDecodeFailed
}

func decodeSearchCapabilities(img image.Image, quit func() bool, tr *routeTrace, capabilities wire.Capabilities) (data *Message, ok bool) {
	return decodeSearchScaled(img, quit, tr, capabilities, true)
}

// decodeSearchScaled is decodeSearchCapabilities with the enlarged detection
// scale as an explicit stage. baseScale distinguishes the two ladders: the
// frame's own scale runs every stage and may escalate, the enlarged copy runs
// the whole-frame stages only and never escalates again.
func decodeSearchScaled(
	img image.Image,
	quit func() bool,
	tr *routeTrace,
	capabilities wire.Capabilities,
	baseScale bool,
) (data *Message, ok bool) {
	var f finding
	detail := tr.beginAttempt(-1)
	data, stage, evidence := decodeBitmapFindingTracedCapabilities(core.BitmapFromImage(img), quit, &f, detail, capabilities)
	tr.finishAttempt(routeAttempt{kind: "upright", roi: -1, stage: stage, side: f.side}, detail, messageTransmission(data))
	if stage == readDecoded {
		return data, true
	}
	// A blank or near-uniform image has no finder structure to region-search
	// for either - the cheap uniform bailout.
	if !evidence || (quit != nil && quit()) {
		return nil, false
	}
	if data, ok := decodeRetriesRegionsCapabilities(img, quit, nil, baseScale, tr, capabilities); ok {
		return data, true
	}
	if !baseScale || stage != readNoFinders || (quit != nil && quit()) {
		return nil, false
	}
	return decodeEnlarged(img, quit, tr, capabilities)
}

// decodeEnlarged reruns the whole-frame ladder once on an interpolated
// enlargement. It is the answer to the one failure the rest cannot
// address: run-length seeds by the hundred (the evidence gate above) and no
// finder confirmed anywhere, which is what a capture looks like when its
// modules are too small for the cross-checks rather than too damaged.
//
// The frame must be small enough that no primary symbol placement could have
// resolved in it (detect.SmallestVerifiableFrame: the largest primary side at
// the cross-check module floor). Interpolation invents no detail, so it is
// worth paying exactly where the capture itself is the limit. A larger frame
// that failed did so for some other reason, and enlarging it would multiply
// the cost of the slowest reads in the set to no purpose.
//
// The step is a plain doubling because no module-scale measurement exists at
// this point to size a larger one: before the cross-checks confirm anything,
// the only population available is the raw seeds, whose median follows
// whatever texture dominates the frame rather than the symbol. Enlarging
// cannot add information; it moves the module edges off the pixel grid that
// the run-length quantization rounds against, which is the whole failure.
func decodeEnlarged(
	img image.Image,
	quit func() bool,
	tr *routeTrace,
	capabilities wire.Capabilities,
) (data *Message, ok bool) {
	b := img.Bounds()
	if min(b.Dx(), b.Dy()) >= detect.SmallestVerifiableFrame() {
		return nil, false
	}
	// The enlarged ladder repeats the upright, rotation and region kinds, so
	// its attempts are stamped with their own trace level: a diagnostic reader
	// must be able to tell which scale an attempt ran on.
	sub := &routeTrace{level: enlargedTraceLevel}
	if tr != nil {
		sub.detailed = tr.detailed
	}
	data, ok = decodeSearchScaled(detect.UpscaleNRGBA(nrgbaBase(img), enlargeFactor), quit, sub, capabilities, false)
	tr.merge(sub)
	return data, ok
}

func decodeRetriesRegionsCapabilities(img image.Image, quit func() bool, f *finding, regions bool, tr *routeTrace, capabilities wire.Capabilities) (data *Message, ok bool) {
	return decodeRetriesRegionsLevel(
		levelImageOf(img),
		quit,
		f,
		regions,
		tr,
		capabilities,
	)
}

// levelImage hands the search ladder its CPU-side frame lazily: the
// dimensions are known without pixels, and load materializes the frame on
// first CPU consumption - a route that stays on the GPU never pays for it.
type levelImage struct {
	size image.Point
	load func() image.Image
}

// levelImageOf wraps an already materialized frame.
func levelImageOf(img image.Image) levelImage {
	b := img.Bounds()
	return levelImage{size: image.Pt(b.Dx(), b.Dy()), load: func() image.Image { return img }}
}

// cpuRouteBodies bounds how many full-canvas CPU route bodies run at once
// across the process. Each body already fans its pixel passes over every core
// (core.ParallelRows), so running more bodies than cores adds peak canvas
// memory and scheduler pressure without adding throughput; an escalated
// search's fan-out is bounded here instead of per call site. The bound is a
// scheduling crossover, not an image-processing scale; it never changes which
// routes run or what they return (bodies are pure and results commit in slot
// order), only when they start.
var cpuRouteBodies = make(chan struct{}, max(2, runtime.GOMAXPROCS(0)))

// acquireCPURouteBody blocks until a CPU route body slot frees, then rechecks
// quit: a route that lost while it waited releases the slot and reports it
// should not run. The caller must release() exactly once when ok.
func acquireCPURouteBody(quit func() bool) (release func(), ok bool) {
	cpuRouteBodies <- struct{}{}
	if quit != nil && quit() {
		<-cpuRouteBodies
		return nil, false
	}
	return func() { <-cpuRouteBodies }, true
}

// routeSlotResult is one concurrent route slot's outcome plus the geometry
// needed to convert its finding into image coordinates during the ordered
// commit.
type routeSlotResult struct {
	data  *Message
	stage readStage
	rf    finding
	off   image.Point
}

// runRouteSlots runs count route slots concurrently and commits their results
// in slot order, reproducing the sequential ladder's semantics: the first
// slot that decodes wins, every located slot up to the winner updates f under
// the same rule the sequential ladder used (a decode always publishes its
// finding, a locate-only result never overwrites an earlier locate), and slot
// traces merge in slot order up to the winner. Slots after a winner are told
// to quit through their quit hook and are not waited for - each slot writes
// only its own result, so stragglers cannot corrupt the commit, but they may
// keep reading their inputs briefly after the search returns, which is why
// every slot input must be decoder-owned memory. The committed outcome is
// independent of scheduling as long as every slot's route body is
// deterministic; context admission on an adapter that reports its memory
// size is a pure function of the frame and the device, so the remaining
// timing-dependence is genuine device-memory exhaustion - an adapter without
// a reported size, or memory lost to other users of the device - where a
// route's CPU-or-GPU backend choice can flip and a serial decode the
// backends correct differently can then differ between runs (the documented
// determinism caveat in ARCHITECTURE.md).
func runRouteSlots(
	quit func() bool,
	tr *routeTrace,
	f *finding,
	count int,
	run func(slot int, slotQuit func() bool, slotTr *routeTrace) routeSlotResult,
) (data *Message, ok bool) {
	if count == 0 {
		return nil, false
	}
	results := make([]routeSlotResult, count)
	done := make([]chan struct{}, count)
	traces := make([]*routeTrace, count)
	if tr != nil {
		for slot := range traces {
			traces[slot] = &routeTrace{level: tr.level, detailed: tr.detailed}
		}
	}
	var winner atomic.Int64
	winner.Store(int64(count))
	for slot := range results {
		done[slot] = make(chan struct{})
		go func() {
			defer close(done[slot])
			slotQuit := func() bool {
				return (quit != nil && quit()) || winner.Load() < int64(slot)
			}
			if slotQuit() {
				results[slot] = routeSlotResult{stage: readAborted}
				return
			}
			results[slot] = run(slot, slotQuit, traces[slot])
			if results[slot].stage != readDecoded {
				return
			}
			for {
				w := winner.Load()
				if int64(slot) >= w || winner.CompareAndSwap(w, int64(slot)) {
					return
				}
			}
		}()
	}
	for slot := range results {
		<-done[slot]
		tr.merge(traces[slot])
		r := &results[slot]
		decoded := r.stage == readDecoded
		if r.rf.located && f != nil && (decoded || !f.located) {
			r.rf.toImage(r.off)
			r.rf.payload = cloneMessage(r.data)
			*f = r.rf
		}
		if decoded {
			return r.data, true
		}
	}
	return nil, false
}

func decodeRetriesRegionsLevel(
	li levelImage,
	quit func() bool,
	f *finding,
	regions bool,
	tr *routeTrace,
	capabilities wire.Capabilities,
) (data *Message, ok bool) {
	// No route resamples the frame to search another angle. The directional
	// finder scan reads orientation out of the symbol's own basis, so the
	// upright attempt that already ran covered every orientation this level
	// can offer, and a second pass over rotated pixels would only re-ask a
	// question the scan has already answered.
	//
	// Regions survive that, but not for the reason the rotation ladder had.
	// Cropping cannot restore resolution: a crop of a level carries exactly the
	// pixels that level already had, so a symbol's pixels per module are the
	// same either way. What a crop changes is context - binarization
	// thresholds are computed over the region instead of the whole frame, and
	// the finder scan sees the symbol without the rest of the frame's clutter
	// competing for its candidate budget.
	if !regions || (quit != nil && quit()) {
		return nil, false
	}
	// From here every route needs CPU pixels: region proposal, the region
	// probes and the region rotations all read the frame directly.
	img := li.load()
	b := img.Bounds()
	// Region-of-interest retry: read each proposed region on its own, so
	// binarization and the finder scan work against the region's statistics
	// rather than the whole frame's. A region spanning the full frame would
	// repeat the upright read exactly, so it is skipped.
	var rois []detect.ROICandidate
	if tr != nil && tr.detailed {
		var tileMap detect.ROITileMap
		rois, tileMap = detect.ProposeROIsTraced(img, maxDecodeROIs)
		tr.rois = append(tr.rois, DiagnosticROIs{
			Level: tr.level, Image: img, TileMap: tileMap,
			Candidates: append([]detect.ROICandidate(nil), rois...),
		})
	} else {
		rois = detect.ProposeROIs(img, maxDecodeROIs)
	}
	// Probe every region concurrently first - each probe is a pure function of
	// its crop and the plans keep proposal order, so both the probe traces and
	// the flattened route order stay deterministic.
	type roiPlan struct {
		index  int
		bounds image.Rectangle
		crop   *image.NRGBA
		off    image.Point
		tr     *routeTrace
	}
	plans := make([]*roiPlan, 0, len(rois))
	for r, roi := range rois {
		if roi.Bounds == img.Bounds() {
			continue
		}
		plans = append(plans, &roiPlan{index: r, bounds: roi.Bounds})
	}
	var probes sync.WaitGroup
	for _, plan := range plans {
		probes.Add(1)
		go func() {
			defer probes.Done()
			plan.crop = detect.CropImage(img, plan.bounds)
			plan.off = plan.bounds.Intersect(img.Bounds()).Min.Sub(b.Min)
			if tr != nil {
				plan.tr = &routeTrace{level: tr.level, detailed: tr.detailed}
			}
		}()
	}
	probes.Wait()
	for _, plan := range plans {
		tr.merge(plan.tr)
	}
	// One route per region: no upright read ran on a crop yet, and nothing
	// presents the crop at another orientation.
	return runRouteSlots(quit, tr, f, len(plans),
		func(index int, slotQuit func() bool, slotTr *routeTrace) routeSlotResult {
			plan := plans[index]
			var rf finding
			detail := slotTr.beginAttempt(plan.index)
			data, stage, _ := decodeRouteFindingCapabilities(
				func() image.Image { return plan.crop },
				slotQuit,
				&rf,
				detail,
				capabilities,
			)
			slotTr.finishAttempt(routeAttempt{kind: "roi", roi: plan.index, stage: stage, side: rf.side}, detail, messageTransmission(data))
			return routeSlotResult{
				data: data, stage: stage, rf: rf,
				off: plan.off,
			}
		})
}

// DecodeImage attempts one full read of img as given: binarize, locate and decode
// the primary symbol, then its docked secondaries, then assemble the message. It
// runs the entire session on one image so the primary, the alignment-pattern
// fallback and the secondaries share a single coherent coordinate frame. evidence
// reports whether the finder search saw any finder structure at all, so Decode can
// skip the region search outright on blank or near-uniform input.
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
	return decodeBitmapFindingTraced(bm, quit, f, nil)
}

func decodeBitmapFindingTraced(bm *core.Bitmap, quit func() bool, f *finding, detail *DiagnosticAttempt) (data []byte, stage readStage, evidence bool) {
	message, stage, evidence := decodeBitmapFindingTracedCapabilities(bm, quit, f, detail, compiledCapabilities())
	return messageTransmission(message), stage, evidence
}

func decodeBitmapFindingTracedOnly(bm *core.Bitmap, quit func() bool, f *finding, detail *DiagnosticAttempt, variant wire.Variant) (data []byte, stage readStage, evidence bool) {
	message, stage, evidence := decodeBitmapFindingTracedCapabilities(bm, quit, f, detail, variant.Mask())
	return messageTransmission(message), stage, evidence
}

func decodeBitmapFindingTracedCapabilities(bm *core.Bitmap, quit func() bool, f *finding, detail *DiagnosticAttempt, capabilities wire.Capabilities) (data *Message, stage readStage, evidence bool) {
	// Ports decodeJABCode/decodeJABCodeEx (NORMAL_DECODE mode) in detector.c.
	detect.BalanceRGB(bm)
	if detail != nil {
		detail.Balanced = bm
	}
	if quit != nil && quit() {
		return nil, readAborted, false
	}
	ch, ok := detect.BinarizerRGBUntil(bm, nil, quit)
	if !ok {
		return nil, readAborted, false
	}
	if detail != nil {
		detail.InitialChannels = ch
	}
	stage = readNoFinders
	d := &detect.PrimaryDetector{BM: bm, Ch: ch, Mode: detect.IntensiveDetect, Quit: quit}
	if detail != nil {
		d.Trace = &detail.DetectorTrace
	}
	wantedFinders := finderFamiliesForCapabilities(capabilities)
	foundFinders := d.LocateFinderFamilies(wantedFinders)
	return decodeLocatedDetector(d, foundFinders, f, detail, capabilities)
}

func decodeBitmapFindingGPUCapabilities(
	quit func() bool,
	f *finding,
	detail *DiagnosticAttempt,
	capabilities wire.Capabilities,
	session *detect.GPUDecodeSession,
	level int,
) (data *Message, stage readStage, evidence bool, handled bool) {
	if session == nil {
		return nil, readNoFinders, false, false
	}
	var trace *detect.DetectorTrace
	if detail != nil {
		trace = &detail.DetectorTrace
	}
	d, foundFinders, err := session.LocateLevelFamilies(
		level,
		finderFamiliesForCapabilities(capabilities),
		detect.IntensiveDetect,
		quit,
		trace,
	)
	if err != nil || d == nil {
		return nil, readNoFinders, false, false
	}
	data, stage, evidence = decodeGPUDetectorCapabilities(
		d,
		foundFinders,
		f,
		detail,
		capabilities,
	)
	return data, stage, evidence, true
}

func decodeGPUDetectorCapabilities(
	d *detect.PrimaryDetector,
	foundFinders detect.FinderFamilySet,
	f *finding,
	detail *DiagnosticAttempt,
	capabilities wire.Capabilities,
) (data *Message, stage readStage, evidence bool) {
	if detail != nil {
		detail.Balanced = d.BM
		if len(detail.DetectorTrace.PassChannels) > 0 {
			detail.InitialChannels = detail.DetectorTrace.PassChannels[0]
		}
	}
	return decodeLocatedDetector(d, foundFinders, f, detail, capabilities)
}

// decodeRouteFindingCapabilities reads one region crop on the CPU. There is no
// device path here any more: the resident route canvas existed to rotate a
// level, and a crop that is never rotated is just pixels the CPU already has.
func decodeRouteFindingCapabilities(
	cpuImage func() image.Image,
	quit func() bool,
	f *finding,
	detail *DiagnosticAttempt,
	capabilities wire.Capabilities,
) (data *Message, stage readStage, evidence bool) {
	release, ok := acquireCPURouteBody(quit)
	if !ok {
		return nil, readAborted, false
	}
	defer release()
	return decodeBitmapFindingTracedCapabilities(
		core.BitmapFromImage(cpuImage()),
		quit,
		f,
		detail,
		capabilities,
	)
}

func decodePyramidLevelFindingCapabilities(
	img func() image.Image,
	quit func() bool,
	f *finding,
	detail *DiagnosticAttempt,
	capabilities wire.Capabilities,
	session *detect.GPUDecodeSession,
	level int,
) (data *Message, stage readStage, evidence bool) {
	if data, stage, evidence, handled := decodeBitmapFindingGPUCapabilities(
		quit,
		f,
		detail,
		capabilities,
		session,
		level,
	); handled {
		return data, stage, evidence
	}
	release, ok := acquireCPURouteBody(quit)
	if !ok {
		return nil, readAborted, false
	}
	defer release()
	return decodeBitmapFindingTracedCapabilities(
		core.BitmapFromImage(img()),
		quit,
		f,
		detail,
		capabilities,
	)
}

func finderFamiliesForCapabilities(capabilities wire.Capabilities) detect.FinderFamilySet {
	wanted := detect.FinderFamilySet(0)
	if capabilities&currentFamilyCapabilities != 0 {
		wanted |= detect.FinderFamilyCurrent.Mask()
	}
	if capabilities.Has(wire.BSI) || capabilities.Has(wire.PreV2C) {
		wanted |= detect.FinderFamilyBSI.Mask()
	}
	return wanted
}

type currentFinderHypothesisResult struct {
	data     *Message
	stage    readStage
	finding  finding
	patterns [4]detect.FinderPattern
	corner   detect.CornerSource
	detail   *DiagnosticAttempt
}

func newCurrentFinderHypothesisDetail(base *DiagnosticAttempt) *DiagnosticAttempt {
	if base == nil {
		return nil
	}
	result := *base
	result.FinalChannels = [3]*core.Bitmap{}
	result.Finders = nil
	result.PrintDetected = false
	result.Side = image.Point{}
	result.Transform = core.Perspective{}
	result.HasTransform = false
	result.ChannelOffsets = [3]core.PointF{}
	result.Sampled = nil
	result.Primary = nil
	result.Alignments = nil
	result.Secondaries = nil
	result.Payload = nil
	result.FinderHypotheses = 0
	result.AmbiguousFinders = false
	return &result
}

func decodeCurrentFinderHypothesis(
	d *detect.PrimaryDetector,
	corner detect.CornerSource,
	detail *DiagnosticAttempt,
	capabilities wire.Capabilities,
) currentFinderHypothesisResult {
	result := currentFinderHypothesisResult{
		stage:  readNoFinders,
		corner: corner,
		detail: newCurrentFinderHypothesisDetail(detail),
	}
	if result.detail != nil {
		result.detail.FinderCorner = corner
	}
	base := core.DecodedSymbol{}
	matrix, currentStage := sampleLocatedPrimaryTraced(
		d,
		detect.FinderFamilyCurrent,
		&base,
		&result.finding,
		result.detail,
	)
	result.stage = currentStage
	if currentStage != readSampled || d.Quitting() {
		finishCurrentFinderHypothesisDetail(d, result.detail)
		return result
	}

	variants, variantCount := currentObservationVariants(capabilities)
	var moduleEvidence decode.ModuleEvidenceCache
	var moduleEvidenceCache *decode.ModuleEvidenceCache
	var alignmentSamples alignmentSampleCache
	var alignmentCache *alignmentSampleCache
	if shareCurrentFamilyEvidence && variantCount > 1 {
		moduleEvidenceCache = &moduleEvidence
		alignmentCache = &alignmentSamples
	}
	for _, variant := range variants[:variantCount] {
		if d.Quitting() {
			break
		}
		traceStart := primaryTraceCount(result.detail)
		symbol := base
		symbol.WireVariant = variant
		variantStage := decodePrimaryMatrixTraced(
			d,
			matrix,
			&symbol,
			result.detail,
			moduleEvidenceCache,
			alignmentCache,
		)
		normalizeCurrentVariant(&symbol, result.detail, capabilities, traceStart)
		if variantStage != readDecoded {
			continue
		}
		symbols := make([]core.DecodedSymbol, maxSymbolNumber)
		symbols[0] = symbol
		data, ok := decodeSymbolsTraced(d.BM, d.Ch, symbols, 1, result.detail)
		if !ok {
			continue
		}
		result.data = data
		result.stage = readDecoded
		result.finding.payload = cloneMessage(data)
		break
	}
	finishCurrentFinderHypothesisDetail(d, result.detail)
	return result
}

func finishCurrentFinderHypothesisDetail(d *detect.PrimaryDetector, detail *DiagnosticAttempt) {
	if detail == nil {
		return
	}
	detail.FinalChannels = d.Ch
	detail.Detector = d.Stats
	if len(d.FPs) >= 4 {
		detail.Finders = append([]detect.FinderPattern(nil), d.FPs[:4]...)
		detail.FindersFamily, _ = d.ActiveFinderFamily()
	}
	detail.PrintDetected = d.PrintDetected()
}

// mergeDecodedFinderHypothesis retains the first successful geometry while all
// later successes agree on the full interpreted message. A disagreement is not
// ranked by geometry: hard correction has no payload integrity check, so neither
// result is safe to publish.
func mergeDecodedFinderHypothesis(winner *currentFinderHypothesisResult, candidate currentFinderHypothesisResult) bool {
	if winner.data == nil {
		*winner = candidate
		winner.data = cloneMessage(candidate.data)
		return true
	}
	return equalMessages(winner.data, candidate.data)
}

func commitCurrentFinderHypothesis(
	d *detect.PrimaryDetector,
	f *finding,
	detail *DiagnosticAttempt,
	result currentFinderHypothesisResult,
) {
	if len(d.FPs) >= 4 {
		copy(d.FPs[:4], result.patterns[:])
	}
	if f != nil {
		*f = result.finding
	}
	if detail != nil && result.detail != nil {
		*detail = *result.detail
	}
}

// decodeLocatedDetector interprets an already-located detector: it samples the
// primary symbol, then corrects and interprets the payload for each enabled
// wire variant.
//
// It polls the detector's quit hook only after sampling, never before it. A
// cancelled route still publishes the geometry it located into f, which is what
// the coarsest level feeds to the seeded route; aborting during the locate
// would withhold that seed and hand the read to a lower-priority route. After
// sampling the seed is already published and everything remaining is payload
// work whose result a lost route cannot use.
func decodeLocatedDetector(
	d *detect.PrimaryDetector,
	foundFinders detect.FinderFamilySet,
	f *finding,
	detail *DiagnosticAttempt,
	capabilities wire.Capabilities,
) (data *Message, stage readStage, evidence bool) {
	stage = readNoFinders
	evidence = finderEvidence(d)
	wantHistorical := capabilities.Has(wire.BSI) || capabilities.Has(wire.PreV2C)

	if capabilities&currentFamilyCapabilities != 0 && foundFinders.Has(detect.FinderFamilyCurrent) {
		d.SelectFinderFamily(detect.FinderFamilyCurrent)
		hypotheses := d.FinderQuadHypotheses(detect.FinderFamilyCurrent)
		var winner, best currentFinderHypothesisResult
		haveWinner, haveBest, ambiguous := false, false, false
		tried := 0
		for _, hypothesis := range hypotheses {
			if len(d.FPs) < 4 {
				break
			}
			copy(d.FPs[:4], hypothesis.Patterns[:])
			result := decodeCurrentFinderHypothesis(d, hypothesis.Corner, detail, capabilities)
			tried++
			copy(result.patterns[:], d.FPs[:4])
			if result.stage > stage {
				stage = result.stage
			}
			if !haveBest || result.stage > best.stage {
				best, haveBest = result, true
			}
			if result.stage == readDecoded {
				if !mergeDecodedFinderHypothesis(&winner, result) {
					ambiguous = true
					break
				}
				haveWinner = true
			}
			if d.Quitting() {
				break
			}
		}
		if ambiguous {
			winner.stage = readSampled
			winner.data = nil
			winner.finding.payload = nil
			if winner.detail != nil {
				winner.detail.AmbiguousFinders = true
				winner.detail.FinderHypotheses = tried
			}
			commitCurrentFinderHypothesis(d, f, detail, winner)
			return nil, readSampled, evidence
		}
		if haveWinner {
			if winner.detail != nil {
				winner.detail.FinderHypotheses = tried
			}
			commitCurrentFinderHypothesis(d, f, detail, winner)
			return winner.data, readDecoded, evidence
		}
		if haveBest {
			if best.detail != nil {
				best.detail.FinderHypotheses = tried
			}
			commitCurrentFinderHypothesis(d, f, detail, best)
		}
	}

	if wantHistorical && foundFinders.Has(detect.FinderFamilyBSI) && !d.Quitting() {
		historicalData, historicalStage, historicalEvidence := decodeHistoricalLocated(d, f, detail, capabilities)
		evidence = evidence || historicalEvidence
		if historicalStage == readDecoded {
			return historicalData, readDecoded, evidence
		}
		if historicalStage > stage {
			stage = historicalStage
		}
	}
	if detail != nil {
		detail.FinalChannels = d.Ch
		detail.Detector = d.Stats
		if len(d.FPs) >= 4 {
			detail.Finders = append([]detect.FinderPattern(nil), d.FPs[:4]...)
			detail.FindersFamily, _ = d.ActiveFinderFamily()
		}
		detail.PrintDetected = d.PrintDetected()
	}
	return nil, stage, evidence
}

// normalizeCurrentVariant gives a low-color observation made under the
// permissive ISO high-color representative its stricter ISO identity when ISO
// is enabled. Both variants use identical physical, palette, PRNG, interleave,
// LDPC and message rules for four and eight colors, so no decode work is
// repeated merely to choose that identity.
func normalizeCurrentVariant(symbol *core.DecodedSymbol, detail *DiagnosticAttempt, capabilities wire.Capabilities, traceStart int) {
	if symbol.WireVariant != wire.ISOHighColor || !capabilities.Has(wire.ISO23634) {
		return
	}
	if symbol.Meta.NC <= 2 {
		symbol.WireVariant = wire.ISO23634
	}
	if detail == nil {
		return
	}
	for i := traceStart; i < len(detail.Primary); i++ {
		if detail.Primary[i].Symbol.WireVariant == wire.ISOHighColor && detail.Primary[i].Symbol.Meta.NC <= 2 {
			detail.Primary[i].Symbol.WireVariant = wire.ISO23634
		}
	}
}

func primaryTraceCount(detail *DiagnosticAttempt) int {
	if detail == nil {
		return 0
	}
	return len(detail.Primary)
}

// finderEvidence reports whether the upright finder search saw any finder structure at
// all - the cheap uniform bailout that lets Decode skip the region search on blank or
// near-uniform input. It gates on raw run-length hits (the n-1-1-1-m seed scan), which
// are rotation-robust: a code produces hundreds at every angle (the rotation gating
// measurement) even when the cross-check survivors collapse, whereas a blank image
// produces almost none. It deliberately does not try to judge orientation - that is the
// coarse search's job; a structured non-code image clears this gate and is then rejected
// by the coarse search finding no orientation with aligned finders.
func finderEvidence(d *detect.PrimaryDetector) bool {
	const minRawHits = 100
	for _, p := range d.Stats.Passes {
		bsi, _ := p.BSIFamilyStats()
		if p.RawHits >= minRawHits || bsi.RawHits >= minRawHits {
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
	return observePrimaryTraced(d, symbol, f, nil)
}

func observePrimaryTraced(d *detect.PrimaryDetector, symbol *core.DecodedSymbol, f *finding, detail *DiagnosticAttempt) (*decode.PrimaryObservation, readStage) {
	matrix, stage := samplePrimaryTraced(d, symbol, f, detail)
	if stage != readSampled {
		return nil, stage
	}
	obs, _ := observePrimaryMatrix(matrix, symbol, detail)
	return obs, readSampled
}

// samplePrimaryTraced performs the variant-independent current-family work:
// finder location, perspective construction and one module-grid sample. Wire
// metadata and payload interpretation happen after this boundary, so an
// additive variant mask never repeats image preparation or finder detection.
func samplePrimaryTraced(d *detect.PrimaryDetector, symbol *core.DecodedSymbol, f *finding, detail *DiagnosticAttempt) (*core.Bitmap, readStage) {
	// Ports the detection phase of detectMaster in detector.c.
	if !d.LocateFinders() {
		// Greedy per-type selection located no quad on any pass, but the
		// cross-pass candidate union may still hold a consistent quad (or a
		// consistent triple to complete) - the module-scale-floor case where no
		// single pass survives enough finders to locate. Try the geometric
		// consensus before giving up; a wrong grid it assembles is caught by the
		// palette-coherence admission gate.
		if !d.SelectConsensusQuad() {
			return nil, readNoFinders
		}
	}
	return sampleLocatedPrimaryTraced(d, detect.FinderFamilyCurrent, symbol, f, detail)
}

// sampleLocatedPrimaryTraced performs geometry and sampling from the active
// finder result of an already completed integrated traversal. Family records
// which physical signature owns that geometry; it does not select another
// detector route.
func sampleLocatedPrimaryTraced(d *detect.PrimaryDetector, family detect.FinderFamily, symbol *core.DecodedSymbol, f *finding, detail *DiagnosticAttempt) (*core.Bitmap, readStage) {
	fps := d.FPs

	sideSize := detect.CalculateSideSize(d.BM, fps)
	// Per-type finder selection scores each type's best by foundCount with no
	// cross-type geometry, so a noisy capture can let a spurious small-scale
	// cluster win one type and leave the chosen four disagreeing on module scale
	// or not forming a symbol quad (the observed field class A). That surfaces
	// either as an invalid side size or as a plausible-but-wrong side whose
	// degenerate quad samples off the grid, so both route to a geometric
	// consensus over the cross-pass candidate union: first the full
	// four-candidate search, then, when one type has no consistent candidate at
	// all, a consistent-triple search that interpolates the missing corner.
	// Either result is adopted only
	// when it passes the scale-agreement and perspective gates itself, so an
	// already-consistent selection is left untouched and a good quad is never
	// traded for a worse one.
	if sideSize.X == -1 || sideSize.Y == -1 || !detect.ConsistentFinderQuad(fps) {
		if quad, ok := d.SelectFinderQuadByGeometry(); ok {
			copy(fps, quad[:])
			sideSize = detect.CalculateSideSize(d.BM, fps)
		} else if quad, ok := d.SelectFinderQuadByInterpolatedTriple(); ok {
			copy(fps, quad[:])
			sideSize = detect.CalculateSideSize(d.BM, fps)
		}
	}
	if detail != nil {
		detail.Side = sideSize
	}
	if sideSize.X == -1 || sideSize.Y == -1 {
		return nil, readNoSideSize
	}
	if f != nil {
		for i := range 4 {
			f.quad[i] = fps[i].Center
			f.sizes[i] = fps[i].ModuleSize
		}
		f.side = sideSize
		f.family = family
		f.located = true
	}

	pt := core.PerspectiveTransform(fps[0].Center, fps[1].Center, fps[2].Center, fps[3].Center, sideSize)
	if detail != nil {
		detail.Transform = pt
		detail.HasTransform = true
	}
	// A print-level detection samples each channel where its colorant plane
	// actually landed: misregistered planes displace every channel's content
	// from the finder grid, and the offset search recovers the displacement.
	var matrix *core.Bitmap
	if d.PrintDetected() {
		offsets := detect.SearchChannelOffsets(d.BM, pt, sideSize)
		if detail != nil {
			detail.ChannelOffsets = offsets
		}
		matrix = detect.SampleSymbolOffset(d.BM, pt, sideSize, offsets)
	} else {
		matrix = detect.SampleSymbol(d.BM, pt, sideSize)
	}
	if matrix == nil {
		return nil, readNoSample
	}
	if detail != nil {
		detail.Sampled = matrix
	}

	symbol.Index = 0
	symbol.HostIndex = 0
	symbol.SideSize = sideSize
	symbol.ModuleSize = (fps[0].ModuleSize + fps[1].ModuleSize + fps[2].ModuleSize + fps[3].ModuleSize) / 4.0
	for i := range 4 {
		symbol.PatternPositions[i] = fps[i].Center
	}

	return matrix, readSampled
}

func observePrimaryMatrix(matrix *core.Bitmap, symbol *core.DecodedSymbol, detail *DiagnosticAttempt) (*decode.PrimaryObservation, int) {
	if detail == nil {
		return decode.ObservePrimary(matrix, symbol)
	}
	detail.Primary = append(detail.Primary, decode.PrimaryTrace{})
	return decode.ObservePrimaryTraced(matrix, symbol, &detail.Primary[len(detail.Primary)-1])
}

func admitPrimary(obs *decode.PrimaryObservation, detail *DiagnosticAttempt) bool {
	if obs == nil {
		return false
	}
	admitted := obs.AdmitPayloadCorrection()
	if detail != nil && len(detail.Primary) > 0 {
		detail.Primary[len(detail.Primary)-1].AdmissionChecked = true
		detail.Primary[len(detail.Primary)-1].Admitted = admitted
	}
	return admitted
}

// detectPrimary runs a full primary read: the observation (locate, sample,
// metadata), payload correction on it, and the alignment-pattern fallback
// when the finder-pattern read fails. It reports the furthest stage reached
// (readDecoded on success).
func detectPrimary(d *detect.PrimaryDetector, symbol *core.DecodedSymbol, f *finding) readStage {
	return detectPrimaryTraced(d, symbol, f, nil)
}

func detectPrimaryTraced(d *detect.PrimaryDetector, symbol *core.DecodedSymbol, f *finding, detail *DiagnosticAttempt) readStage {
	// Ports detectMaster in detector.c.
	matrix, stage := samplePrimaryTraced(d, symbol, f, detail)
	if stage != readSampled {
		return stage
	}
	return decodePrimaryMatrixTraced(d, matrix, symbol, detail, nil, nil)
}

// decodePrimaryMatrixTraced interprets one shared current-family sample under
// exactly one wire variant, including its variant-specific alignment fallback.
func decodePrimaryMatrixTraced(d *detect.PrimaryDetector, matrix *core.Bitmap, symbol *core.DecodedSymbol, detail *DiagnosticAttempt, moduleCache *decode.ModuleEvidenceCache, alignmentCache *alignmentSampleCache) readStage {
	obs, _ := observePrimaryMatrix(matrix, symbol, detail)
	if admitPrimary(obs, detail) && correctPrimaryPayload(obs, moduleCache) == core.Success {
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
	apMatrix := samplePrimaryByAlignment(d.BM, d.Ch, symbol, d.FPs, detail, alignmentCache)
	if apMatrix == nil {
		return readSampled
	}
	if apObs, ret := observePrimaryMatrix(apMatrix, symbol, detail); ret == core.Success && admitPrimary(apObs, detail) && correctPrimaryPayload(apObs, moduleCache) == core.Success {
		return readDecoded
	}
	return readSampled
}

func correctPrimaryPayload(obs *decode.PrimaryObservation, cache *decode.ModuleEvidenceCache) int {
	if cache != nil {
		return obs.CorrectPayloadWithCache(cache)
	}
	return obs.CorrectPayload()
}
