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

// errDecodeFailed is returned when no orientation of img yields a readable symbol.
var errDecodeFailed = errors.New("jabcode: detecting or decoding the JAB Code failed")

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

// finding is the detection geometry a read route publishes instead of dropping
// it on a failure exit: where the primary symbol's finder quad sits, at which
// module side size, under which physical finder signature and pre-rotation.
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
	deg     float64             // pre-rotation of the canvas the quad was located on
	payload *Message            // full decoded message when the route also decoded
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
		if data, _, _, ok := decodePyramidCapabilities(p, tr, capabilities); ok {
			return data, nil
		}
		return nil, errDecodeFailed
	}
	// The search abandons losing route slots without joining them, so every
	// slot input must be decoder-owned memory: convert the caller's image
	// exactly once here (the pyramid path gets the same guarantee from its
	// level copies). A straggler may then briefly keep reading this base
	// after the decode returns without racing the caller's buffer reuse.
	if data, _, ok := decodeSearchCapabilities(pyramidBase(img), nil, tr, capabilities); ok {
		return data, nil
	}
	return nil, errDecodeFailed
}

// decodeSearch runs the full single-resolution read ladder on img: upright,
// then the coarse orientation rungs, then per-region orientation retries. img
// must be decoder-owned memory (a pyramid level or a pyramidBase conversion),
// never a caller's image: the ladder's losing route slots are not joined and
// may keep reading it briefly after the search returns. On
// success deg reports the pre-rotation that read (0 for upright) - the
// hypothesis a Stream reuses on its next frame. A non-nil quit is polled
// between ladder stages; once it reports true the search returns early with
// ok=false (the pyramid cancels levels that can no longer win this way,
// bounding their wasted work to one stage). Route attempts are collected into
// tr (nil to skip).
func decodeSearch(img image.Image, quit func() bool, tr *routeTrace) (data []byte, deg float64, ok bool) {
	message, deg, ok := decodeSearchCapabilities(img, quit, tr, compiledCapabilities())
	return messageTransmission(message), deg, ok
}

func decodeSearchOnly(img image.Image, quit func() bool, tr *routeTrace, variant wire.Variant) (data []byte, deg float64, ok bool) {
	message, deg, ok := decodeSearchCapabilities(img, quit, tr, variant.Mask())
	return messageTransmission(message), deg, ok
}

func decodeSearchCapabilities(img image.Image, quit func() bool, tr *routeTrace, capabilities wire.Capabilities) (data *Message, deg float64, ok bool) {
	var f finding
	detail := tr.beginAttempt("upright", 0, -1)
	data, stage, evidence := decodeBitmapFindingTracedCapabilities(core.BitmapFromImage(img), quit, &f, detail, capabilities)
	tr.finishAttempt(routeAttempt{deg: 0, roi: -1, stage: stage, side: f.side}, detail, messageTransmission(data))
	if stage == readDecoded {
		return data, 0, true
	}
	// A blank or near-uniform image has no finder structure at any orientation, so skip
	// the rotation search entirely - the cheap uniform bailout.
	if !evidence || (quit != nil && quit()) {
		return nil, 0, false
	}
	return decodeRetriesFindingCapabilities(img, quit, nil, nil, tr, capabilities)
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
	message, deg, ok := decodeRetriesFindingCapabilities(img, quit, f, rungs, tr, compiledCapabilities())
	return messageTransmission(message), deg, ok
}

func decodeRetriesFindingOnly(img image.Image, quit func() bool, f *finding, rungs []float64, tr *routeTrace, variant wire.Variant) (data []byte, deg float64, ok bool) {
	message, deg, ok := decodeRetriesFindingCapabilities(img, quit, f, rungs, tr, variant.Mask())
	return messageTransmission(message), deg, ok
}

func decodeRetriesFindingCapabilities(img image.Image, quit func() bool, f *finding, rungs []float64, tr *routeTrace, capabilities wire.Capabilities) (data *Message, deg float64, ok bool) {
	return decodeRetriesFindingGPUCapabilities(
		levelImageOf(img),
		quit,
		f,
		rungs,
		tr,
		capabilities,
		nil,
		-1,
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
	data   *Message
	deg    float64
	stage  readStage
	rf     finding
	canvas image.Point
	srcW   int
	srcH   int
	off    image.Point
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
) (data *Message, deg float64, ok bool) {
	if count == 0 {
		return nil, 0, false
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
			r.rf.toImage(r.deg, r.canvas.X, r.canvas.Y, r.srcW, r.srcH, r.off)
			r.rf.payload = cloneMessage(r.data)
			*f = r.rf
		}
		if decoded {
			return r.data, r.deg, true
		}
	}
	return nil, 0, false
}

func decodeRetriesFindingGPUCapabilities(
	li levelImage,
	quit func() bool,
	f *finding,
	rungs []float64,
	tr *routeTrace,
	capabilities wire.Capabilities,
	gpuSession *detect.GPUDecodeSession,
	gpuLevel int,
) (data *Message, deg float64, ok bool) {
	if rungs == nil {
		rungs = orientationRungs(li.load(), tr, "full frame", -1)
	}
	// Spend a full-resolution decode only on the orientations the coarse search
	// found promising; counter-rotating a strongly-rotated code to near upright
	// restores the integer run-lengths its single-module finders need. The
	// rungs run concurrently with results committed in ladder order. The
	// upright attempt already ran (this ladder only starts after it failed),
	// so the zero rung would repeat the same canvas and binarizations; region
	// rungs below keep their zero rung - no upright ran on a crop.
	frameRungs := make([]float64, 0, len(rungs))
	for _, deg := range rungs {
		if deg != 0 {
			frameRungs = append(frameRungs, deg)
		}
	}
	full := image.Rect(0, 0, li.size.X, li.size.Y)
	data, deg, ok = runRouteSlots(quit, tr, f, len(frameRungs),
		func(slot int, slotQuit func() bool, slotTr *routeTrace) routeSlotResult {
			deg := frameRungs[slot]
			var rf finding
			detail := slotTr.beginAttempt("rotated", deg, -1)
			data, stage, _, canvasSize := decodeRouteFindingCapabilities(
				li.load,
				full,
				deg,
				slotQuit,
				&rf,
				detail,
				capabilities,
				gpuSession,
				gpuLevel,
			)
			slotTr.finishAttempt(routeAttempt{deg: deg, roi: -1, stage: stage, side: rf.side}, detail, messageTransmission(data))
			return routeSlotResult{
				data: data, deg: deg, stage: stage, rf: rf,
				canvas: canvasSize, srcW: li.size.X, srcH: li.size.Y,
			}
		})
	if ok {
		return data, deg, true
	}
	if quit != nil && quit() {
		return nil, 0, false
	}
	// From here every route needs CPU pixels: region proposal, the region
	// probes and the region rotations all read the frame directly.
	img := li.load()
	b := img.Bounds()
	// Region-of-interest retry: probe orientation per proposed region at the
	// region's own scale, restoring the module resolution a small symbol loses
	// in the whole-frame probe downscale. A region spanning the full frame
	// would repeat the search above at the same scale, so it is skipped.
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
		rungs  []float64
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
			plan.rungs = roiRungsTraced(plan.crop, plan.tr, plan.index)
		}()
	}
	probes.Wait()
	type roiSlot struct {
		plan *roiPlan
		deg  float64
	}
	var slots []roiSlot
	for _, plan := range plans {
		tr.merge(plan.tr)
		for _, deg := range plan.rungs {
			slots = append(slots, roiSlot{plan: plan, deg: deg})
		}
	}
	return runRouteSlots(quit, tr, f, len(slots),
		func(index int, slotQuit func() bool, slotTr *routeTrace) routeSlotResult {
			s := slots[index]
			var rf finding
			detail := slotTr.beginAttempt("roi", s.deg, s.plan.index)
			data, stage, _, canvasSize := decodeRouteFindingCapabilities(
				func() image.Image { return s.plan.crop },
				s.plan.bounds.Sub(b.Min),
				s.deg,
				slotQuit,
				&rf,
				detail,
				capabilities,
				gpuSession,
				gpuLevel,
			)
			slotTr.finishAttempt(routeAttempt{deg: s.deg, roi: s.plan.index, stage: stage, side: rf.side}, detail, messageTransmission(data))
			return routeSlotResult{
				data: data, deg: s.deg, stage: stage, rf: rf,
				canvas: canvasSize, srcW: s.plan.crop.Rect.Dx(), srcH: s.plan.crop.Rect.Dy(),
				off: s.plan.off,
			}
		})
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
	return roiRungsTraced(crop, nil, -1)
}

func orientationRungs(img image.Image, tr *routeTrace, label string, roi int) []float64 {
	if tr == nil || !tr.detailed {
		return detect.CoarseOrientationRungs(img)
	}
	families, probe := detect.CoarseProbeFamiliesTraced(img)
	rungs := detect.FamiliesToRungs(families)
	tr.probes = append(tr.probes, DiagnosticProbe{
		Level: tr.level, ROI: roi, Label: label, Probe: probe,
		Rungs: append([]float64(nil), rungs...),
	})
	return rungs
}

func roiRungsTraced(crop *image.NRGBA, tr *routeTrace, roi int) []float64 {
	if rungs := orientationRungs(crop, tr, fmt.Sprintf("ROI %d", roi), roi); len(rungs) > 0 {
		return rungs
	}
	levels := pyramidLevels(crop)
	if levels == nil {
		return nil
	}
	for k, lvl := range levels[1:] {
		var fams []detect.CoarseFamily
		if tr != nil && tr.detailed {
			var probe detect.CoarseProbeTrace
			fams, probe = detect.CoarseProbeFamiliesWithinTraced(lvl, detect.CoarseMaxDim<<(k+1))
			rungs := detect.FamiliesToRungsUncapped(fams)
			tr.probes = append(tr.probes, DiagnosticProbe{
				Level: tr.level, ROI: roi,
				Label: fmt.Sprintf("ROI %d escalation %d", roi, k+1),
				Probe: probe, Rungs: append([]float64(nil), rungs...),
			})
			if len(rungs) > 0 {
				return rungs
			}
			continue
		}
		fams = detect.CoarseProbeFamiliesWithin(lvl, detect.CoarseMaxDim<<(k+1))
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
	ch := detect.BinarizerRGB(bm, nil)
	if detail != nil {
		detail.InitialChannels = ch
	}
	if quit != nil && quit() {
		return nil, readAborted, false
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

func decodeRouteFindingCapabilities(
	cpuImage func() image.Image,
	gpuCrop image.Rectangle,
	angle float64,
	quit func() bool,
	f *finding,
	detail *DiagnosticAttempt,
	capabilities wire.Capabilities,
	session *detect.GPUDecodeSession,
	level int,
) (data *Message, stage readStage, evidence bool, size image.Point) {
	if session != nil {
		var trace *detect.DetectorTrace
		if detail != nil {
			trace = &detail.DetectorTrace
		}
		d, foundFinders, gpuSize, err := session.LocateRouteFamilies(
			level,
			gpuCrop,
			angle,
			finderFamiliesForCapabilities(capabilities),
			detect.IntensiveDetect,
			quit,
			trace,
		)
		if err == nil && d != nil {
			data, stage, evidence = decodeGPUDetectorCapabilities(
				d,
				foundFinders,
				f,
				detail,
				capabilities,
			)
			return data, stage, evidence, gpuSize
		}
		// A quit-cancelled acquisition must not burn a full-resolution CPU
		// rotation for a route that already lost; genuine GPU errors keep
		// their CPU fallback.
		if quit != nil && quit() {
			return nil, readAborted, false, image.Point{}
		}
	}
	release, ok := acquireCPURouteBody(quit)
	if !ok {
		return nil, readAborted, false, image.Point{}
	}
	defer release()
	bm := detect.RotateToBitmap(cpuImage(), angle)
	data, stage, evidence = decodeBitmapFindingTracedCapabilities(
		bm,
		quit,
		f,
		detail,
		capabilities,
	)
	return data, stage, evidence, image.Pt(bm.Width, bm.Height)
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

func decodeLocatedDetector(
	d *detect.PrimaryDetector,
	foundFinders detect.FinderFamilySet,
	f *finding,
	detail *DiagnosticAttempt,
	capabilities wire.Capabilities,
) (data *Message, stage readStage, evidence bool) {
	bm := d.BM
	stage = readNoFinders
	evidence = finderEvidence(d)
	wantHistorical := capabilities.Has(wire.BSI) || capabilities.Has(wire.PreV2C)

	if capabilities&currentFamilyCapabilities != 0 && foundFinders.Has(detect.FinderFamilyCurrent) {
		d.SelectFinderFamily(detect.FinderFamilyCurrent)
		base := core.DecodedSymbol{}
		matrix, currentStage := sampleLocatedPrimaryTraced(d, detect.FinderFamilyCurrent, &base, f, detail)
		stage = currentStage
		if currentStage == readSampled {
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
				traceStart := primaryTraceCount(detail)
				symbol := base
				symbol.WireVariant = variant
				variantStage := decodePrimaryMatrixTraced(d, matrix, &symbol, detail, moduleEvidenceCache, alignmentCache)
				normalizeCurrentVariant(&symbol, detail, capabilities, traceStart)
				if variantStage > stage {
					stage = variantStage
				}
				if variantStage != readDecoded {
					continue
				}
				// The docked traversal reads mask pixels, which a GPU-located
				// detector defers until a consumer needs them.
				if !d.EnsureChannels() {
					stage = readSampled
					continue
				}
				symbols := make([]core.DecodedSymbol, maxSymbolNumber)
				symbols[0] = symbol
				data, ok := decodeSymbolsTraced(bm, d.Ch, symbols, 1, detail)
				if !ok {
					stage = readSampled
					continue
				}
				if f != nil && f.located {
					f.payload = cloneMessage(data)
				}
				if detail != nil {
					detail.FinalChannels = d.Ch
					detail.Detector = d.Stats
					if len(d.FPs) >= 4 {
						detail.Finders = append([]detect.FinderPattern(nil), d.FPs[:4]...)
					}
					detail.PrintDetected = d.PrintDetected()
				}
				return data, readDecoded, evidence
			}
		}
		if detail != nil {
			detail.FinalChannels = d.Ch
			detail.Detector = d.Stats
			if len(d.FPs) >= 4 {
				detail.Finders = append([]detect.FinderPattern(nil), d.FPs[:4]...)
			}
			detail.PrintDetected = d.PrintDetected()
		}
	}

	if wantHistorical && foundFinders.Has(detect.FinderFamilyBSI) {
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
		return nil, readNoFinders
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
	// consensus over all candidates. The consensus quad is adopted only when it
	// passes the scale-agreement and perspective gates itself, so an
	// already-consistent selection is left untouched and a good quad is never
	// traded for a worse one.
	if sideSize.X == -1 || sideSize.Y == -1 || !detect.ConsistentFinderQuad(fps) {
		if quad, ok := d.SelectFinderQuadByGeometry(); ok {
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
	// Alignment-pattern seeking and default-mode size confirmation read mask
	// pixels, which a GPU-located detector defers until a consumer needs them.
	if !d.EnsureChannels() {
		return readSampled
	}
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
