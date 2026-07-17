package detect

import (
	"fmt"

	"github.com/srlehn/jabcode/internal/core"
)

// FinderFamily identifies one physical primary-finder signature.
type FinderFamily uint8

const (
	// FinderFamilyCurrent is the ISO/current-C finder signature.
	FinderFamilyCurrent FinderFamily = iota
	// FinderFamilyBSI is the primary finder signature defined by BSI
	// TR-03137 and retained by pre-v2.0 releases of the C reference. Its
	// classifier is compiled only when one of those wire variants is enabled.
	FinderFamilyBSI
	finderFamilyCount
)

// FinderFamilySet is the set of physical finder signatures located by one
// integrated detector pass.
type FinderFamilySet uint8

// Mask returns the one-signature set for family.
func (family FinderFamily) Mask() FinderFamilySet {
	if family >= finderFamilyCount {
		return 0
	}
	return 1 << family
}

// Has reports whether set contains family.
func (set FinderFamilySet) Has(family FinderFamily) bool {
	return family < finderFamilyCount && set&(1<<family) != 0
}

// FinderFamilyPassStats records one physical signature's counters inside a
// shared image pass. They are observation only and never influence detection.
type FinderFamilyPassStats struct {
	RawHits        int             // n-1-1-1-m run-length hits (horizontal + conditional vertical scan)
	BranchBlue     int             // green seeds where the blue cross-check fired (-> {FP0,FP3} path)
	BranchRed      int             // green seeds where blue failed and the red cross-check fired (-> {FP1,FP2} path)
	RedColor       int             // red-path candidates passing the inner core-colour check (fp2found)
	RedClassified  int             // red-path candidates matched to fp1/fp2 by core-colour classification
	CrossSurvivors [4]int          // candidates passing crossCheckPattern, by finder type
	Preprune       [4]int          // selectBestPatterns group sizes before the 0.5*maxFound prune
	Selected       [4]int          // FoundCount of the selected pattern per type after the prune (0 = absent)
	Missing        int             // types absent after selection
	Status         int             // findPrimarySymbol status for the pass
	Interpolated   bool            // whether the single-missing-finder estimate fired
	Candidates     []FinderPattern // merged finder candidates this pass (pre-prune)
}

// FinderPassStats records one shared finder-detection pass. The embedded
// counters are for the current signature so existing diagnostics retain their
// field names; tagged builds add the optional BSI-era signature's counters to
// the same pass without enlarging the untagged structure.
type FinderPassStats struct {
	optionalFinderPassStats
	Label string // raw, avg-RGB, descreen or print input shared by all signatures
	FinderFamilyPassStats
}

// DetectorStats aggregates finder-detection instrumentation across the raw,
// average-RGB, descreen and conditional print passes LocateFinders runs.
type DetectorStats struct {
	Passes []FinderPassStats // one entry per prepared image pass
	RGBAvg [3]float32        // retry thresholds from averagePixelValue, between passes
}

// DetectorTrace retains the binarized channels used by each finder pass. Its
// entries align with DetectorStats.Passes. It is populated only when attached
// to a PrimaryDetector by the detailed read trace.
type DetectorTrace struct {
	PassInputs   []*core.Bitmap
	PassChannels [][3]*core.Bitmap
	FinderPasses []FinderPassTrace
}

// FinderPassTrace retains the requested signatures and each successful
// signature's selected quad. It is allocated only for an attached diagnostic
// trace, so ordinary decoding does not retain this rendering state.
type FinderPassTrace struct {
	Families FinderFamilySet
	Finders  [finderFamilyCount][]FinderPattern
}

// PrimaryDetector orchestrates primary-symbol finder detection over the three
// binarized channels. Its findPrimarySymbol/selectBestPatterns/scanPatternVertical
// methods populate stats, the single source of truth for the diagnostic. The Ch
// field is a by-value [3]*core.Bitmap: the retry's re-binarization (LocateFinders)
// is scoped to this detector and never leaks into secondary decoding.
type PrimaryDetector struct {
	BM         *core.Bitmap
	Ch         [3]*core.Bitmap
	Mode       int
	FPs        []FinderPattern
	Candidates []FinderPattern // last pass's pre-prune candidates, for the geometric quad fallback
	Stats      DetectorStats
	Trace      *DetectorTrace

	familyResults [finderFamilyCount]finderFamilyResult

	// Quit, when set, is polled between binarization passes; once it reports
	// true the search abandons its remaining retries and fails. The resolution
	// pyramid cancels levels that can no longer win this way, so an abandoned
	// level stops burning cores within one pass instead of finishing its whole
	// retry ladder in the background.
	Quit func() bool

	// seedModules and bsiFamilySeedModules collect the per-seed module-size
	// estimate of every raw n-1-1-1-m hit for their respective signatures
	// across this detection's passes. Each signature gates its own print retry
	// and derives its own low-pass radius, so enabling another signature cannot
	// perturb the established current-family retry. This is working state kept
	// off Stats so those stay observation-only.
	seedModules          []float64
	bsiFamilySeedModules []float64

	// printPass marks the print-level retry passes, where the finder
	// cross-checks scale their pixel tolerances with the module size:
	// colorant-plane misregistration shifts each channel's pattern and
	// fringes every module boundary by a module fraction, which the fixed
	// 3 px slack of the ported walks cannot absorb. Off elsewhere so the
	// default passes stay byte-compatible. printDetected records that the
	// successful pass was a print-level one, making the per-channel
	// sampling-offset estimate available to the sampler.
	printPass     bool
	printDetected bool
	passFamilies  FinderFamilySet

	// materializeBitmap supplies balanced RGBA pixels only when a compact-mask
	// finder pass proves they are needed for missing-finder confirmation,
	// geometry or sampling. CPU detectors already carry pixels and leave it nil.
	materializeBitmap func() error
	materializeErr    error

	// rowHits carries the device row scan's raw hits for the next
	// findPrimaryFamilies call, which consumes them instead of walking the
	// binarized rows itself; the hits are bit-identical to that walk. Nil or
	// invalid (overflowed) hits keep the CPU walk. Single-shot per pass.
	rowHits *finderPassRowHits
}

type finderFamilyResult struct {
	fps           []FinderPattern
	candidates    []FinderPattern
	channels      [3]*core.Bitmap
	status        int
	printDetected bool
}

type finderPassPreparer interface {
	averagePixelValue([]FinderPattern) ([3]float32, error)
	estimatePitch() (int, int, error)
	// prepare builds one retry pass's input and binarized channels.
	// scanChannels selects the channels whose finder row scan should run
	// where the masks live; a preparer without a device scan returns nil
	// hits and the detector walks the rows itself.
	prepare(rx, ry int, thresholds []float32, printLevels bool, scanChannels uint32) (*core.Bitmap, [3]*core.Bitmap, *finderPassRowHits, error)
}

type cpuFinderPassPreparer struct {
	bm *core.Bitmap
}

func (preparer cpuFinderPassPreparer) averagePixelValue(fps []FinderPattern) ([3]float32, error) {
	return averagePixelValue(preparer.bm, fps), nil
}

func (preparer cpuFinderPassPreparer) estimatePitch() (int, int, error) {
	px, py := EstimatePitch(preparer.bm)
	return px, py, nil
}

func (preparer cpuFinderPassPreparer) prepare(
	rx, ry int,
	thresholds []float32,
	printLevels bool,
	scanChannels uint32,
) (*core.Bitmap, [3]*core.Bitmap, *finderPassRowHits, error) {
	input := preparer.bm
	if rx > 0 || ry > 0 {
		input = descreen(input, rx, ry)
	}
	if printLevels {
		return input, BinarizerRGBPrint(input), nil, nil
	}
	return input, BinarizerRGB(input, thresholds), nil, nil
}

// SelectFinderFamily selects one located signature as the detector's active
// finder list for geometry and sampling. It returns false when that signature
// did not form a usable finder quad in the last integrated search.
func (d *PrimaryDetector) SelectFinderFamily(family FinderFamily) bool {
	if family >= finderFamilyCount {
		return false
	}
	result := &d.familyResults[family]
	d.FPs = result.fps
	d.Candidates = result.candidates
	if result.status == core.Success {
		d.Ch = result.channels
		d.printDetected = result.printDetected
	}
	return result.status == core.Success
}

// PrintDetected reports whether the successful finder pass was a print-level
// one, which is the gate for the per-channel sampling-offset search.
func (d *PrimaryDetector) PrintDetected() bool { return d.printDetected }

// ccSlack returns the cross-check pixel slack for a candidate of the given
// module size: the ported constant 3 normally, half a module (misregistration
// fringes scale with the module) in the print-level passes.
func (d *PrimaryDetector) ccSlack(moduleSize float64) int {
	if d.printPass {
		return max(3, int(moduleSize/2+0.5))
	}
	return 3
}

// pass returns the current (last-appended) finder pass's stats.
func (d *PrimaryDetector) pass() *FinderPassStats {
	return &d.Stats.Passes[len(d.Stats.Passes)-1]
}

func (d *PrimaryDetector) recordTracePass(input *core.Bitmap) {
	if d.Trace == nil {
		return
	}
	d.Trace.PassInputs = append(d.Trace.PassInputs, input)
	d.Trace.PassChannels = append(d.Trace.PassChannels, d.Ch)
	pass := FinderPassTrace{Families: d.passFamilies}
	for family := FinderFamily(0); family < finderFamilyCount; family++ {
		result := &d.familyResults[family]
		if result.status == core.Success {
			pass.Finders[family] = append([]FinderPattern(nil), result.fps[:4]...)
		}
	}
	d.Trace.FinderPasses = append(d.Trace.FinderPasses, pass)
}

func (d *PrimaryDetector) ensureBitmap() bool {
	if d == nil || d.BM == nil {
		return false
	}
	want := d.BM.Width * d.BM.Height * d.BM.Channels
	if want > 0 && len(d.BM.Pix) >= want {
		return true
	}
	if d.materializeBitmap == nil {
		return false
	}
	materialize := d.materializeBitmap
	d.materializeBitmap = nil
	if err := materialize(); err != nil {
		d.materializeErr = err
		return false
	}
	return len(d.BM.Pix) >= want
}

// quitting reports whether an installed Quit hook has cancelled this search.
func (d *PrimaryDetector) quitting() bool {
	return d.Quit != nil && d.Quit()
}

// LocateFinders locates the current ISO/current-C finder signature. Optional
// signatures are not enabled by this compatibility wrapper.
func (d *PrimaryDetector) LocateFinders() bool {
	return d.LocateFinderFamilies(FinderFamilyCurrent.Mask()).Has(FinderFamilyCurrent)
}

// LocateInitialFinderFamilies runs only the first balanced-image finder pass.
// It is the compact-mask boundary used by the automatic GPU path: a successful
// pass can materialize pixels for geometry and sampling, while a failed pass
// falls back to the complete CPU retry ladder without changing its behavior.
func (d *PrimaryDetector) LocateInitialFinderFamilies(wanted FinderFamilySet) FinderFamilySet {
	found, _, _, _ := d.locateInitialFinderFamilies(wanted)
	return found
}

// LocateFinderFamilies runs one finder traversal per prepared image pass and
// classifies every requested physical signature inside that traversal. The
// retry re-binarizes d.Ch in place; because the channel array is held by value,
// that swap is scoped to this detector and does not propagate to secondary
// detection. The C reference differs here: its detectMaster overwrites the
// caller's channel array, so it detects docked secondaries on the retry's
// re-binarization while this port detects them on the first-pass channels. The
// two can diverge only for a multi-symbol code whose primary needed the retry;
// the wire format is unaffected.
func (d *PrimaryDetector) LocateFinderFamilies(wanted FinderFamilySet) FinderFamilySet {
	found, _ := d.locateFinderFamilies(wanted, cpuFinderPassPreparer{bm: d.BM})
	return found
}

func (d *PrimaryDetector) locateFinderFamilies(
	wanted FinderFamilySet,
	preparer finderPassPreparer,
) (FinderFamilySet, error) {
	// Ports the retry orchestration of detectMaster in detector.c.
	found, wantCurrent, wantBSI, stop := d.locateInitialFinderFamilies(wanted)
	if stop {
		return found, nil
	}
	maxSurvivors := d.familySurvivors(wantCurrent, wantBSI)

	scanChannels := finderScanChannelMask(wantCurrent, wantBSI)

	// Retry 1: re-binarize using adaptive thresholds from around the found patterns.
	rgbAvg, err := preparer.averagePixelValue(d.retrySeedFinders(wantCurrent, wantBSI))
	if err != nil {
		return 0, err
	}
	d.Stats.RGBAvg = rgbAvg
	input, ch2, hits, err := preparer.prepare(0, 0, rgbAvg[:], false, scanChannels)
	if err != nil {
		return 0, err
	}
	d.Ch[0], d.Ch[1], d.Ch[2] = ch2[0], ch2[1], ch2[2]
	d.rowHits = hits
	found = d.findPrimaryFamilies(wantCurrent, wantBSI)
	d.pass().Label = "avg-RGB retry"
	d.recordTracePass(input)
	if found != 0 {
		d.selectLocatedFinderFamily(found)
		return found, nil
	}
	mergeFamilySurvivors(&maxSurvivors, d.familySurvivors(wantCurrent, wantBSI))

	// Retry 2 (descreen): screen captures inject the display's subpixel/diode lattice
	// and moiré, which can leave the raw and avg-RGB passes without enough surviving
	// finders. Estimate the lattice pitch per image and low-pass ≈ one grid cell (then
	// a coarser pass) before binarizing - the kernel is derived, not a fixed radius.
	// bm is left untouched so colour sampling still reads the original pixels; the
	// d.Ch swap stays primary-scoped.
	px, py, err := preparer.estimatePitch()
	if err != nil {
		return 0, err
	}
	for _, r := range descreenSchedule(px, py) {
		if d.quitting() {
			return 0, nil
		}
		filtered, chN, hitsN, err := preparer.prepare(r[0], r[1], nil, false, scanChannels)
		if err != nil {
			return 0, err
		}
		d.Ch[0], d.Ch[1], d.Ch[2] = chN[0], chN[1], chN[2]
		d.rowHits = hitsN
		found = d.findPrimaryFamilies(wantCurrent, wantBSI)
		d.pass().Label = fmt.Sprintf("descreen %dx%d", r[0], r[1])
		d.recordTracePass(filtered)
		if found != 0 {
			d.selectLocatedFinderFamily(found)
			return found, nil
		}
		mergeFamilySurvivors(&maxSurvivors, d.familySurvivors(wantCurrent, wantBSI))
	}

	// Retry 3 (print levels): subtractive print colours are dark - a printed
	// blue's own channel can sit below the block mean, so the default black
	// gate swallows whole colour modules as black. When the failed passes
	// show the print signature - raw run-length seeds by the hundred with
	// cross-check survivors near zero - re-binarize with the black gate on
	// the block-floor anchor, then once more on a copy low-passed at a
	// quarter of the seeds' own module-size estimate, which fuses halftone
	// cells, dither grain and colorant-plane fringes.
	currentPrint := wantCurrent && len(d.seedModules) >= printRetryMinSeeds &&
		maxSurvivors[FinderFamilyCurrent] <= printRetryMaxSurvivors
	bsiPrint := wantBSI && len(d.bsiFamilySeedModules) >= printRetryMinSeeds &&
		maxSurvivors[FinderFamilyBSI] <= printRetryMaxSurvivors
	if currentPrint || bsiPrint {
		// Two binarizations, and the first success wins, so order matters:
		// on coarse grain the sharp pass can succeed with a wrong finder
		// quad and poison the downstream side estimate - the low-passed one
		// lands the true geometry and goes first. On small modules the
		// integer blur radius collapses to a large module fraction and
		// shifts the finder centres instead, so there the sharp pass leads.
		// The radius itself separates the regimes: quantization dominates
		// it below printBlurLeadRadius.
		printSeeds := d.seedModules
		printCurrent := wantCurrent
		if !currentPrint {
			printSeeds = d.bsiFamilySeedModules
			printCurrent = false
		}
		r := max(1, int(seedModuleScale(printSeeds)/4+0.5))
		passes := [2]struct {
			label  string
			rx, ry int
		}{
			{label: fmt.Sprintf("print blurred r=%d", r), rx: r, ry: r},
			{label: "print sharp"},
		}
		if r < printBlurLeadRadius {
			passes[0], passes[1] = passes[1], passes[0]
		}
		d.printPass = true
		defer func() { d.printPass = false }()
		for _, p := range passes {
			if d.quitting() {
				return 0, nil
			}
			input, chP, hitsP, err := preparer.prepare(
				p.rx, p.ry, nil, true, finderScanChannelMask(printCurrent, wantBSI),
			)
			if err != nil {
				return 0, err
			}
			d.Ch[0], d.Ch[1], d.Ch[2] = chP[0], chP[1], chP[2]
			d.rowHits = hitsP
			found = d.findPrimaryFamilies(printCurrent, wantBSI)
			d.pass().Label = p.label
			d.recordTracePass(input)
			if found != 0 {
				d.selectLocatedFinderFamily(found)
				return found, nil
			}
		}
	}
	return 0, nil
}

func (d *PrimaryDetector) locateInitialFinderFamilies(
	wanted FinderFamilySet,
) (found FinderFamilySet, wantCurrent, wantBSI, stop bool) {
	wantCurrent = wanted.Has(FinderFamilyCurrent)
	wantBSI = wanted.Has(FinderFamilyBSI) && bsiFamilyFinderEnabled
	d.seedModules = d.seedModules[:0]
	d.bsiFamilySeedModules = d.bsiFamilySeedModules[:0]
	d.printDetected = false
	clear(d.familyResults[:])
	if d.Trace != nil {
		d.Trace.PassInputs = d.Trace.PassInputs[:0]
		d.Trace.PassChannels = d.Trace.PassChannels[:0]
		d.Trace.FinderPasses = d.Trace.FinderPasses[:0]
	}
	if d.quitting() {
		return 0, wantCurrent, wantBSI, true
	}
	found = d.findPrimaryFamilies(wantCurrent, wantBSI)
	d.pass().Label = "raw"
	d.recordTracePass(d.BM)
	if wantCurrent && d.familyResults[FinderFamilyCurrent].status == core.FatalError {
		return 0, wantCurrent, wantBSI, true
	}
	if found != 0 {
		d.selectLocatedFinderFamily(found)
		return found, wantCurrent, wantBSI, true
	}
	if d.quitting() {
		return 0, wantCurrent, wantBSI, true
	}
	return 0, wantCurrent, wantBSI, false
}

func (d *PrimaryDetector) selectLocatedFinderFamily(found FinderFamilySet) {
	if found.Has(FinderFamilyCurrent) {
		d.SelectFinderFamily(FinderFamilyCurrent)
		return
	}
	d.SelectFinderFamily(FinderFamilyBSI)
}

func (d *PrimaryDetector) familySurvivors(wantCurrent, wantBSI bool) [finderFamilyCount]int {
	var n [finderFamilyCount]int
	if wantCurrent {
		n[FinderFamilyCurrent] = len(d.familyResults[FinderFamilyCurrent].candidates)
	}
	if wantBSI {
		n[FinderFamilyBSI] = len(d.familyResults[FinderFamilyBSI].candidates)
	}
	return n
}

func mergeFamilySurvivors(dst *[finderFamilyCount]int, src [finderFamilyCount]int) {
	for family := range finderFamilyCount {
		dst[family] = max(dst[family], src[family])
	}
}

func (d *PrimaryDetector) retrySeedFinders(wantCurrent, wantBSI bool) []FinderPattern {
	current := &d.familyResults[FinderFamilyCurrent]
	if wantCurrent {
		return current.fps
	}
	if wantBSI {
		return d.familyResults[FinderFamilyBSI].fps
	}
	return nil
}
