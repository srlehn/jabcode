package detect

import (
	"fmt"

	"github.com/srlehn/jabcode/internal/core"
)

// FinderPassStats records per-pass finder-detection counters for diagnostics.
// They are observation only and never influence detection.
type FinderPassStats struct {
	Label          string          // which LocateFinders pass produced this entry (raw, avg-RGB, descreen, print)
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

// DetectorStats aggregates finder-detection instrumentation across the up-to-two
// binarization passes LocateFinders runs.
type DetectorStats struct {
	Passes []FinderPassStats // one entry per findPrimarySymbol pass
	RGBAvg [3]float32        // retry thresholds from averagePixelValue, between passes
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

	// Quit, when set, is polled between binarization passes; once it reports
	// true the search abandons its remaining retries and fails. The resolution
	// pyramid cancels levels that can no longer win this way, so an abandoned
	// level stops burning cores within one pass instead of finishing its whole
	// retry ladder in the background.
	Quit func() bool

	// seedModules collects the per-seed module-size estimate of every raw
	// n-1-1-1-m hit across this detection's passes: the evidence the print
	// retry gates on and derives its low-pass radius from. Working state,
	// kept off Stats so those stay observation-only.
	seedModules []float64

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

// quitting reports whether an installed Quit hook has cancelled this search.
func (d *PrimaryDetector) quitting() bool {
	return d.Quit != nil && d.Quit()
}

// LocateFinders runs the finder search, falling back to a finder-seeded second
// binarization pass on failure. The retry re-binarizes d.Ch in place; because the
// channel array is held by value, that swap is scoped to this detector and does
// not propagate to secondary detection. The C reference differs here: its
// detectMaster overwrites the caller's channel array, so it detects docked
// secondaries on the retry's re-binarization while this port detects them on the
// first-pass channels. The two can diverge only for a multi-symbol code whose
// primary needed the retry; the wire format is unaffected.
func (d *PrimaryDetector) LocateFinders() bool {
	// Ports the retry orchestration of detectMaster in detector.c.
	d.seedModules = d.seedModules[:0]
	d.printDetected = false
	if d.quitting() {
		return false
	}
	status := d.findPrimarySymbol()
	d.pass().Label = "raw"
	if status == core.FatalError {
		return false
	}
	if status == core.Success {
		return true
	}
	maxSurvivors := len(d.Candidates)
	if d.quitting() {
		return false
	}

	// Retry 1: re-binarize using adaptive thresholds from around the found patterns.
	rgbAvg := averagePixelValue(d.BM, d.FPs)
	d.Stats.RGBAvg = rgbAvg
	ch2 := BinarizerRGB(d.BM, rgbAvg[:])
	d.Ch[0], d.Ch[1], d.Ch[2] = ch2[0], ch2[1], ch2[2]
	status = d.findPrimarySymbol()
	d.pass().Label = "avg-RGB retry"
	if status == core.Success {
		return true
	}
	maxSurvivors = max(maxSurvivors, len(d.Candidates))

	// Retry 2 (descreen): screen captures inject the display's subpixel/diode lattice
	// and moiré, which can leave the raw and avg-RGB passes without enough surviving
	// finders. Estimate the lattice pitch per image and low-pass ≈ one grid cell (then
	// a coarser pass) before binarizing - the kernel is derived, not a fixed radius.
	// bm is left untouched so colour sampling still reads the original pixels; the
	// d.Ch swap stays primary-scoped.
	px, py := EstimatePitch(d.BM)
	for _, r := range descreenSchedule(px, py) {
		if d.quitting() {
			return false
		}
		chN := BinarizerRGB(descreen(d.BM, r[0], r[1]), nil)
		d.Ch[0], d.Ch[1], d.Ch[2] = chN[0], chN[1], chN[2]
		status = d.findPrimarySymbol()
		d.pass().Label = fmt.Sprintf("descreen %dx%d", r[0], r[1])
		if status == core.Success {
			return true
		}
		maxSurvivors = max(maxSurvivors, len(d.Candidates))
	}

	// Retry 3 (print levels): subtractive print colours are dark - a printed
	// blue's own channel can sit below the block mean, so the default black
	// gate swallows whole colour modules as black. When the failed passes
	// show the print signature - raw run-length seeds by the hundred with
	// cross-check survivors near zero - re-binarize with the black gate on
	// the block-floor anchor, then once more on a copy low-passed at a
	// quarter of the seeds' own module-size estimate, which fuses halftone
	// cells, dither grain and colorant-plane fringes.
	if len(d.seedModules) >= printRetryMinSeeds && maxSurvivors <= printRetryMaxSurvivors {
		// Two binarizations, and the first success wins, so order matters:
		// on coarse grain the sharp pass can succeed with a wrong finder
		// quad and poison the downstream side estimate - the low-passed one
		// lands the true geometry and goes first. On small modules the
		// integer blur radius collapses to a large module fraction and
		// shifts the finder centres instead, so there the sharp pass leads.
		// The radius itself separates the regimes: quantization dominates
		// it below printBlurLeadRadius.
		r := max(1, int(seedModuleScale(d.seedModules)/4+0.5))
		passes := [2]struct {
			label    string
			binarize func() [3]*core.Bitmap
		}{
			{fmt.Sprintf("print blurred r=%d", r), func() [3]*core.Bitmap { return BinarizerRGBPrint(descreen(d.BM, r, r)) }},
			{"print sharp", func() [3]*core.Bitmap { return BinarizerRGBPrint(d.BM) }},
		}
		if r < printBlurLeadRadius {
			passes[0], passes[1] = passes[1], passes[0]
		}
		d.printPass = true
		defer func() { d.printPass = false }()
		for _, p := range passes {
			if d.quitting() {
				return false
			}
			chP := p.binarize()
			d.Ch[0], d.Ch[1], d.Ch[2] = chP[0], chP[1], chP[2]
			status = d.findPrimarySymbol()
			d.pass().Label = p.label
			if status == core.Success {
				d.printDetected = true
				return true
			}
		}
	}
	return false
}
