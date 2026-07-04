package detect

import (
	"github.com/srlehn/jabcode/internal/core"
)

// FinderPassStats records the per-pass finder-detection counters that the
// jabdiag-tagged diagnostic reads off the detector. They are observation only
// and never influence detection.
type FinderPassStats struct {
	RawHits        int             // n-1-1-1-m run-length hits (horizontal + conditional vertical scan)
	BranchBlue     int             // green seeds where the blue cross-check fired (-> {FP0,FP3} path)
	BranchRed      int             // green seeds where blue failed and the red cross-check fired (-> {FP1,FP2} path)
	RedColor       int             // red-path candidates passing the inner core-colour check (fp2found)
	RedClassified  int             // red-path candidates matched to fp1/fp2 by classifyFinderPattern
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
}

// pass returns the current (last-appended) finder pass's stats.
func (d *PrimaryDetector) pass() *FinderPassStats {
	return &d.Stats.Passes[len(d.Stats.Passes)-1]
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
	status := d.findPrimarySymbol()
	if status == core.FatalError {
		return false
	}
	if status == core.Success {
		return true
	}

	// Retry 1: re-binarize using adaptive thresholds from around the found patterns.
	rgbAvg := averagePixelValue(d.BM, d.FPs)
	d.Stats.RGBAvg = rgbAvg
	ch2 := BinarizerRGB(d.BM, rgbAvg[:])
	d.Ch[0], d.Ch[1], d.Ch[2] = ch2[0], ch2[1], ch2[2]
	if d.findPrimarySymbol() == core.Success {
		return true
	}

	// Retry 2 (descreen): screen captures inject the display's subpixel/diode lattice
	// and moiré, which can leave the raw and avg-RGB passes without enough surviving
	// finders. Estimate the lattice pitch per image and low-pass ≈ one grid cell (then
	// a coarser pass) before binarizing — the kernel is derived, not a fixed radius.
	// bm is left untouched so colour sampling still reads the original pixels; the
	// d.Ch swap stays primary-scoped.
	px, py := EstimatePitch(d.BM)
	for _, r := range descreenSchedule(px, py) {
		chN := BinarizerRGB(descreen(d.BM, r[0], r[1]), nil)
		d.Ch[0], d.Ch[1], d.Ch[2] = chN[0], chN[1], chN[2]
		if d.findPrimarySymbol() == core.Success {
			return true
		}
	}
	return false
}
