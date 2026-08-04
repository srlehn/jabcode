//go:build !js

package detect

import (
	"math"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
)

// The device chain runs native f32 while the host chain runs float64, so the
// two disagree in the last digits and, for a hit sitting on a threshold, in
// which branch it takes. On the ring and noise fixtures that is about three
// percent of hits, and none of them changes whether the hit is accepted, so
// survivor flips are gated at zero wherever these are used while branch
// bookkeeping is allowed to drift. Real breakage - a wrong binding, a
// mis-sized record, a kernel that never ran - moves both far past these.
const (
	chainScalarTolerance   = 1e-3
	chainDecisionDriftRate = 0.05
	chainCounterDriftRate  = 0.10
)

// finderPatternsAgree reports whether two pattern lists describe the same
// findings. Type, direction and count must match exactly; positions and module
// sizes only within the arithmetic tolerance.
func finderPatternsAgree(got, want []FinderPattern) bool {
	if len(got) != len(want) {
		return false
	}
	for index, a := range got {
		b := want[index]
		if a.Typ != b.Typ || a.direction != b.direction || a.FoundCount != b.FoundCount ||
			math.Abs(a.Center.X-b.Center.X) > chainScalarTolerance ||
			math.Abs(a.Center.Y-b.Center.Y) > chainScalarTolerance ||
			math.Abs(a.ModuleSize-b.ModuleSize) > chainScalarTolerance {
			return false
		}
	}
	return true
}

// finderPatternsNearlyAgree is finderPatternsAgree for the lists a threshold
// decision can lengthen: contextual seeds are extra hypotheses for later
// stages, not findings, so a bounded number may appear on one side only.
func finderPatternsNearlyAgree(got, want []FinderPattern) bool {
	allowed := max(1, int(chainDecisionDriftRate*float64(len(want))))
	unmatched := 0
	matched := make([]bool, len(got))
	for _, b := range want {
		found := false
		for index, a := range got {
			if matched[index] || a.Typ != b.Typ || a.direction != b.direction {
				continue
			}
			if math.Abs(a.Center.X-b.Center.X) <= chainScalarTolerance &&
				math.Abs(a.Center.Y-b.Center.Y) <= chainScalarTolerance {
				matched[index], found = true, true
				break
			}
		}
		if !found {
			unmatched++
		}
	}
	for _, ok := range matched {
		if !ok {
			unmatched++
		}
	}
	return unmatched <= allowed
}

// chainCountersAgree compares one pass's counters. The survivor tally and the
// raw hit count are decisions and must match exactly; the branch counters
// record how many hits entered each stage and may drift with rounding.
func chainCountersAgree(got, want FinderPassStats) bool {
	if got.RawHits != want.RawHits || got.CrossSurvivors != want.CrossSurvivors {
		return false
	}
	for _, pair := range [][2]int{
		{got.BranchBlue, want.BranchBlue},
		{got.BranchRed, want.BranchRed},
		{got.RedColor, want.RedColor},
		{got.RedClassified, want.RedClassified},
	} {
		drift := math.Abs(float64(pair[0] - pair[1]))
		if drift > chainCounterDriftRate*math.Max(float64(pair[1]), 1) {
			return false
		}
	}
	return true
}

// chainParityBitmap composes an RGBA capture-like input from the binary ring
// and noise fixture channels, so the device pipeline binarizes, scans and
// chains the same structures the equivalence tests cover.
func chainParityBitmap(width, height int, seed int64, withRings bool) *core.Bitmap {
	ch := chainTestMasks(width, height, seed, withRings)
	bm := core.NewBitmap(width, height, 4)
	for pixel := 0; pixel < width*height; pixel++ {
		for c := range 3 {
			bm.Pix[pixel*4+c] = ch[c].Pix[pixel]
		}
		bm.Pix[pixel*4+3] = 255
	}
	return bm
}

// chainParitySession runs the resident device pipeline over RGBA fixtures and
// hands each pass's binarized channels and scan output to verify, in both
// slack modes. Shared by the untagged current-family parity test and the
// tagged BSI parity test.
func chainParitySession(
	t *testing.T,
	verify func(t *testing.T, fixture string, balanced *core.Bitmap, ch [3]*core.Bitmap, hits *finderPassRowHits, printLevels bool),
) {
	const maxWidth = 360
	const maxHeight = 300
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	input, err := device.NewBuffer(maxWidth * maxHeight * 4)
	if err != nil {
		_ = device.Close()
		t.Fatalf("allocate GPU chain parity input: %v", err)
	}
	resident, err := newGPUResidentBinarizerWithDevice(device, maxWidth, maxHeight)
	if err != nil {
		_ = input.Close()
		_ = device.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		if err := resident.Close(); err != nil {
			t.Errorf("close resident GPU binarizer: %v", err)
		}
		if err := input.Close(); err != nil {
			t.Errorf("close GPU chain parity input: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close GPU chain parity device: %v", err)
		}
	})

	tests := []struct {
		name   string
		bitmap *core.Bitmap
	}{
		{name: "rings", bitmap: chainParityBitmap(360, 300, 21, true)},
		{name: "noise", bitmap: chainParityBitmap(331, 257, 22, false)},
	}
	const scanChannels = (1 << 0) | (1 << 1)
	for _, test := range tests {
		for _, printLevels := range []bool{false, true} {
			name := test.name
			if printLevels {
				name += "-print"
			}
			t.Run(name, func(t *testing.T) {
				bm := test.bitmap
				if err := input.Upload(bm.Pix); err != nil {
					t.Fatalf("upload GPU chain parity input: %v", err)
				}
				channels, hits, materialize, err := resident.Binarize(
					input, bm.Width, bm.Height, nil, printLevels, scanChannels,
				)
				if err != nil {
					t.Fatalf("binarize with device chain: %v", err)
				}
				if err := materialize(); err != nil {
					t.Fatalf("materialize device chain masks: %v", err)
				}
				if hits == nil || !hits.valid {
					t.Fatal("device scan returned no valid hits")
				}
				// The chain decides the source-colour signal on the device, so
				// a host arm without these pixels would answer a different
				// question and the two would diverge by construction.
				balanced, err := resident.DownloadBalanced(bm.Width, bm.Height)
				if err != nil {
					t.Fatalf("download balanced parity image: %v", err)
				}
				verify(t, test.name, balanced, channels, hits, printLevels)
			})
		}
	}
}

// TestGPUFinderChainFallbackParity pins the scan-only degraded mode: a pass
// consumed before the background chain kernels are ready must reach the same
// finder families and patterns as the outcome replay, so which of the two ran
// never changes what the pass found.
func TestGPUFinderChainFallbackParity(t *testing.T) {
	chainParitySession(t, func(t *testing.T, fixture string, balanced *core.Bitmap, ch [3]*core.Bitmap, hits *finderPassRowHits, printLevels bool) {
		if !hits.chained(1) {
			t.Fatal("device pass ran without the current-family chain")
		}
		run := func(rowHits *finderPassRowHits) *PrimaryDetector {
			d := &PrimaryDetector{BM: balanced, Ch: ch, Mode: normalDetect, printPass: printLevels}
			d.rowHits = rowHits
			d.findPrimaryFamilies(true, false)
			return d
		}
		scanOnly := *hits
		scanOnly.outcomes = nil
		scanOnly.outcomeChannels = 0
		chained := run(hits)
		fallback := run(&scanOnly)
		chainedResult := chained.familyResults[FinderFamilyCurrent]
		fallbackResult := fallback.familyResults[FinderFamilyCurrent]
		if chainedResult.status != fallbackResult.status ||
			len(chainedResult.candidates) != len(fallbackResult.candidates) {
			t.Fatalf("fallback diverged: status %d vs %d, candidates %d vs %d",
				chainedResult.status, fallbackResult.status,
				len(chainedResult.candidates), len(fallbackResult.candidates))
		}
		if !finderPatternsAgree(chainedResult.candidates, fallbackResult.candidates) {
			t.Fatalf("fallback candidates diverged: %+v vs %+v",
				fallbackResult.candidates, chainedResult.candidates)
		}
		chainedStats := chained.Stats.Passes[0]
		fallbackStats := fallback.Stats.Passes[0]
		if !chainCountersAgree(chainedStats, fallbackStats) {
			t.Fatalf("fallback pass counters diverged: %+v vs %+v", fallbackStats, chainedStats)
		}
		if testing.Verbose() {
			t.Logf("%s: %d candidates agree across replay and fallback", fixture, len(chainedResult.candidates))
		}
	})
}

// TestGPUFinderChainParity pins the offload contract of the device chain for
// the current family: every raw green-row hit's outcome record carries the
// decision the CPU per-hit chain reaches on the same binarized channels. Hits
// whose measurements sit on a threshold may fall either way, so the number of
// those is bounded rather than required to be zero, and a survivor both sides
// accept must describe the same pattern.
func TestGPUFinderChainParity(t *testing.T) {
	chainParitySession(t, func(t *testing.T, fixture string, balanced *core.Bitmap, ch [3]*core.Bitmap, hits *finderPassRowHits, printLevels bool) {
		if !hits.chained(1) {
			t.Fatal("device pass ran without the current-family chain")
		}
		d := &PrimaryDetector{printPass: printLevels}
		survivors, diverged := 0, 0
		// The colour verdict has no counterpart in the mask-only CPU twin, so
		// it is compared against its own host oracle below rather than folded
		// into the flag equality.
		const colorBits = chainFlagColorEvaluated | chainFlagColorOK
		colorChecked := 0
		for _, hit := range hits.channels[1] {
			flags, fp := cpuChainCurrentHit(ch, d, hit.y, hit.center(), hit.moduleSize())
			outcome := hits.outcomes[hit.rec]
			if outcome.flags&chainFlagColorOK != 0 && outcome.flags&chainFlagColorEvaluated == 0 {
				t.Fatalf("hit y=%d seq=%d: colour passed without being evaluated", hit.y, hit.seq)
			}
			if outcome.flags&chainFlagSurvivor != 0 && outcome.flags&chainFlagColorEvaluated != 0 &&
				(fp.Typ == fp1 || fp.Typ == fp2) {
				want := finderPatternHasColorSignal(balanced, fp, newScanDirection(0))
				if got := outcome.flags&chainFlagColorOK != 0; got != want {
					t.Fatalf("hit y=%d seq=%d typ=%d: device colour verdict %t, host %t",
						hit.y, hit.seq, fp.Typ, got, want)
				}
				colorChecked++
			}
			if outcome.flags&^colorBits != flags {
				if (outcome.flags^flags)&chainFlagSurvivor != 0 {
					t.Fatalf("hit y=%d seq=%d: survivor decision flipped, device %#x, CPU %#x",
						hit.y, hit.seq, outcome.flags, flags)
				}
				diverged++
				continue
			}
			if flags&chainFlagSurvivor == 0 {
				continue
			}
			survivors++
			if outcome.typ != fp.Typ || outcome.direction != fp.direction ||
				math.Abs(outcome.centerX-fp.Center.X) > chainScalarTolerance ||
				math.Abs(outcome.centerY-fp.Center.Y) > chainScalarTolerance ||
				math.Abs(outcome.moduleSize-fp.ModuleSize) > chainScalarTolerance {
				t.Fatalf("hit y=%d seq=%d: device survivor (typ %d dir %d cx %v cy %v ms %v), CPU (typ %d dir %d cx %v cy %v ms %v)",
					hit.y, hit.seq,
					outcome.typ, outcome.direction, outcome.centerX, outcome.centerY, outcome.moduleSize,
					fp.Typ, fp.direction, fp.Center.X, fp.Center.Y, fp.ModuleSize)
			}
		}
		total := len(hits.channels[1])
		if float64(diverged) > chainDecisionDriftRate*float64(total) {
			t.Fatalf("%d of %d hits took a different branch on the device", diverged, total)
		}
		// The ring fixture must drive real survivors through the deep chain;
		// zero would mean the comparison lost its acceptance-path coverage.
		if fixture == "rings" && survivors == 0 {
			t.Fatal("ring parity pass produced no chain survivors")
		}
		// A gate that never reached the colour stage would pass while the
		// device silently stopped evaluating it.
		if fixture == "rings" && colorChecked == 0 {
			t.Fatal("ring parity pass evaluated no device colour verdicts")
		}
		if testing.Verbose() {
			t.Logf("%d green hits, %d agreeing survivors, %d threshold divergences",
				total, survivors, diverged)
		}
	})
}
