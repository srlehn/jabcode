package detect

import (
	"math"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
)

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
	verify func(t *testing.T, fixture string, ch [3]*core.Bitmap, hits *finderPassRowHits, printLevels bool),
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
				verify(t, test.name, channels, hits, printLevels)
			})
		}
	}
}

// TestGPUFinderChainFallbackParity pins the scan-only degraded mode: a pass
// consumed before the background chain kernels are ready must produce the
// same finder families, patterns and pass counters as the outcome replay,
// because the consumer's CPU per-hit chain is bit-identical to the device
// chain.
func TestGPUFinderChainFallbackParity(t *testing.T) {
	chainParitySession(t, func(t *testing.T, fixture string, ch [3]*core.Bitmap, hits *finderPassRowHits, printLevels bool) {
		if !hits.chained(1) {
			t.Fatal("device pass ran without the current-family chain")
		}
		run := func(rowHits *finderPassRowHits) *PrimaryDetector {
			d := &PrimaryDetector{Ch: ch, Mode: normalDetect, printPass: printLevels}
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
		for index, candidate := range chainedResult.candidates {
			if candidate != fallbackResult.candidates[index] {
				t.Fatalf("fallback candidate %d = %+v, chained %+v",
					index, fallbackResult.candidates[index], candidate)
			}
		}
		chainedStats := chained.Stats.Passes[0]
		fallbackStats := fallback.Stats.Passes[0]
		if chainedStats.RawHits != fallbackStats.RawHits ||
			chainedStats.BranchBlue != fallbackStats.BranchBlue ||
			chainedStats.BranchRed != fallbackStats.BranchRed ||
			chainedStats.RedColor != fallbackStats.RedColor ||
			chainedStats.RedClassified != fallbackStats.RedClassified ||
			chainedStats.CrossSurvivors != fallbackStats.CrossSurvivors {
			t.Fatalf("fallback pass counters diverged: %+v vs %+v", fallbackStats, chainedStats)
		}
		if testing.Verbose() {
			t.Logf("%s: %d candidates identical across replay and fallback", fixture, len(chainedResult.candidates))
		}
	})
}

// TestGPUFinderChainParity pins the offload contract of the device chain for
// the current family: every raw green-row hit's outcome record is
// bit-identical to the CPU per-hit chain run on the same binarized channels -
// same flags, and for survivors the same type, direction and float64 center
// and module size.
func TestGPUFinderChainParity(t *testing.T) {
	chainParitySession(t, func(t *testing.T, fixture string, ch [3]*core.Bitmap, hits *finderPassRowHits, printLevels bool) {
		if !hits.chained(1) {
			t.Fatal("device pass ran without the current-family chain")
		}
		d := &PrimaryDetector{printPass: printLevels}
		survivors := 0
		for _, hit := range hits.channels[1] {
			flags, fp := cpuChainCurrentHit(ch, d, hit.y, hit.center(), hit.moduleSize())
			outcome := hits.outcomes[hit.rec]
			if outcome.flags != flags {
				t.Fatalf("hit y=%d seq=%d: device flags %#x, CPU flags %#x",
					hit.y, hit.seq, outcome.flags, flags)
			}
			if flags&chainFlagSurvivor == 0 {
				continue
			}
			survivors++
			if outcome.typ != fp.Typ || outcome.direction != fp.direction ||
				math.Float64bits(outcome.centerX) != math.Float64bits(fp.Center.X) ||
				math.Float64bits(outcome.centerY) != math.Float64bits(fp.Center.Y) ||
				math.Float64bits(outcome.moduleSize) != math.Float64bits(fp.ModuleSize) {
				t.Fatalf("hit y=%d seq=%d: device survivor (typ %d dir %d cx %v cy %v ms %v), CPU (typ %d dir %d cx %v cy %v ms %v)",
					hit.y, hit.seq,
					outcome.typ, outcome.direction, outcome.centerX, outcome.centerY, outcome.moduleSize,
					fp.Typ, fp.direction, fp.Center.X, fp.Center.Y, fp.ModuleSize)
			}
		}
		// The ring fixture must drive real survivors through the deep chain;
		// zero would mean the comparison lost its acceptance-path coverage.
		if fixture == "rings" && survivors == 0 {
			t.Fatal("ring parity pass produced no chain survivors")
		}
		if testing.Verbose() {
			t.Logf("%d green hits, %d survivors bit-identical", len(hits.channels[1]), survivors)
		}
	})
}
