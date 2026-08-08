//go:build (jabcode_bsi || jabcode_legacy) && !js

package detect

import (
	"math"
	"testing"
)

// TestGPUFinderChainBSIParity pins the device chain's BSI-family outcomes
// against the CPU per-hit chain on the device's own binarized channels, with
// the same threshold allowance the current-family gate carries: native f32 on
// the device and float64 on the host can decide a borderline hit differently.
func TestGPUFinderChainBSIParity(t *testing.T) {
	chainParitySession(t, func(t *testing.T, pass chainParityPass) {
		ch, hits := pass.ch, pass.hits
		if !hits.chained(0) {
			t.Fatal("device pass ran without the BSI chain")
		}
		d := &PrimaryDetector{printPass: pass.printLevels}
		survivors, diverged := 0, 0
		// The compacted candidates stay on the device for a consumer that folds
		// them there, so this comparison fetches them the way a host arm does.
		channelHits := hits.hitsFor(0)
		for _, hit := range channelHits {
			flags, fp := cpuChainBSIHit(ch, d, hit.y, hit.center(), hit.moduleSize())
			outcome := hits.outcomes[hit.rec]
			if outcome.flags != flags {
				if (outcome.flags^flags)&chainFlagSurvivor != 0 {
					t.Fatalf("hit y=%d seq=%d: BSI survivor decision flipped, device %#x, CPU %#x",
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
		total := len(channelHits)
		if float64(diverged) > chainDecisionDriftRate*float64(total) {
			t.Fatalf("%d of %d BSI hits took a different branch on the device", diverged, total)
		}
		if pass.fixture == "rings" && survivors == 0 {
			t.Fatal("ring parity pass produced no BSI chain survivors")
		}
		if testing.Verbose() {
			t.Logf("%d red hits, %d agreeing survivors, %d threshold divergences",
				total, survivors, diverged)
		}
	})
}
