//go:build (jabcode_bsi || jabcode_legacy) && !js

package detect

import (
	"math"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
)

// TestGPUFinderChainBSIParity pins the device chain's BSI-family outcomes
// against the CPU per-hit chain on the device's own binarized channels.
func TestGPUFinderChainBSIParity(t *testing.T) {
	chainParitySession(t, func(t *testing.T, fixture string, ch [3]*core.Bitmap, hits *finderPassRowHits, printLevels bool) {
		if !hits.chained(0) {
			t.Fatal("device pass ran without the BSI chain")
		}
		d := &PrimaryDetector{printPass: printLevels}
		survivors := 0
		for _, hit := range hits.channels[0] {
			flags, fp := cpuChainBSIHit(ch, d, hit.y, hit.center(), hit.moduleSize())
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
		if fixture == "rings" && survivors == 0 {
			t.Fatal("ring parity pass produced no BSI chain survivors")
		}
		if testing.Verbose() {
			t.Logf("%d red hits, %d survivors bit-identical", len(hits.channels[0]), survivors)
		}
	})
}
