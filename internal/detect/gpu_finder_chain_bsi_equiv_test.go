//go:build jabcode_bsi || jabcode_legacy

package detect

import (
	"testing"

	"github.com/srlehn/jabcode/internal/core"
)

// cpuChainBSIHit is processBSIFamilyHit's per-hit body over the real
// cross-check functions, capturing the outcome instead of detector state.
func cpuChainBSIHit(ch [3]*core.Bitmap, d *PrimaryDetector, y int, center0, module0 float64) (uint32, FinderPattern) {
	var flags uint32
	w := ch[0].Width
	slack := d.ccSlack(module0)

	center := [3]float64{center0, center0, center0}
	moduleSize := [3]float64{module0}
	if !crossCheckPatternHorizontal(ch[1], module0*2,
		&center[1], float64(y), &moduleSize[1], slack) ||
		!crossCheckPatternHorizontal(ch[2], module0*2,
			&center[2], float64(y), &moduleSize[2], slack) ||
		!checkModuleSize3(moduleSize[0], moduleSize[1], moduleSize[2]) {
		return flags, FinderPattern{}
	}

	fp := FinderPattern{
		Center: core.PointF{
			X: (center[0] + center[1] + center[2]) / 3,
			Y: float64(y),
		},
		ModuleSize: (moduleSize[0] + moduleSize[1] + moduleSize[2]) / 3,
		FoundCount: 1,
	}
	if !fp.classifyBSIFamily(
		core.BoolColor(ch[0].Pix[y*w+int(center[0])] > 0),
		core.BoolColor(ch[1].Pix[y*w+int(center[1])] > 0),
		core.BoolColor(ch[2].Pix[y*w+int(center[2])] > 0),
	) || !crossCheckPatternBSIFamily(ch, &fp, 0, d.ccSlack(fp.ModuleSize)) {
		return flags, FinderPattern{}
	}
	flags |= chainFlagSurvivor
	return flags, fp
}

// TestGPUFinderChainBSIEquivalence proves the mirrored device chain
// bit-identical to the CPU per-hit chain for the BSI family over ring and
// noise fixtures in both slack modes, without a device.
func TestGPUFinderChainBSIEquivalence(t *testing.T) {
	fixtures := []struct {
		name  string
		masks [3]*core.Bitmap
	}{
		{name: "rings", masks: chainTestMasks(360, 300, 11, true)},
		{name: "noise", masks: chainTestMasks(331, 257, 12, false)},
		{name: "narrow", masks: chainTestMasks(31, 64, 13, false)},
	}
	for _, fixture := range fixtures {
		masks := packChainMasks(fixture.masks)
		hits := chainTestRowHits(t, fixture.masks[0])
		for _, printPass := range []bool{false, true} {
			d := &PrimaryDetector{printPass: printPass}
			survivors := 0
			for _, hit := range hits {
				flags, fp := cpuChainBSIHit(fixture.masks, d, hit.y, hit.center(), hit.moduleSize())
				mirror := sfChainBSIHit(masks, hit, printPass)
				compareChainOutcome(t, fixture.name, hit, mirror, flags, fp)
				if flags&chainFlagSurvivor != 0 {
					survivors++
				}
			}
			if fixture.name == "rings" && !printPass && survivors == 0 {
				t.Fatal("ring fixture produced no BSI chain survivors")
			}
			if testing.Verbose() {
				t.Logf("%s print=%v: %d hits, %d survivors bit-identical", fixture.name, printPass, len(hits), survivors)
			}
		}
	}
}
