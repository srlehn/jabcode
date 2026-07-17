package detect

import (
	"math"
	"math/rand"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/spec"
)

// chainTestRowHits replays the row-scan machine over one binarized channel
// and returns the raw integer hit records the device scan emits, asserting
// each derived center and module size against seekPatternHorizontal.
func chainTestRowHits(t *testing.T, bin *core.Bitmap) []finderRowHit {
	t.Helper()
	w, h := bin.Width, bin.Height
	var hits []finderRowHit
	for y := 0; y < h; y++ {
		row := bin.Pix[y*w : (y+1)*w]
		startX, endX, skip := 0, w, 0
		seq := 0
		for first := true; first || (startX < w && endX < w); {
			first = false
			startX += skip
			endX = w
			seekStart := startX
			var sc [5]int
			curState := 0
			resStart := startX
			ok := false
			hitEnd := 0
			if startX < w {
				sc[0] = 1
				for j := startX + 1; j < w; j++ {
					if row[j] == row[j-1] {
						sc[curState]++
					}
					if row[j] != row[j-1] || j == w-1 {
						if curState < 4 {
							if sc[curState] < 3 {
								if curState == 0 {
									sc[0] = 1
									resStart = j
								} else {
									sc[curState-1] += sc[curState]
									sc[curState] = 0
									curState--
									sc[curState]++
								}
							} else {
								curState++
								sc[curState]++
							}
						} else {
							if sc[4] < 3 {
								sc[3] += sc[4]
								sc[4] = 0
								curState = 3
								sc[3]++
								continue
							}
							if _, ok2 := checkPatternCross(sc); ok2 {
								endX = j + 1
								hitEnd = j
								if j == w-1 && row[j] == row[j-1] {
									hitEnd = j + 1
								}
								ok = true
								break
							}
							resStart += sc[0]
							copy(sc[:4], sc[1:])
							sc[4] = 1
							curState = 4
						}
					}
				}
			}
			if !ok {
				break
			}
			hit := finderRowHit{
				y: y, seq: seq, endPos: hitEnd,
				s2: sc[2], s3: sc[3], s4: sc[4],
				inside: sc[1] + sc[2] + sc[3],
			}
			ps := seekPatternHorizontal(row, seekStart, w)
			if !ps.ok ||
				math.Float64bits(hit.center()) != math.Float64bits(ps.Center) ||
				math.Float64bits(hit.moduleSize()) != math.Float64bits(ps.ModuleSize) {
				t.Fatalf("row %d hit %d: walker record (center %v module %v) disagrees with seekPatternHorizontal (%v %v %v)",
					y, seq, hit.center(), hit.moduleSize(), ps.ok, ps.Center, ps.ModuleSize)
			}
			hits = append(hits, hit)
			seq++
			startX = resStart
			skip = sc[0]
		}
	}
	return hits
}

// drawChainRings paints a finder-like pattern of concentric square rings
// (core, inverse ring, core-colored ring) into binary channel masks. Ring
// colors are given as mask bits per channel.
func drawChainRings(ch [3]*core.Bitmap, cx, cy, module int, ringBits [3][3]byte) {
	w, h := ch[0].Width, ch[0].Height
	half := module / 2
	extent := half + 2*module
	for dy := -extent; dy <= extent; dy++ {
		for dx := -extent; dx <= extent; dx++ {
			x, y := cx+dx, cy+dy
			if x < 0 || x >= w || y < 0 || y >= h {
				continue
			}
			cheb := max(dx, -dx)
			if v := max(dy, -dy); v > cheb {
				cheb = v
			}
			ring := 0
			if cheb > half+module {
				ring = 2
			} else if cheb > half {
				ring = 1
			}
			for c := range 3 {
				ch[c].Pix[y*w+x] = ringBits[ring][c] * 255
			}
		}
	}
}

// chainTestMasks builds deterministic binary channel fixtures: blocky run
// noise that feeds the machines' merge and reject paths, plus ring patterns
// of both families and several module sizes that survive deep into the
// cross-check chain.
func chainTestMasks(width, height int, seed int64, withRings bool) [3]*core.Bitmap {
	var ch [3]*core.Bitmap
	for c := range ch {
		ch[c] = core.NewBitmap(width, height, 1)
	}
	rng := rand.New(rand.NewSource(seed))
	for c := range ch {
		for y := 0; y < height; y++ {
			value := byte(0)
			for x := 0; x < width; {
				run := 1 + rng.Intn(8)
				for i := 0; i < run && x < width; i++ {
					ch[c].Pix[y*width+x] = value
					x++
				}
				value ^= 255
			}
		}
	}
	if !withRings {
		return ch
	}
	cyan := [3][3]byte{{0, 1, 1}, {0, 0, 0}, {0, 1, 1}}
	yellow := [3][3]byte{{1, 1, 0}, {0, 0, 0}, {1, 1, 0}}
	black := [3][3]byte{{0, 0, 0}, {1, 1, 1}, {0, 0, 0}}
	magenta := [3][3]byte{{1, 0, 1}, {0, 1, 0}, {1, 0, 1}}
	blue := [3][3]byte{{0, 0, 1}, {1, 1, 0}, {0, 0, 1}}
	drawChainRings(ch, 40, 40, 4, cyan)
	drawChainRings(ch, 150, 60, 6, cyan)
	drawChainRings(ch, 262, 82, 9, yellow)
	drawChainRings(ch, 70, 180, 5, yellow)
	drawChainRings(ch, 180, 200, 6, magenta)
	drawChainRings(ch, 296, 214, 5, blue)
	drawChainRings(ch, 40, 250, 4, black)
	// A cyan pattern with red streaks on both diagonals: every direction of
	// the survivor chain passes except the diagonal core color check, whose
	// rejection depends on the real crossCheckColor diagonal offset. A kernel
	// whose diagonal length constant collapses to zero accepts this pattern
	// and fails parity. The streaks are three rows thick because the chain
	// refines the center by about a pixel, which shifts the checked diagonal
	// intercepts.
	drawChainRings(ch, 310, 55, 6, cyan)
	for j := 3; j <= 11; j++ {
		for dy := -1; dy <= 1; dy++ {
			ch[0].Pix[(55+j+dy)*width+310+j] = 255
			ch[0].Pix[(55-j+dy)*width+310+j] = 255
		}
	}
	return ch
}

// cpuChainCurrentHit is processCurrentFamilyHit's per-hit body over the real
// cross-check functions, capturing the outcome instead of detector state.
func cpuChainCurrentHit(ch [3]*core.Bitmap, d *PrimaryDetector, y int, centerG, moduleG float64) (uint32, FinderPattern) {
	var flags uint32
	w := ch[0].Width
	rowR := ch[0].Pix[y*w : (y+1)*w]
	rowG := ch[1].Pix[y*w : (y+1)*w]
	rowB := ch[2].Pix[y*w : (y+1)*w]

	typeG := core.BoolColor(rowG[int(centerG)] > 0)
	centerR, centerB := centerG, centerG
	var typeR, typeB int
	var moduleR, moduleB float64
	blueBranch, redBranch := false, false
	slack := d.ccSlack(moduleG)

	if crossCheckPatternHorizontal(ch[2], moduleG*2, &centerB, float64(y), &moduleB, slack) {
		flags |= chainFlagBranchBlue
		typeB = core.BoolColor(rowB[int(centerB)] > 0)
		moduleR = moduleG
		coreRed := int(palette.Default[spec.FP3CoreColor*3])
		if crossCheckColor(ch[0], coreRed, int(moduleR), 5, int(centerR), y, 0, slack) {
			typeR = 0
			blueBranch = true
		}
	} else if crossCheckPatternHorizontal(ch[0], moduleG*2, &centerR, float64(y), &moduleR, slack) {
		flags |= chainFlagBranchRed
		typeR = core.BoolColor(rowR[int(centerR)] > 0)
		moduleB = moduleG
		coreBlue := int(palette.Default[spec.FP2CoreColor*3+2])
		if crossCheckColor(ch[2], coreBlue, int(moduleB), 5, int(centerB), y, 0, slack) {
			typeB = 0
			redBranch = true
			flags |= chainFlagRedColor
		}
	}

	if !(blueBranch || redBranch) {
		return flags, FinderPattern{}
	}
	fp := FinderPattern{Center: core.PointF{Y: float64(y)}, FoundCount: 1}
	if blueBranch {
		if !checkModuleSize2(moduleG, moduleB) {
			return flags, FinderPattern{}
		}
		fp.Center.X = (centerG + centerB) / 2
		fp.ModuleSize = (moduleG + moduleB) / 2
		if !fp.classify([]int{fp0, fp3}, typeR, typeG, typeB) {
			return flags, FinderPattern{}
		}
	} else {
		if !checkModuleSize2(moduleR, moduleG) {
			return flags, FinderPattern{}
		}
		fp.Center.X = (centerR + centerG) / 2
		fp.ModuleSize = (moduleR + moduleG) / 2
		if !fp.classify([]int{fp1, fp2}, typeR, typeG, typeB) {
			return flags, FinderPattern{}
		}
		flags |= chainFlagRedClassified
	}
	if crossCheckPattern(ch, &fp, 0, d.ccSlack(fp.ModuleSize)) {
		flags |= chainFlagSurvivor
	}
	return flags, fp
}

// compareChainOutcome checks a mirror outcome against the CPU chain's flags
// and surviving pattern, bit for bit.
func compareChainOutcome(t *testing.T, label string, hit finderRowHit, mirror chainOutcome, flags uint32, fp FinderPattern) {
	t.Helper()
	if mirror.flags != flags {
		t.Fatalf("%s hit y=%d seq=%d: mirror flags %#x, CPU flags %#x", label, hit.y, hit.seq, mirror.flags, flags)
	}
	if flags&chainFlagSurvivor == 0 {
		return
	}
	if int(mirror.typ) != fp.Typ || int(mirror.dir) != fp.direction ||
		math.Float64bits(mirror.cx.float()) != math.Float64bits(fp.Center.X) ||
		math.Float64bits(mirror.cy.float()) != math.Float64bits(fp.Center.Y) ||
		math.Float64bits(mirror.ms.float()) != math.Float64bits(fp.ModuleSize) {
		t.Fatalf("%s hit y=%d seq=%d: mirror survivor (typ %d dir %d cx %v cy %v ms %v), CPU (typ %d dir %d cx %v cy %v ms %v)",
			label, hit.y, hit.seq,
			mirror.typ, mirror.dir, mirror.cx.float(), mirror.cy.float(), mirror.ms.float(),
			fp.Typ, fp.direction, fp.Center.X, fp.Center.Y, fp.ModuleSize)
	}
}

// TestGPUFinderChainCurrentEquivalence proves the mirrored device chain
// bit-identical to the CPU per-hit chain for the current family over ring
// and noise fixtures in both slack modes, without a device.
func TestGPUFinderChainCurrentEquivalence(t *testing.T) {
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
		hits := chainTestRowHits(t, fixture.masks[1])
		if fixture.name == "rings" && len(hits) == 0 {
			t.Fatal("ring fixture produced no raw hits")
		}
		for _, printPass := range []bool{false, true} {
			d := &PrimaryDetector{printPass: printPass}
			survivors := 0
			streaked := 0
			for _, hit := range hits {
				flags, fp := cpuChainCurrentHit(fixture.masks, d, hit.y, hit.center(), hit.moduleSize())
				mirror := sfChainCurrentHit(masks, hit, printPass)
				compareChainOutcome(t, fixture.name, hit, mirror, flags, fp)
				if flags&chainFlagSurvivor != 0 {
					survivors++
					if math.Abs(fp.Center.X-310) < 20 && math.Abs(fp.Center.Y-55) < 20 {
						streaked++
					}
				}
			}
			if fixture.name == "rings" && !printPass && survivors == 0 {
				t.Fatal("ring fixture produced no chain survivors")
			}
			// The diagonal-streaked pattern must die at the diagonal color
			// check; a survivor there means the diagonal offset collapsed.
			if streaked != 0 {
				t.Fatalf("%s print=%v: %d survivors inside the diagonal-streaked pattern", fixture.name, printPass, streaked)
			}
			if testing.Verbose() {
				t.Logf("%s print=%v: %d hits, %d survivors bit-identical", fixture.name, printPass, len(hits), survivors)
			}
		}
	}
}
