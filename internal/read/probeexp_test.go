//go:build jabharness

package read

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"math/bits"
	mrand "math/rand"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/testutil"
)

// TestProbeEscalationExperiment prints per-pyramid-level orientation-probe
// families for capture fixtures that still fail after the probe escalation -
// a temporary diagnostic to attribute the remaining failures to probe
// retention vs downstream stages. Not part of any gate.
func TestProbeEscalationExperiment(t *testing.T) {
	fixtures := []string{
		"display_camera/4c_frontal_rot45_normal.webp",
		"display_camera/8c_side_rot45_normal.webp",
		"display_camera/16c_side_rot45_normal.webp",
		"print_camera/8c_frontal_rot45.webp",
		"print_camera/8c_side_rot45.webp",
	}
	for _, rel := range fixtures {
		path := filepath.Join(testutil.CapturePath(t), filepath.FromSlash(rel))
		img, err := loadCaptureImage(path)
		if err != nil {
			t.Skipf("load %s: %v", rel, err)
		}
		levels := pyramidLevels(img)
		if levels == nil {
			t.Fatalf("%s: no pyramid levels", rel)
		}
		for k, lvl := range levels {
			bound := detect.CoarseMaxDim << k
			fams := detect.CoarseProbeFamiliesWithin(lvl, bound)
			line := fmt.Sprintf("%s level %d (%dx%d, bound %d):", rel, k, lvl.Rect.Dx(), lvl.Rect.Dy(), bound)
			for _, f := range fams {
				line += fmt.Sprintf("  %g:t%d,s%d", f.Deg, f.Types, f.Sum)
			}
			t.Logf("%s  -> rungs %v", line, detect.FamiliesToRungs(fams))
		}
	}
}

// TestRungStageExperiment stage-walks specific pre-rotation rungs at specific
// pyramid levels for still-failing captures, attributing where a correctly
// oriented full decode dies. Temporary diagnostic, not part of any gate.
func TestRungStageExperiment(t *testing.T) {
	cases := []struct {
		rel   string
		level int
		rungs []float64
	}{
		{"display_camera/16c_side_rot45_normal.webp", 2, []float64{45, 135, 225, 315}},
		{"display_camera/16c_side_rot45_normal.webp", 3, []float64{45, 135, 225, 315}},
		{"display_camera/8c_side_rot45_normal.webp", 2, []float64{45, 135, 225, 315}},
		{"display_camera/8c_side_rot45_normal.webp", 3, []float64{45, 135, 225, 315}},
		{"display_camera/4c_frontal_rot45_normal.webp", 2, []float64{45, 135, 225, 315}},
		{"print_camera/8c_frontal_rot45.webp", 1, []float64{30, 120, 210, 300}},
		{"print_camera/8c_frontal_rot45.webp", 3, []float64{45, 135, 225, 315}},
	}
	for _, c := range cases {
		path := filepath.Join(testutil.CapturePath(t), filepath.FromSlash(c.rel))
		img, err := loadCaptureImage(path)
		if err != nil {
			t.Skipf("load %s: %v", c.rel, err)
		}
		levels := pyramidLevels(img)
		if levels == nil || c.level >= len(levels) {
			t.Fatalf("%s: level %d not available", c.rel, c.level)
		}
		for _, deg := range c.rungs {
			bm := detect.RotateToBitmap(levels[c.level], deg)
			stage, sampled := samplePipelineBitmap(bm)
			dims := "-"
			if sampled != nil {
				dims = fmt.Sprintf("%dx%d", sampled.Width, sampled.Height)
			}
			t.Logf("%s level %d rot %g: stage %s, sampled grid %s", c.rel, c.level, deg, stage, dims)
		}
	}
}

// TestRouteTraceExperiment prints the full route trace for a few
// still-failing captures - a smoke check that the route-aware attribution
// records every attempted level/rung/region route with plausible stages.
// Temporary diagnostic, not part of any gate.
func TestRouteTraceExperiment(t *testing.T) {
	fixtures := []string{
		"display_camera/8c_side_rot45_normal.webp",
		"print_camera/8c_frontal_rot45.webp",
		"display_camera/64c_frontal_rot00_normal.webp",
	}
	for _, rel := range fixtures {
		path := filepath.Join(testutil.CapturePath(t), filepath.FromSlash(rel))
		img, err := loadCaptureImage(path)
		if err != nil {
			t.Skipf("load %s: %v", rel, err)
		}
		data, tr, err := decodeTraced(img)
		t.Logf("%s: err=%v payload=%d bytes, %d attempts", rel, err, len(data), len(tr.attempts))
		for _, a := range tr.attempts {
			t.Logf("  L%d rot%-5g roi%-2d stage=%s grid=%dx%d",
				a.level, a.deg, a.roi, captureStageString(a.stage), a.side.X, a.side.Y)
		}
		if best, ok := tr.best(); ok {
			t.Logf("  best: %s", captureRouteNote(best, tr, image.Point{}))
		}
	}
}

// TestFrameTraceExperiment is TestRouteTraceExperiment for an arbitrary image
// given via $JABTRACE_IMAGE: it prints every attempted route with its stage,
// plus the stage tally and total wall time - the attempt-count side of the
// per-frame latency attribution (each sampled-stage attempt paid at least one
// full data-LDPC decode). Temporary diagnostic, not part of any gate.
func TestFrameTraceExperiment(t *testing.T) {
	path := os.Getenv("JABTRACE_IMAGE")
	if path == "" {
		t.Skip("JABTRACE_IMAGE not set")
	}
	img, err := loadCaptureImage(path)
	if err != nil {
		t.Fatalf("load %s: %v", path, err)
	}
	start := time.Now()
	data, tr, err := decodeTraced(img)
	wall := time.Since(start)
	t.Logf("%s: err=%v payload=%d bytes, %d attempts, wall %s", path, err, len(data), len(tr.attempts), wall)
	counts := map[readStage]int{}
	for _, a := range tr.attempts {
		counts[a.stage]++
		t.Logf("  L%d rot%-5g roi%-2d stage=%s grid=%dx%d",
			a.level, a.deg, a.roi, captureStageString(a.stage), a.side.X, a.side.Y)
	}
	for _, s := range []readStage{readAborted, readNoFinders, readNoSideSize, readNoSample, readSampled, readDecoded} {
		if counts[s] > 0 {
			t.Logf("  stage %-12s x%d", captureStageString(s), counts[s])
		}
	}
}

// TestAdmissionSignalsExperiment prints the observation admission signals
// (fixed-pattern agreement, palette coherence) for every observation an
// image's full-resolution upright read, its coarsest pyramid level, and its
// first coarse orientation rungs produce, next to the payload-correction
// outcome - the true-vs-phantom separation measurement for the quality
// admission gate. $JABADMIT_DIR names a directory of captures;
// $JABADMIT_MAX caps the file count (default 8). Temporary diagnostic, not
// part of any gate.
func TestAdmissionSignalsExperiment(t *testing.T) {
	dir := os.Getenv("JABADMIT_DIR")
	if dir == "" {
		t.Skip("JABADMIT_DIR not set")
	}
	maxN := 8
	if s := os.Getenv("JABADMIT_MAX"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 {
			maxN = n
		}
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read %s: %v", dir, err)
	}
	var files []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		switch strings.ToLower(filepath.Ext(e.Name())) {
		case ".webp", ".png", ".jpg", ".jpeg":
			files = append(files, e.Name())
		}
	}
	slices.Sort(files)
	if len(files) > maxN {
		files = files[:maxN]
	}
	if len(files) == 0 {
		t.Fatalf("no images in %s", dir)
	}

	observeCanvas := func(label string, bm *core.Bitmap) {
		detect.BalanceRGB(bm)
		ch := detect.BinarizerRGB(bm, nil)
		d := &detect.PrimaryDetector{BM: bm, Ch: ch, Mode: detect.IntensiveDetect}
		var sym core.DecodedSymbol
		obs, stage := observePrimary(d, &sym, nil)
		if obs == nil {
			t.Logf("  %-16s stage=%s grid=%dx%d", label, captureStageString(stage), sym.SideSize.X, sym.SideSize.Y)
			return
		}
		agree, checked := obs.FixedPatternAgreement()
		dis, sep := obs.PaletteCoherence()
		meta := "meta=explicit"
		if sym.Meta.DefaultMode {
			meta = "meta=default"
		}
		res := obs.CorrectPayload()
		t.Logf("  %-16s grid=%dx%d Nc=%d %s syn=%v/%v fixed=%d/%d (%.2f) pal=%.1f/%.1f (r=%.2f) payload=%v",
			label, sym.SideSize.X, sym.SideSize.Y, sym.Meta.NC, meta,
			obs.PartISyndromeOK, obs.PartIISyndromeOK,
			agree, checked, float64(agree)/float64(max(checked, 1)),
			dis, sep, dis/math.Max(sep, 1e-9), res == core.Success)
	}

	for _, name := range files {
		img, err := loadCaptureImage(filepath.Join(dir, name))
		if err != nil {
			t.Logf("%s: load failed: %v", name, err)
			continue
		}
		t.Logf("%s:", name)
		observeCanvas("full upright", core.BitmapFromImage(img))
		levels := pyramidLevels(img)
		if levels != nil {
			observeCanvas("L0 upright", core.BitmapFromImage(levels[0]))
		}
		// Rungs as the pipeline finds them: the flat probe first, then the
		// escalated per-level probe (mirrors roiRungs).
		rungs := detect.CoarseOrientationRungs(img)
		if len(rungs) == 0 && levels != nil {
			for k, lvl := range levels[1:] {
				if fams := detect.CoarseProbeFamiliesWithin(lvl, detect.CoarseMaxDim<<(k+1)); len(fams) > 0 {
					rungs = detect.FamiliesToRungsUncapped(fams)
					break
				}
			}
		}
		for i, deg := range rungs {
			if i >= 3 {
				break
			}
			observeCanvas(fmt.Sprintf("full rot%g", deg), detect.RotateToBitmap(img, deg))
			if levels != nil {
				observeCanvas(fmt.Sprintf("L0 rot%g", deg), detect.RotateToBitmap(levels[0], deg))
				observeCanvas(fmt.Sprintf("L1 rot%g", deg), detect.RotateToBitmap(levels[1], deg))
			}
		}
	}
}

// TestDegradedAdmissionExperiment prints the admission signals of the
// degradation-harness rows nearest the gate boundary (blur, lattice, jpeg on
// the default-mode synthetic codes). Temporary diagnostic, not part of any
// gate.
func TestDegradedAdmissionExperiment(t *testing.T) {
	cases := []struct {
		name  string
		apply func(image.Image, float64, *mrand.Rand) image.Image
		lvls  []float64
	}{
		{"blur-r", boxBlurDeg, []float64{1, 2, 4}},
		{"lattice-p", screenLattice, []float64{3, 5}},
		{"jpeg-q", jpegRecompress, []float64{10}},
		{"noise-sd", gaussianNoise, []float64{50}},
	}
	for _, payload := range [][]byte{
		[]byte("HARNESS round-trip 0123456789"),
		[]byte("The quick brown fox jumps over the lazy dog."),
	} {
		gt := encodeGroundTruth(t, payload)
		for _, c := range cases {
			for _, lvl := range c.lvls {
				rng := mrand.New(mrand.NewSource(1))
				img := c.apply(gt.img, lvl, rng)
				bm := core.BitmapFromImage(img)
				detect.BalanceRGB(bm)
				ch := detect.BinarizerRGB(bm, nil)
				d := &detect.PrimaryDetector{BM: bm, Ch: ch, Mode: detect.IntensiveDetect}
				var sym core.DecodedSymbol
				obs, stage := observePrimary(d, &sym, nil)
				if obs == nil {
					t.Logf("%-10s %2g: stage=%s", c.name, lvl, captureStageString(stage))
					continue
				}
				agree, checked := obs.FixedPatternAgreement()
				dis, sep := obs.PaletteCoherence()
				t.Logf("%-10s %2g: grid=%dx%d default=%v syn=%v/%v fixed=%d/%d (%.2f) pal=%.1f/%.1f (r=%.2f) admit=%v payload=%v",
					c.name, lvl, sym.SideSize.X, sym.SideSize.Y, sym.Meta.DefaultMode,
					obs.PartISyndromeOK, obs.PartIISyndromeOK,
					agree, checked, float64(agree)/float64(max(checked, 1)),
					dis, sep, dis/math.Max(sep, 1e-9),
					obs.AdmitPayloadCorrection(), obs.CorrectPayload() == core.Success)
			}
		}
	}
}

// TestLevelTimingExperiment times each pyramid level's upright read of
// $JABTRACE_IMAGE separately (sequential, no cancellation), then the seeded
// route replayed from the coarsest level's published finding - the split of a
// frame's wall time into per-route attempt costs. Temporary diagnostic, not
// part of any gate.
func TestLevelTimingExperiment(t *testing.T) {
	path := os.Getenv("JABTRACE_IMAGE")
	if path == "" {
		t.Skip("JABTRACE_IMAGE not set")
	}
	img, err := loadCaptureImage(path)
	if err != nil {
		t.Fatalf("load %s: %v", path, err)
	}
	levels := pyramidLevels(img)
	if levels == nil {
		t.Skip("single-level image")
	}
	var seedF finding
	for i, lvl := range levels {
		f := &finding{}
		start := time.Now()
		data, stage, evidence := decodeBitmapFinding(core.BitmapFromImage(lvl), nil, f)
		t.Logf("L%d %4dx%4d upright: stage=%s evidence=%v grid=%dx%d payload=%d bytes in %s",
			i, lvl.Rect.Dx(), lvl.Rect.Dy(), captureStageString(stage), evidence,
			f.side.X, f.side.Y, len(data), time.Since(start))
		if i == 0 {
			seedF = *f
		}
	}
	if seedF.located {
		start := time.Now()
		data, side, ok := decodeSeeded(levels, seedF, func() bool { return false })
		t.Logf("seeded from L0 grid %dx%d: ok=%v side=%d payload=%d bytes in %s",
			seedF.side.X, seedF.side.Y, ok, side, len(data), time.Since(start))
	}
}

// TestROITileMapExperiment prints the ROI tile score map for glued multi-code
// captures at several working resolutions and grid densities, as an ASCII
// heat map with the proposer's threshold bands, to show whether the
// inter-code gutters ever form a sub-annex-threshold valley. Temporary
// diagnostic, not part of any gate.
func TestROITileMapExperiment(t *testing.T) {
	fixtures := []string{
		"print_camera/8c_frontal_rot45.webp",
		"print_camera/8c_side_rot45.webp",
		"print_camera/16c_frontal_rot45.webp",
	}
	configs := []struct{ maxDim, grid int }{
		{512, 32},
		{512, 64},
		{1024, 64},
		{1024, 96},
	}
	for _, rel := range fixtures {
		path := filepath.Join(testutil.CapturePath(t), filepath.FromSlash(rel))
		img, err := loadCaptureImage(path)
		if err != nil {
			t.Skipf("load %s: %v", rel, err)
		}
		for _, c := range configs {
			for _, roi := range detect.ProposeROIsWithin(img, 8, c.maxDim, c.grid) {
				t.Logf("%s dim%d grid%d: box %v score %.1f tiles %d",
					rel, c.maxDim, c.grid, roi.Bounds, roi.Score, roi.Tiles)
			}
			m := detect.BuildROITileMapWithin(img, c.maxDim, c.grid)
			peak := m.Peak()
			if peak == 0 {
				t.Logf("%s dim%d grid%d: empty map", rel, c.maxDim, c.grid)
				continue
			}
			var b strings.Builder
			fmt.Fprintf(&b, "%s dim%d grid%d (%dx%d tiles, tile %dpx): bands  ' '=0  .<5%%  -<20%%  +<50%%  #>=50%% of peak\n",
				rel, c.maxDim, c.grid, m.GX, m.GY, m.Tile)
			for ty := range m.GY {
				for tx := range m.GX {
					s := m.Score[ty*m.GX+tx] / peak
					switch {
					case s == 0:
						b.WriteByte(' ')
					case s < 0.05:
						b.WriteByte('.')
					case s < 0.20:
						b.WriteByte('-')
					case s < 0.50:
						b.WriteByte('+')
					default:
						b.WriteByte('#')
					}
				}
				b.WriteByte('\n')
			}
			t.Log(b.String())
		}
	}
}

// TestSeedFloodROIExperiment prototypes seed-relative flood proposals: grow a
// box from each strong tile-map local peak through 8-connected tiles scoring
// at least a fraction of the SEED's score (not the global annex floor), so an
// in-focus code separates from its dimmer sheet neighbours. Prints the boxes
// and end-to-end decodes the top crops against ground truth. Temporary
// diagnostic, not part of any gate.
func TestSeedFloodROIExperiment(t *testing.T) {
	dir := testutil.CapturePath(t)
	known := captureGroundTruth(t, dir)
	fixtures := []string{
		"print_camera/8c_frontal_rot45.webp",
		"print_camera/8c_side_rot45.webp",
		"print_camera/16c_frontal_rot45.webp",
		"print_camera/32c_frontal_rot45.webp",
	}
	const falloff = 0.25
	for _, rel := range fixtures {
		img, err := loadCaptureImage(filepath.Join(dir, filepath.FromSlash(rel)))
		if err != nil {
			t.Skipf("load %s: %v", rel, err)
		}
		colors, _ := captureColorCount(rel)
		m := detect.BuildROITileMapWithin(img, 512, 32)
		if m.Peak() == 0 {
			t.Logf("%s: empty tile map", rel)
			continue
		}
		boxes := seedFloodBoxes(m, img.Bounds(), falloff, 4)
		for i, bx := range boxes {
			if i >= 3 {
				t.Logf("%s flood box %d: %v (not decoded)", rel, i, bx)
				continue
			}
			crop := detect.CropImage(img, bx)
			data, err := Decode(crop)
			verdict := "FAIL"
			if err == nil {
				switch match := matchKnownPayload(data, known); {
				case match == colors:
					verdict = "OK"
				case match != 0:
					verdict = fmt.Sprintf("other-code %dc", match)
				default:
					verdict = "corrupt"
				}
			}
			t.Logf("%s flood box %d: %v (%dx%d) -> %s", rel, i, bx, bx.Dx(), bx.Dy(), verdict)
		}
	}
}

// seedFloodBoxes builds boxes by processing tiles in descending score order:
// each unclaimed tile at or above the proposer's seed threshold floods
// 8-connected tiles scoring at least falloff*seed, claims them, and emits the
// padded bounding box mapped to full-resolution coordinates.
func seedFloodBoxes(m detect.ROITileMap, fb image.Rectangle, falloff float64, maxN int) []image.Rectangle {
	peak := m.Peak()
	type st struct {
		i int
		s float64
	}
	order := make([]st, 0, len(m.Score))
	for i, s := range m.Score {
		if s > 0 {
			order = append(order, st{i, s})
		}
	}
	slices.SortStableFunc(order, func(a, b st) int {
		if a.s != b.s {
			if a.s > b.s {
				return -1
			}
			return 1
		}
		return a.i - b.i
	})
	claimed := make([]bool, len(m.Score))
	sx := float64(fb.Dx()) / float64(m.W)
	sy := float64(fb.Dy()) / float64(m.H)
	var boxes []image.Rectangle
	for _, seed := range order {
		if len(boxes) >= maxN {
			break
		}
		if claimed[seed.i] || seed.s < 0.20*peak {
			continue
		}
		thr := falloff * seed.s
		stack := []int{seed.i}
		claimed[seed.i] = true
		minTx, minTy, maxTx, maxTy := m.GX, m.GY, -1, -1
		for len(stack) > 0 {
			i := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			tx, ty := i%m.GX, i/m.GX
			minTx, minTy = min(minTx, tx), min(minTy, ty)
			maxTx, maxTy = max(maxTx, tx), max(maxTy, ty)
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					nx, ny := tx+dx, ty+dy
					if (dx == 0 && dy == 0) || nx < 0 || ny < 0 || nx >= m.GX || ny >= m.GY {
						continue
					}
					j := ny*m.GX + nx
					if !claimed[j] && m.Score[j] >= thr {
						claimed[j] = true
						stack = append(stack, j)
					}
				}
			}
		}
		x0 := max((minTx-1)*m.Tile, 0)
		y0 := max((minTy-1)*m.Tile, 0)
		x1 := min((maxTx+2)*m.Tile, m.W)
		y1 := min((maxTy+2)*m.Tile, m.H)
		boxes = append(boxes, image.Rect(
			fb.Min.X+int(float64(x0)*sx), fb.Min.Y+int(float64(y0)*sy),
			fb.Min.X+int(float64(x1)*sx), fb.Min.Y+int(float64(y1)*sy)))
	}
	return boxes
}

// TestMegaBoxDecodeExperiment feeds the production ROI proposals of the glued
// print captures to the FULL Decode (pyramid + escalating probe) instead of
// the ROI retry's flat 512 px rung probe - isolating whether the crops fail
// for content or for probe resolution. Temporary diagnostic, not part of any
// gate.
func TestMegaBoxDecodeExperiment(t *testing.T) {
	dir := testutil.CapturePath(t)
	known := captureGroundTruth(t, dir)
	fixtures := []string{
		"print_camera/8c_frontal_rot45.webp",
		"print_camera/8c_side_rot45.webp",
		"print_camera/16c_frontal_rot45.webp",
	}
	for _, rel := range fixtures {
		img, err := loadCaptureImage(filepath.Join(dir, filepath.FromSlash(rel)))
		if err != nil {
			t.Skipf("load %s: %v", rel, err)
		}
		colors, _ := captureColorCount(rel)
		for i, roi := range detect.ProposeROIs(img, 2) {
			crop := detect.CropImage(img, roi.Bounds)
			data, err := Decode(crop)
			verdict := "FAIL"
			if err == nil {
				switch match := matchKnownPayload(data, known); {
				case match == colors:
					verdict = "OK"
				case match != 0:
					verdict = fmt.Sprintf("other-code %dc", match)
				default:
					verdict = "corrupt"
				}
			}
			t.Logf("%s roi%d %v (%dx%d) full Decode -> %s", rel, i, roi.Bounds, roi.Bounds.Dx(), roi.Bounds.Dy(), verdict)
		}
	}
}

// TestEscalatedROIRungExperiment checks whether the mega-crop win reduces to
// the escalating orientation probe alone: compute the crop's rungs via the
// same per-level escalation the frame pyramid uses, then decode the crop at
// FULL crop resolution per rung - no pyramid goroutines, no seeded route, no
// recursion. Temporary diagnostic, not part of any gate.
func TestEscalatedROIRungExperiment(t *testing.T) {
	dir := testutil.CapturePath(t)
	known := captureGroundTruth(t, dir)
	fixtures := []string{
		"print_camera/8c_frontal_rot45.webp",
		"print_camera/8c_side_rot45.webp",
		"print_camera/16c_frontal_rot45.webp",
	}
	for _, rel := range fixtures {
		img, err := loadCaptureImage(filepath.Join(dir, filepath.FromSlash(rel)))
		if err != nil {
			t.Skipf("load %s: %v", rel, err)
		}
		colors, _ := captureColorCount(rel)
		for i, roi := range detect.ProposeROIs(img, 2) {
			crop := detect.CropImage(img, roi.Bounds)
			var rungs []float64
			if levels := pyramidLevels(crop); levels != nil {
				if rungs = detect.CoarseOrientationRungs(levels[0]); len(rungs) == 0 {
					for k, lvl := range levels[1:] {
						fams := detect.CoarseProbeFamiliesWithin(lvl, detect.CoarseMaxDim<<(k+1))
						if rungs = detect.FamiliesToRungsUncapped(fams); len(rungs) > 0 {
							t.Logf("%s roi%d: escalated to level %d for rungs %v", rel, i, k+1, rungs)
							break
						}
					}
				} else {
					t.Logf("%s roi%d: coarsest rungs %v", rel, i, rungs)
				}
			} else {
				rungs = detect.CoarseOrientationRungs(crop)
				t.Logf("%s roi%d: single-level rungs %v", rel, i, rungs)
			}
			verdict := "FAIL"
			winDeg := math.NaN()
			for _, deg := range rungs {
				bm := detect.RotateToBitmap(crop, deg)
				data, stage, _ := decodeBitmapFinding(bm, nil, nil)
				if stage == readDecoded {
					switch match := matchKnownPayload(data, known); {
					case match == colors:
						verdict, winDeg = "OK", deg
					case match != 0:
						verdict, winDeg = fmt.Sprintf("other-code %dc", match), deg
					default:
						verdict, winDeg = "corrupt", deg
					}
					break
				}
			}
			t.Logf("%s roi%d %v: escalated-rung full-res decode -> %s (deg %g)", rel, i, roi.Bounds, verdict, winDeg)
		}
	}
}

// quadLadderOn locates finders on bm, logs the per-type candidate pool and
// the chosen quad, enumerates every candidate quad passing ScoreFinderQuad,
// and decodes the best topK from their quads directly. Returns true when a
// quad decodes the expected payload.
func quadLadderOn(t *testing.T, label string, bm *core.Bitmap, known map[int]captureTruth, colors, topK int) bool {
	ch := detect.BinarizerRGB(bm, nil)
	d := &detect.PrimaryDetector{BM: bm, Ch: ch, Mode: detect.IntensiveDetect}
	if !d.LocateFinders() {
		t.Logf("%s: no finders", label)
		return false
	}
	var g [4][]detect.FinderPattern
	for _, cand := range d.Candidates {
		if cand.Typ >= 0 && cand.Typ < 4 {
			g[cand.Typ] = append(g[cand.Typ], cand)
		}
	}
	chosen := detect.CalculateSideSize(bm, d.FPs)
	t.Logf("%s: cands %d/%d/%d/%d chosen quad ms %.1f/%.1f/%.1f/%.1f side %v",
		label, len(g[0]), len(g[1]), len(g[2]), len(g[3]),
		d.FPs[0].ModuleSize, d.FPs[1].ModuleSize, d.FPs[2].ModuleSize, d.FPs[3].ModuleSize, chosen)

	combos := 1
	for typ := range 4 {
		if len(g[typ]) == 0 {
			t.Logf("%s: empty type-%d pool, no quad ladder", label, typ)
			return false
		}
		combos *= len(g[typ])
	}
	if combos > 5_000_000 {
		t.Logf("%s: %d combos, skipping ladder", label, combos)
		return false
	}
	type scored struct {
		quad  [4]detect.FinderPattern
		score float64
	}
	var quads []scored
	for _, p0 := range g[0] {
		for _, p1 := range g[1] {
			for _, p2 := range g[2] {
				for _, p3 := range g[3] {
					if s, ok := detect.ScoreFinderQuad(p0, p1, p2, p3); ok {
						quads = append(quads, scored{[4]detect.FinderPattern{p0, p1, p2, p3}, s})
					}
				}
			}
		}
	}
	slices.SortStableFunc(quads, func(a, b scored) int {
		if a.score < b.score {
			return -1
		}
		if a.score > b.score {
			return 1
		}
		return 0
	})
	t.Logf("%s: %d passing quads of %d combos", label, len(quads), combos)
	for i, q := range quads[:min(topK, len(quads))] {
		side := detect.CalculateSideSize(bm, q.quad[:])
		if side.X <= 0 || side.Y <= 0 {
			t.Logf("%s quad %d score %.3f: invalid side", label, i, q.score)
			continue
		}
		data, ok := decodeFromQuad(bm, q.quad, side, func() bool { return false })
		verdict := "FAIL"
		if ok {
			switch match := matchKnownPayload(data, known); {
			case match == colors:
				verdict = "OK"
			case match != 0:
				verdict = fmt.Sprintf("other-code %dc", match)
			default:
				verdict = "corrupt"
			}
		}
		t.Logf("%s quad %d score %.3f side %v ms %.1f/%.1f/%.1f/%.1f -> %s",
			label, i, q.score, side,
			q.quad[0].ModuleSize, q.quad[1].ModuleSize, q.quad[2].ModuleSize, q.quad[3].ModuleSize, verdict)
		if verdict == "OK" {
			return true
		}
	}
	return false
}

// TestCropQuadLadderExperiment runs the quad ladder where the candidate
// pools are rich: on the ROI crops (print multi-code rows) and the best
// pyramid-level rungs (display side rows, single-code perspective) of the
// remaining pure-geometry failures. The existence test for a bounded
// multi-quad retry at working scale. Temporary diagnostic, not part of any
// gate.
func TestCropQuadLadderExperiment(t *testing.T) {
	dir := testutil.CapturePath(t)
	known := captureGroundTruth(t, dir)
	prints := []string{
		"print_camera/16c_frontal_rot45.webp",
		"print_camera/8c_side_rot45.webp",
		"print_camera/32c_frontal_rot45.webp",
	}
	for _, rel := range prints {
		img, err := loadCaptureImage(filepath.Join(dir, filepath.FromSlash(rel)))
		if err != nil {
			t.Skipf("load %s: %v", rel, err)
		}
		colors, _ := captureColorCount(rel)
		rois := detect.ProposeROIs(img, 2)
		if len(rois) == 0 {
			t.Logf("%s: no ROIs", rel)
			continue
		}
		crop := detect.CropImage(img, rois[0].Bounds)
		for _, deg := range roiRungs(crop) {
			bm := detect.RotateToBitmap(crop, deg)
			detect.BalanceRGB(bm)
			if quadLadderOn(t, fmt.Sprintf("%s roi0 rot%g", rel, deg), bm, known, colors, 6) {
				break
			}
		}
	}
	displays := []struct {
		rel   string
		level int
	}{
		{"display_camera/8c_side_rot45_normal.webp", 2},
		{"display_camera/16c_side_rot45_normal.webp", 2},
	}
	for _, c := range displays {
		img, err := loadCaptureImage(filepath.Join(dir, filepath.FromSlash(c.rel)))
		if err != nil {
			t.Skipf("load %s: %v", c.rel, err)
		}
		colors, _ := captureColorCount(c.rel)
		levels := pyramidLevels(img)
		if levels == nil || c.level >= len(levels) {
			t.Fatalf("%s: level %d unavailable", c.rel, c.level)
		}
		for _, deg := range []float64{45, 135, 225, 315} {
			bm := detect.RotateToBitmap(levels[c.level], deg)
			detect.BalanceRGB(bm)
			if quadLadderOn(t, fmt.Sprintf("%s L%d rot%g", c.rel, c.level, deg), bm, known, colors, 6) {
				break
			}
		}
	}
}

// scoreQuadRelaxed mirrors detect.ScoreFinderQuad with parameterized
// tolerances, so the experiment can test whether the production gates - not
// missing candidates - reject the true quad under strong perspective.
func scoreQuadRelaxed(p0, p1, p2, p3 detect.FinderPattern, edgeTol, moduleTol, consistTol float64) (float64, bool) {
	pts := [4][2]float64{
		{p0.Center.X, p0.Center.Y}, {p1.Center.X, p1.Center.Y},
		{p2.Center.X, p2.Center.Y}, {p3.Center.X, p3.Center.Y},
	}
	var sign float64
	for i := range 4 {
		a, b, c := pts[i], pts[(i+1)&3], pts[(i+2)&3]
		cross := (b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0])
		if cross == 0 {
			return 0, false
		}
		if i == 0 {
			sign = cross
		} else if (cross > 0) != (sign > 0) {
			return 0, false
		}
	}
	edge := func(a, b [2]float64) float64 { return math.Hypot(a[0]-b[0], a[1]-b[1]) }
	rat := func(a, b float64) float64 {
		if a <= 0 || b <= 0 {
			return math.Inf(1)
		}
		return math.Max(a, b) / math.Min(a, b)
	}
	top, right := edge(pts[0], pts[1]), edge(pts[1], pts[2])
	bot, left := edge(pts[2], pts[3]), edge(pts[3], pts[0])
	edgeDev := math.Max(rat(top, bot), rat(left, right))
	if edgeDev > edgeTol {
		return 0, false
	}
	msMin := min(p0.ModuleSize, p1.ModuleSize, p2.ModuleSize, p3.ModuleSize)
	msMax := max(p0.ModuleSize, p1.ModuleSize, p2.ModuleSize, p3.ModuleSize)
	if msMin <= 0 || msMax/msMin > moduleTol {
		return 0, false
	}
	ss := detect.CalculateSideSize(nil, []detect.FinderPattern{p0, p1, p2, p3})
	if ss.X <= 0 || ss.Y <= 0 {
		return 0, false
	}
	ms := (p0.ModuleSize + p1.ModuleSize + p2.ModuleSize + p3.ModuleSize) / 4
	consist := math.Max(
		math.Max(rat(top/float64(ss.X), ms), rat(bot/float64(ss.X), ms)),
		math.Max(rat(left/float64(ss.Y), ms), rat(right/float64(ss.Y), ms)),
	)
	if consist > consistTol {
		return 0, false
	}
	return (edgeDev - 1) + (msMax/msMin - 1) + (consist - 1), true
}

// TestRelaxedQuadLadderExperiment re-runs the quad ladder with
// perspective-relaxed gates (edge 1.9, module 2.2, consistency 1.9) on the
// cases whose pools were non-empty but yielded ZERO passing quads under the
// production tolerances - testing whether the gates, not the candidates,
// reject the true quad on side views. Temporary diagnostic, not part of any
// gate.
func TestRelaxedQuadLadderExperiment(t *testing.T) {
	dir := testutil.CapturePath(t)
	known := captureGroundTruth(t, dir)
	cases := []struct {
		rel   string
		level int
		degs  []float64
	}{
		{"display_camera/8c_side_rot45_normal.webp", 2, []float64{45, 225}},
		{"display_camera/16c_side_rot45_normal.webp", 2, []float64{45}},
	}
	const topK = 8
	for _, c := range cases {
		img, err := loadCaptureImage(filepath.Join(dir, filepath.FromSlash(c.rel)))
		if err != nil {
			t.Skipf("load %s: %v", c.rel, err)
		}
		colors, _ := captureColorCount(c.rel)
		levels := pyramidLevels(img)
		for _, deg := range c.degs {
			bm := detect.RotateToBitmap(levels[c.level], deg)
			detect.BalanceRGB(bm)
			ch := detect.BinarizerRGB(bm, nil)
			d := &detect.PrimaryDetector{BM: bm, Ch: ch, Mode: detect.IntensiveDetect}
			if !d.LocateFinders() {
				t.Logf("%s L%d rot%g: no finders", c.rel, c.level, deg)
				continue
			}
			var g [4][]detect.FinderPattern
			for _, cand := range d.Candidates {
				if cand.Typ >= 0 && cand.Typ < 4 {
					g[cand.Typ] = append(g[cand.Typ], cand)
				}
			}
			type scored struct {
				quad  [4]detect.FinderPattern
				score float64
			}
			var quads []scored
			for _, p0 := range g[0] {
				for _, p1 := range g[1] {
					for _, p2 := range g[2] {
						for _, p3 := range g[3] {
							if s, ok := scoreQuadRelaxed(p0, p1, p2, p3, 1.9, 2.2, 1.9); ok {
								quads = append(quads, scored{[4]detect.FinderPattern{p0, p1, p2, p3}, s})
							}
						}
					}
				}
			}
			slices.SortStableFunc(quads, func(a, b scored) int {
				if a.score < b.score {
					return -1
				}
				if a.score > b.score {
					return 1
				}
				return 0
			})
			t.Logf("%s L%d rot%g: cands %d/%d/%d/%d, %d passing quads under relaxed gates",
				c.rel, c.level, deg, len(g[0]), len(g[1]), len(g[2]), len(g[3]), len(quads))
			for i, q := range quads[:min(topK, len(quads))] {
				side := detect.CalculateSideSize(bm, q.quad[:])
				if side.X <= 0 || side.Y <= 0 {
					continue
				}
				data, ok := decodeFromQuad(bm, q.quad, side, func() bool { return false })
				verdict := "FAIL"
				if ok {
					switch match := matchKnownPayload(data, known); {
					case match == colors:
						verdict = "OK"
					case match != 0:
						verdict = fmt.Sprintf("other-code %dc", match)
					default:
						verdict = "corrupt"
					}
				}
				t.Logf("  quad %d score %.3f side %v ms %.1f/%.1f/%.1f/%.1f -> %s",
					i, q.score, side,
					q.quad[0].ModuleSize, q.quad[1].ModuleSize, q.quad[2].ModuleSize, q.quad[3].ModuleSize, verdict)
				if verdict == "OK" {
					break
				}
			}
		}
	}
}

// overlayFinderCandidates draws every finder candidate (colour-coded by
// type: 0 red, 1 green, 2 blue, 3 yellow; cross size by module size) plus
// the chosen quad (white, when a quad was located) onto the pre-balance
// canvas of bm, and writes the overlay PNG into outDir. The PNG is written
// when force is set or any candidate exists. Returns whether the detector
// located a quad.
func overlayFinderCandidates(t *testing.T, label string, bm *core.Bitmap, outDir string, force bool) bool {
	canvas := bm.NRGBA()
	detect.BalanceRGB(bm)
	ch := detect.BinarizerRGB(bm, nil)
	d := &detect.PrimaryDetector{BM: bm, Ch: ch, Mode: detect.IntensiveDetect}
	found := d.LocateFinders()
	typeCol := [4]color.NRGBA{
		{R: 255, A: 255},         // type 0 red
		{G: 255, A: 255},         // type 1 green
		{B: 255, A: 255},         // type 2 blue
		{R: 255, G: 255, A: 255}, // type 3 yellow
	}
	mark := func(x, y, r int, c color.NRGBA) {
		for dx := -r; dx <= r; dx++ {
			for _, p := range [][2]int{{x + dx, y}, {x, y + dx}} {
				if p[0] >= 0 && p[0] < canvas.Rect.Dx() && p[1] >= 0 && p[1] < canvas.Rect.Dy() {
					canvas.SetNRGBA(p[0], p[1], c)
				}
			}
		}
	}
	var counts [4]int
	for _, cand := range d.Candidates {
		if cand.Typ < 0 || cand.Typ > 3 {
			continue
		}
		counts[cand.Typ]++
		r := max(int(cand.ModuleSize*2), 6)
		mark(int(cand.Center.X), int(cand.Center.Y), r, typeCol[cand.Typ])
	}
	side := image.Point{}
	if found {
		for _, fp := range d.FPs {
			mark(int(fp.Center.X), int(fp.Center.Y), 24, color.NRGBA{R: 255, G: 255, B: 255, A: 255})
		}
		side = detect.CalculateSideSize(bm, d.FPs)
	}
	if !force && len(d.Candidates) == 0 {
		t.Logf("%s: no candidates, overlay skipped", label)
		return found
	}
	name := strings.NewReplacer("/", "_", ".webp", "", " ", "_", ".", "p").Replace(label) + ".png"
	out := filepath.Join(outDir, name)
	f, err := os.Create(out)
	if err != nil {
		t.Fatalf("create %s: %v", out, err)
	}
	if err := png.Encode(f, canvas); err != nil {
		t.Fatalf("encode: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	t.Logf("%s: cands %d/%d/%d/%d quad=%v side %v -> %s (%dx%d)",
		label, counts[0], counts[1], counts[2], counts[3], found, side, out,
		canvas.Rect.Dx(), canvas.Rect.Dy())
	return found
}

// TestCandidateOverlayExperiment writes finder-candidate overlays under
// $JABSCRATCH_DIR/candidate_overlays (skipping when unset) for the remaining
// pure-geometry capture failures: the display side rows on their best pyramid
// rung, and the print residue rows on their first ROI crop across the crop's
// escalated rungs. Visual inspection splits the rows between
// missing-true-corner candidates (finder-scan sensitivity) and cross-code
// candidate confusion (multi-code scenes). Temporary diagnostic, not part of
// any gate.
func TestCandidateOverlayExperiment(t *testing.T) {
	scratch := os.Getenv("JABSCRATCH_DIR")
	if scratch == "" {
		t.Skip("JABSCRATCH_DIR not set; overlays need an output directory")
	}
	dir := testutil.CapturePath(t)
	outDir := filepath.Join(scratch, "candidate_overlays")
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		t.Fatal(err)
	}
	displays := []struct {
		rel   string
		level int
		degs  []float64
	}{
		{"display_camera/8c_side_rot45_normal.webp", 2, []float64{45}},
		{"display_camera/16c_side_rot45_normal.webp", 2, []float64{45}},
	}
	for _, c := range displays {
		img, err := loadCaptureImage(filepath.Join(dir, filepath.FromSlash(c.rel)))
		if err != nil {
			t.Skipf("load %s: %v", c.rel, err)
		}
		levels := pyramidLevels(img)
		if levels == nil || c.level >= len(levels) {
			t.Fatalf("%s: level %d unavailable", c.rel, c.level)
		}
		for _, deg := range c.degs {
			bm := detect.RotateToBitmap(levels[c.level], deg)
			overlayFinderCandidates(t, fmt.Sprintf("%s L%d rot%g", c.rel, c.level, deg), bm, outDir, false)
		}
	}
	prints := []string{
		"print_camera/8c_side_rot45.webp",
		"print_camera/16c_frontal_rot45.webp",
		"print_camera/32c_frontal_rot45.webp",
		"print_camera/64c_side_rot45.webp",
	}
	for _, rel := range prints {
		img, err := loadCaptureImage(filepath.Join(dir, filepath.FromSlash(rel)))
		if err != nil {
			t.Skipf("load %s: %v", rel, err)
		}
		rois := detect.ProposeROIs(img, 2)
		if len(rois) == 0 {
			t.Logf("%s: no ROIs", rel)
			continue
		}
		crop := detect.CropImage(img, rois[0].Bounds)
		rungs := roiRungs(crop)
		t.Logf("%s roi0 %v (%dx%d): rungs %v", rel, rois[0].Bounds,
			rois[0].Bounds.Dx(), rois[0].Bounds.Dy(), rungs)
		wrote := false
		for _, deg := range rungs {
			bm := detect.RotateToBitmap(crop, deg)
			label := fmt.Sprintf("%s roi0 rot%g", rel, deg)
			if overlayFinderCandidates(t, label, bm, outDir, false) {
				wrote = true
			}
		}
		if !wrote && len(rungs) > 0 {
			// No rung located a quad; force one plain-context overlay so the
			// crop content itself is inspectable.
			bm := detect.RotateToBitmap(crop, rungs[0])
			overlayFinderCandidates(t, fmt.Sprintf("%s roi0 rot%g ctx", rel, rungs[0]), bm, outDir, true)
		}
	}
}

// TestQuadLadderExperiment enumerates ALL finder-candidate quads at the rungs
// where the print rot45 capture sampled cross-code grids, ranks them by
// ScoreFinderQuad, and decodes the best K from their quads directly - the
// existence test for a bounded multi-quad retry: does ANY candidate quad
// decode the main code where the per-type chosen quad glues across codes?
// Temporary diagnostic, not part of any gate.
func TestQuadLadderExperiment(t *testing.T) {
	dir := testutil.CapturePath(t)
	known := captureGroundTruth(t, dir)
	rel := "print_camera/8c_frontal_rot45.webp"
	img, err := loadCaptureImage(filepath.Join(dir, filepath.FromSlash(rel)))
	if err != nil {
		t.Skipf("load %s: %v", rel, err)
	}
	levels := pyramidLevels(img)
	if levels == nil {
		t.Fatal("no pyramid levels")
	}
	cases := []struct {
		level int
		deg   float64
	}{
		{1, 30}, {1, 45}, {1, 120}, {2, 30}, {2, 45}, {3, 45},
	}
	const topK = 6
	for _, c := range cases {
		if c.level >= len(levels) {
			continue
		}
		bm := detect.RotateToBitmap(levels[c.level], c.deg)
		detect.BalanceRGB(bm)
		ch := detect.BinarizerRGB(bm, nil)
		d := &detect.PrimaryDetector{BM: bm, Ch: ch, Mode: detect.IntensiveDetect}
		if !d.LocateFinders() {
			t.Logf("%s L%d rot%g: no finders", rel, c.level, c.deg)
			continue
		}
		var g [4][]detect.FinderPattern
		for _, cand := range d.Candidates {
			if cand.Typ >= 0 && cand.Typ < 4 {
				g[cand.Typ] = append(g[cand.Typ], cand)
			}
		}
		chosen := detect.CalculateSideSize(bm, d.FPs)
		t.Logf("%s L%d rot%g: cands %d/%d/%d/%d chosen quad ms %.1f/%.1f/%.1f/%.1f side %v",
			rel, c.level, c.deg, len(g[0]), len(g[1]), len(g[2]), len(g[3]),
			d.FPs[0].ModuleSize, d.FPs[1].ModuleSize, d.FPs[2].ModuleSize, d.FPs[3].ModuleSize, chosen)

		type scored struct {
			quad  [4]detect.FinderPattern
			score float64
		}
		var quads []scored
		for _, p0 := range g[0] {
			for _, p1 := range g[1] {
				for _, p2 := range g[2] {
					for _, p3 := range g[3] {
						if s, ok := detect.ScoreFinderQuad(p0, p1, p2, p3); ok {
							quads = append(quads, scored{[4]detect.FinderPattern{p0, p1, p2, p3}, s})
						}
					}
				}
			}
		}
		slices.SortStableFunc(quads, func(a, b scored) int {
			if a.score != b.score {
				if a.score < b.score {
					return -1
				}
				return 1
			}
			return 0
		})
		t.Logf("  %d passing quads", len(quads))
		for i, q := range quads[:min(topK, len(quads))] {
			side := detect.CalculateSideSize(bm, q.quad[:])
			if side.X <= 0 || side.Y <= 0 {
				t.Logf("  quad %d score %.3f: invalid side", i, q.score)
				continue
			}
			data, ok := decodeFromQuad(bm, q.quad, side, func() bool { return false })
			verdict := "FAIL"
			if ok {
				switch match := matchKnownPayload(data, known); {
				case match == 8:
					verdict = "OK"
				case match != 0:
					verdict = fmt.Sprintf("other-code %dc", match)
				default:
					verdict = "corrupt"
				}
			}
			t.Logf("  quad %d score %.3f side %v ms %.1f/%.1f/%.1f/%.1f -> %s",
				i, q.score, side,
				q.quad[0].ModuleSize, q.quad[1].ModuleSize, q.quad[2].ModuleSize, q.quad[3].ModuleSize, verdict)
		}
	}
}

// finderRegistration scores a sampled matrix against the true module matrix
// over the four finder-pattern 5x5 neighbourhoods, classifying each sampled
// module to the nearest colour of pal (the canonical palette, or a captured
// copy for the decoder's own view) in absolute RGB. High mismatch means the
// sampling grid is misplaced or the colours have collapsed; comparing the
// canonical and captured-palette scores separates cast from misregistration.
// Returns the mismatch count, total compared, and per-corner core matches.
func finderRegistration(matrix *core.Bitmap, truth []byte, side image.Point, colorNumber int, pal []byte) (mismatch, total int, coreOK [4]bool) {
	cx := [4]int{spec.DistanceToBorder - 1, side.X - spec.DistanceToBorder, side.X - spec.DistanceToBorder, spec.DistanceToBorder - 1}
	cy := [4]int{spec.DistanceToBorder - 1, spec.DistanceToBorder - 1, side.Y - spec.DistanceToBorder, side.Y - spec.DistanceToBorder}
	bpp := matrix.Channels
	for corner := range 4 {
		for dy := -2; dy <= 2; dy++ {
			for dx := -2; dx <= 2; dx++ {
				x, y := cx[corner]+dx, cy[corner]+dy
				if x < 0 || y < 0 || x >= side.X || y >= side.Y {
					continue
				}
				off := y*matrix.Width*bpp + x*bpp
				best, bi := math.Inf(1), 0
				for i := range colorNumber {
					dr := float64(matrix.Pix[off+0]) - float64(pal[i*3+0])
					dg := float64(matrix.Pix[off+1]) - float64(pal[i*3+1])
					db := float64(matrix.Pix[off+2]) - float64(pal[i*3+2])
					if d := dr*dr + dg*dg + db*db; d < best {
						best, bi = d, i
					}
				}
				total++
				match := byte(bi) == truth[y*side.X+x]
				if !match {
					mismatch++
				}
				if dx == 0 && dy == 0 {
					coreOK[corner] = match
				}
			}
		}
	}
	return mismatch, total, coreOK
}

// samplePaletteCopies reads the embedded palette copies from a sampled matrix.
// When Part I misreads, the known colour mode is forced and the walk restarts
// after the four Part I modules (the non-default layout, which every >8-colour
// symbol has) - the experiment's fixtures are all known non-default. Returns
// nil when the palette walk leaves the matrix.
func samplePaletteCopies(t *testing.T, matrix *core.Bitmap, side image.Point, trueNC int) []byte {
	sym := &core.DecodedSymbol{SideSize: side}
	dataMap := make([]byte, matrix.Width*matrix.Height)
	x, y, count := spec.PrimaryMetadataX, spec.PrimaryMetadataY, 0
	ret, _ := decode.DecodePrimaryMetadataPartI(matrix, sym, dataMap, &count, &x, &y)
	t.Logf("  Part I ret=%d Nc=%d (true Nc %d)", ret, sym.Meta.NC, trueNC)
	if ret != core.Success || sym.Meta.NC != trueNC {
		sym.Meta.NC = trueNC
		x, y, count = spec.PrimaryMetadataX, spec.PrimaryMetadataY, 0
		for count < spec.PrimaryMetadataPart1ModuleNumber {
			count++
			spec.NextMetadataModuleInPrimary(matrix.Height, matrix.Width, count, &x, &y)
		}
	}
	if decode.ReadColorPaletteInPrimary(matrix, sym, dataMap, &count, &x, &y) != core.Success {
		t.Logf("  palette walk left the matrix")
		return nil
	}
	return sym.Palette
}

// logPaletteCopies dumps per-copy palette statistics against the canonical
// palette, then the cross-copy disagreement split into its DC part (a global
// shift between the two copies' readings, which the Part I offset+gain retry
// could model) and the per-colour residual (the spatially varying part it
// cannot). Only the 2-copy >8-colour layout is handled.
func logPaletteCopies(t *testing.T, pal []byte, colorNumber int) {
	copies := spec.PaletteCopies(colorNumber)
	canonical := palette.SetDefault(colorNumber)
	if copies != 2 || len(pal) < colorNumber*3*2 {
		t.Logf("  unsupported layout (copies=%d len=%d)", copies, len(pal))
		return
	}
	n := float64(colorNumber)
	for cp := range copies {
		base := cp * colorNumber * 3
		var sumErr float64
		var off [3]float64
		for c := range colorNumber {
			for ch := range 3 {
				d := float64(pal[base+c*3+ch]) - float64(canonical[c*3+ch])
				sumErr += math.Abs(d)
				off[ch] += d
			}
		}
		t.Logf("  copy %d vs canonical: meanAbsErr %.1f, offset r%+.1f g%+.1f b%+.1f",
			cp, sumErr/(n*3), off[0]/n, off[1]/n, off[2]/n)
	}
	var dc [3]float64
	diffs := make([]float64, colorNumber*3)
	for c := range colorNumber {
		for ch := range 3 {
			d := float64(pal[colorNumber*3+c*3+ch]) - float64(pal[c*3+ch])
			diffs[c*3+ch] = d
			dc[ch] += d
		}
	}
	dc[0], dc[1], dc[2] = dc[0]/n, dc[1]/n, dc[2]/n
	var absSum, resSum float64
	perColour := make([]float64, colorNumber)
	for c := range colorNumber {
		for ch := range 3 {
			absSum += math.Abs(diffs[c*3+ch])
			resSum += math.Abs(diffs[c*3+ch] - dc[ch])
			perColour[c] += math.Abs(diffs[c*3+ch]) / 3
		}
	}
	t.Logf("  cross-copy |copy1-copy0|: mean %.1f, DC r%+.1f g%+.1f b%+.1f, residual after DC %.1f",
		absSum/(n*3), dc[0], dc[1], dc[2], resSum/(n*3))
	order := make([]int, colorNumber)
	for i := range order {
		order[i] = i
	}
	slices.SortStableFunc(order, func(a, b int) int {
		if perColour[a] != perColour[b] {
			if perColour[a] > perColour[b] {
				return -1
			}
			return 1
		}
		return a - b
	})
	for _, c := range order[:4] {
		t.Logf("  colour %3d canonical(%3d,%3d,%3d): copy0(%3d,%3d,%3d) copy1(%3d,%3d,%3d) |d|=%.0f",
			c, canonical[c*3+0], canonical[c*3+1], canonical[c*3+2],
			pal[c*3+0], pal[c*3+1], pal[c*3+2],
			pal[colorNumber*3+c*3+0], pal[colorNumber*3+c*3+1], pal[colorNumber*3+c*3+2],
			perColour[c])
	}
}

// TestPaletteCopyExperiment measures the real per-copy embedded palettes on
// verified-geometry samples of the 32c/64c frontal capture rows: re-encode
// the known payload for the ground-truth matrix, locate finders, sample at
// BOTH the estimated and the true side size, check registration on the finder
// neighbourhoods, and only then compare the two palette copies. The source
// PNGs and the decoding rows are controls. Temporary diagnostic, not part of
// any gate.
func TestPaletteCopyExperiment(t *testing.T) {
	dir := testutil.CapturePath(t)
	known := captureGroundTruth(t, dir)
	truths := map[int]encode.Rendered{}
	for _, colors := range []int{32, 64} {
		r, err := encode.Render(encode.Config{Colors: colors, ModuleSize: 5, ECCLevel: 10, SymbolNumber: 1}, known[colors].payload)
		if err != nil {
			t.Fatalf("re-encode %dc: %v", colors, err)
		}
		if r.SideSize != known[colors].side {
			t.Fatalf("re-encode %dc side %v, truth %v", colors, r.SideSize, known[colors].side)
		}
		truths[colors] = r
	}
	cases := []struct {
		rel    string
		colors int
	}{
		{"source/32c_ecc10_v13_lorem_ms5.png", 32},
		{"display_camera/32c_frontal_rot00_normal.webp", 32},
		{"display_camera/32c_frontal_rot00_redshift.webp", 32},
		{"print_camera/32c_frontal_rot00.webp", 32},
		{"source/64c_ecc10_v11_lorem_ms5.png", 64},
		{"display_camera/64c_frontal_rot00_normal.webp", 64},
	}
	for _, c := range cases {
		img, err := loadCaptureImage(filepath.Join(dir, filepath.FromSlash(c.rel)))
		if err != nil {
			t.Skipf("load %s: %v", c.rel, err)
		}
		truth := truths[c.colors]
		bm := core.BitmapFromImage(img)
		detect.BalanceRGB(bm)
		ch := detect.BinarizerRGB(bm, nil)
		d := &detect.PrimaryDetector{BM: bm, Ch: ch, Mode: detect.IntensiveDetect}
		if !d.LocateFinders() {
			t.Logf("%s: no finders", c.rel)
			continue
		}
		est := detect.CalculateSideSize(bm, d.FPs)
		t.Logf("%s: estimated side %v (true %v)", c.rel, est, truth.SideSize)
		sides := []image.Point{est}
		if est != truth.SideSize {
			sides = append(sides, truth.SideSize)
		}
		for _, side := range sides {
			if side.X <= 0 || side.Y <= 0 {
				continue
			}
			pt := core.PerspectiveTransform(d.FPs[0].Center, d.FPs[1].Center, d.FPs[2].Center, d.FPs[3].Center, side)
			var matrix *core.Bitmap
			if d.PrintDetected() {
				matrix = detect.SampleSymbolOffset(bm, pt, side, detect.SearchChannelOffsets(bm, pt, side))
			} else {
				matrix = detect.SampleSymbol(bm, pt, side)
			}
			if matrix == nil {
				t.Logf("%s side %v: sample failed", c.rel, side)
				continue
			}
			nc := bits.Len(uint(c.colors)) - 2
			pal := samplePaletteCopies(t, matrix, side, nc)
			if side == truth.SideSize {
				mism, tot, coreOK := finderRegistration(matrix, truth.Matrix, side, c.colors, palette.SetDefault(c.colors))
				t.Logf("%s side %v: finder registration vs canonical %d/%d mismatched, cores %v", c.rel, side, mism, tot, coreOK)
				if pal != nil {
					mism, tot, coreOK = finderRegistration(matrix, truth.Matrix, side, c.colors, pal[:c.colors*3])
					t.Logf("%s side %v: finder registration vs captured copy0 %d/%d mismatched, cores %v", c.rel, side, mism, tot, coreOK)
				}
			} else {
				t.Logf("%s side %v: WRONG grid, palette numbers below are for contrast only", c.rel, side)
			}
			if pal != nil {
				logPaletteCopies(t, pal, c.colors)
			}
		}
	}
}

// TestSideSizeExperiment dumps the per-edge module-count evidence (local walk
// vs distance estimate, rounded size, reliability flag, endpoint module sizes)
// for captures whose side-size estimate is known wrong. Temporary diagnostic,
// not part of any gate.
func TestSideSizeExperiment(t *testing.T) {
	cases := []struct {
		rel   string
		level int // -1 = full resolution, no rotation
		deg   float64
	}{
		{"source/256c_ecc10_v9_lorem_ms6.png", -1, 0},
		{"display_camera/8c_side_rot45_normal.webp", 2, 45},
		{"display_camera/16c_side_rot45_normal.webp", 2, 45},
	}
	for _, c := range cases {
		path := filepath.Join(testutil.CapturePath(t), filepath.FromSlash(c.rel))
		img, err := loadCaptureImage(path)
		if err != nil {
			t.Skipf("load %s: %v", c.rel, err)
		}
		var bm *core.Bitmap
		if c.level < 0 {
			bm = core.BitmapFromImage(img)
		} else {
			levels := pyramidLevels(img)
			if levels == nil || c.level >= len(levels) {
				t.Fatalf("%s: level %d not available", c.rel, c.level)
			}
			bm = detect.RotateToBitmap(levels[c.level], c.deg)
		}
		detect.BalanceRGB(bm)
		ch := detect.BinarizerRGB(bm, nil)
		d := &detect.PrimaryDetector{BM: bm, Ch: ch, Mode: detect.IntensiveDetect}
		if !d.LocateFinders() {
			t.Logf("%s level %d rot %g: no finders", c.rel, c.level, c.deg)
			continue
		}
		fps := d.FPs
		edges := []struct {
			name string
			a, b int
		}{{"top(0-1)", 0, 1}, {"bottom(3-2)", 3, 2}, {"left(0-3)", 0, 3}, {"right(1-2)", 1, 2}}
		for _, e := range edges {
			w := detect.LocalModuleCount(bm, fps[e.a], fps[e.b])
			dist := detect.CalculateModuleNumber(fps[e.a], fps[e.b])
			sw, fw := detect.SideSize(w + 7)
			sd, fd := detect.SideSize(dist + 7)
			t.Logf("%s %s: walk=%d(->%d,flag%d) dist=%d(->%d,flag%d) msA=%.2f msB=%.2f",
				c.rel, e.name, w, sw, fw, dist, sd, fd, fps[e.a].ModuleSize, fps[e.b].ModuleSize)
		}
		t.Logf("%s: CalculateSideSize=%v", c.rel, detect.CalculateSideSize(bm, fps))
		if c.level < 0 {
			data, ok, _ := DecodeImage(img)
			t.Logf("%s: upright DecodeImage ok=%v payload=%d bytes", c.rel, ok, len(data))
		}
	}
}
