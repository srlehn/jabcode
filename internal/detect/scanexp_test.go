//go:build jabharness

package detect

import (
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	_ "image/png"
	"os"
	"path/filepath"
	"slices"
	"testing"

	_ "golang.org/x/image/webp"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/testutil"
)

// scanLoadImage decodes a capture fixture (png/jpeg/webp).
func scanLoadImage(t *testing.T, rel string) image.Image {
	path := filepath.Join(testutil.TestdataPath("highcolor_capture"), filepath.FromSlash(rel))
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("open %s: %v", rel, err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		t.Skipf("decode %s: %v", rel, err)
	}
	return img
}

// scanPyramidLevel rebuilds the read package's pyramid (full-res opaque base,
// halved while the shorter side stays at or above 600, coarsest first) and
// returns the requested level.
func scanPyramidLevel(t *testing.T, img image.Image, level int) *image.NRGBA {
	bm := core.BitmapFromImage(img)
	for i := 3; i < len(bm.Pix); i += 4 {
		bm.Pix[i] = 255
	}
	levels := []*image.NRGBA{bm.NRGBA()}
	for {
		last := levels[len(levels)-1]
		if min(last.Rect.Dx(), last.Rect.Dy()) < 600 {
			break
		}
		levels = append(levels, HalveNRGBA(last))
	}
	slices.Reverse(levels)
	if level >= len(levels) {
		t.Fatalf("level %d unavailable (%d levels)", level, len(levels))
	}
	return levels[level]
}

// scanSeedStage replays the horizontal seed pipeline of findPrimarySymbol for
// one green-channel seed and names the stage where it dies.
func scanSeedStage(d *PrimaryDetector, i int, ps patternScan) (FinderPattern, string) {
	ch := d.Ch
	w := ch[0].Width
	rowR := ch[0].Pix[i*w : (i+1)*w]
	rowG := ch[1].Pix[i*w : (i+1)*w]
	rowB := ch[2].Pix[i*w : (i+1)*w]

	centerxG, moduleSizeG := ps.Center, ps.ModuleSize
	typeG := core.BoolColor(rowG[int(centerxG)] > 0)
	centerxR, centerxB := centerxG, centerxG
	var typeR, typeB int
	var moduleSizeR, moduleSizeB float64
	fp1found := false
	slack := d.ccSlack(moduleSizeG)

	var fp FinderPattern
	if crossCheckPatternHorizontal(ch[2], moduleSizeG*2, &centerxB, float64(i), &moduleSizeB, slack) {
		typeB = core.BoolColor(rowB[int(centerxB)] > 0)
		moduleSizeR = moduleSizeG
		coreRed := int(palette.Default[spec.FP3CoreColor*3+0])
		if !crossCheckColor(ch[0], coreRed, int(moduleSizeR), 5, int(centerxR), i, 0, slack) {
			return fp, "blue-h ok, red colour cross-check FAIL"
		}
		typeR = 0
		fp1found = true
	} else if crossCheckPatternHorizontal(ch[0], moduleSizeG*2, &centerxR, float64(i), &moduleSizeR, slack) {
		typeR = core.BoolColor(rowR[int(centerxR)] > 0)
		moduleSizeB = moduleSizeG
		coreBlue := int(palette.Default[spec.FP2CoreColor*3+2])
		if !crossCheckColor(ch[2], coreBlue, int(moduleSizeB), 5, int(centerxB), i, 0, slack) {
			return fp, "red-h ok, blue colour cross-check FAIL"
		}
		typeB = 0
	} else {
		return fp, "green only: neither blue nor red horizontal pattern"
	}

	fp = FinderPattern{Center: core.PointF{Y: float64(i)}, FoundCount: 1}
	if fp1found {
		if !checkModuleSize2(moduleSizeG, moduleSizeB) {
			return fp, fmt.Sprintf("module-size pair FAIL g%.1f b%.1f", moduleSizeG, moduleSizeB)
		}
		fp.Center.X = (centerxG + centerxB) / 2.0
		fp.ModuleSize = (moduleSizeG + moduleSizeB) / 2.0
		if !fp.classify([]int{fp0, fp3}, typeR, typeG, typeB) {
			return fp, fmt.Sprintf("classify FAIL rgb=%d/%d/%d (fp0/fp3 path)", typeR, typeG, typeB)
		}
	} else {
		if !checkModuleSize2(moduleSizeR, moduleSizeG) {
			return fp, fmt.Sprintf("module-size pair FAIL r%.1f g%.1f", moduleSizeR, moduleSizeG)
		}
		fp.Center.X = (centerxR + centerxG) / 2.0
		fp.ModuleSize = (moduleSizeR + moduleSizeG) / 2.0
		if !fp.classify([]int{fp1, fp2}, typeR, typeG, typeB) {
			return fp, fmt.Sprintf("classify FAIL rgb=%d/%d/%d (fp1/fp2 path)", typeR, typeG, typeB)
		}
	}
	if !crossCheckPattern(d.Ch, &fp, 0, d.ccSlack(fp.ModuleSize)) {
		return fp, fmt.Sprintf("full cross-check FAIL as type %d ms %.1f", fp.Typ, fp.ModuleSize)
	}
	return fp, fmt.Sprintf("SURVIVES as type %d at (%.0f,%.0f) ms %.1f", fp.Typ, fp.Center.X, fp.Center.Y, fp.ModuleSize)
}

// scanWindow replays the production horizontal seed scan on every row of win,
// logging each seed's fate, and additionally probes the red and blue channels
// for the same run-length pattern (production seeds only from green - if the
// pattern lives in the other channels but never in green, seeding is the
// bottleneck, not the checks). Only rejection stages are counted per row;
// verbose per-seed lines go to the log.
func scanWindow(t *testing.T, d *PrimaryDetector, win image.Rectangle, label string) {
	ch := d.Ch
	w, h := ch[0].Width, ch[0].Height
	win = win.Intersect(image.Rect(0, 0, w, h))
	counts := map[string]int{}
	chanHits := [3]int{}
	for i := win.Min.Y; i < win.Max.Y; i++ {
		rowG := ch[1].Pix[i*w : (i+1)*w]
		startx := win.Min.X
		for startx < win.Max.X {
			ps := seekPatternHorizontal(rowG, startx, win.Max.X)
			if !ps.ok {
				break
			}
			_, stage := scanSeedStage(d, i, ps)
			counts[stage]++
			if stage[0] == 'S' || counts[stage] <= 3 {
				t.Logf("%s row %d: green seed x=%.1f ms=%.1f -> %s", label, i, ps.Center, ps.ModuleSize, stage)
			}
			adv := ps.start + max(ps.skip, 1)
			if adv <= startx {
				adv = startx + 1
			}
			startx = adv
		}
		for c := range 3 {
			row := ch[c].Pix[i*w : (i+1)*w]
			if ps := seekPatternHorizontal(row, win.Min.X, win.Max.X); ps.ok {
				chanHits[c]++
			}
		}
	}
	t.Logf("%s window %v: rows with r/g/b pattern %d/%d/%d of %d", label, win,
		chanHits[0], chanHits[1], chanHits[2], win.Dy())
	for stage, n := range counts {
		t.Logf("%s   %4d x %s", label, n, stage)
	}
}

// scanDumpWindow writes two crops of win under
// $JABSCRATCH_DIR/scan_instrument (a no-op when unset): the colour canvas and
// the binarized-channel composite (R/G/B = the three binary channels), which
// is exactly what the run-length machine sees.
func scanDumpWindow(t *testing.T, bm *core.Bitmap, ch [3]*core.Bitmap, win image.Rectangle, name string) {
	scratch := os.Getenv("JABSCRATCH_DIR")
	if scratch == "" {
		return
	}
	outDir := filepath.Join(scratch, "scan_instrument")
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		t.Fatal(err)
	}
	win = win.Intersect(image.Rect(0, 0, bm.Width, bm.Height))
	colour := image.NewNRGBA(image.Rect(0, 0, win.Dx(), win.Dy()))
	binary := image.NewNRGBA(image.Rect(0, 0, win.Dx(), win.Dy()))
	for y := win.Min.Y; y < win.Max.Y; y++ {
		for x := win.Min.X; x < win.Max.X; x++ {
			o := ((y-win.Min.Y)*win.Dx() + (x - win.Min.X)) * 4
			bo := (y*bm.Width + x) * bm.Channels
			colour.Pix[o+0], colour.Pix[o+1], colour.Pix[o+2], colour.Pix[o+3] = bm.Pix[bo+0], bm.Pix[bo+1], bm.Pix[bo+2], 255
			binary.Pix[o+0], binary.Pix[o+1], binary.Pix[o+2], binary.Pix[o+3] = ch[0].Pix[y*bm.Width+x], ch[1].Pix[y*bm.Width+x], ch[2].Pix[y*bm.Width+x], 255
		}
	}
	for suffix, img := range map[string]*image.NRGBA{"colour": colour, "binary": binary} {
		out := filepath.Join(outDir, name+"_"+suffix+".png")
		f, err := os.Create(out)
		if err != nil {
			t.Fatal(err)
		}
		if err := png.Encode(f, img); err != nil {
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s", out)
	}
}

// bestOfType returns the highest-FoundCount candidate of the given type.
func bestOfType(cands []FinderPattern, typ int) (FinderPattern, bool) {
	var best FinderPattern
	found := false
	for _, c := range cands {
		if c.Typ == typ && (!found || c.FoundCount > best.FoundCount) {
			best, found = c, true
		}
	}
	return best, found
}

// TestFinderScanWindowExperiment instruments the per-type finder scan at the
// KNOWN true corner regions of the remaining pure-geometry capture failures,
// on the raw binarization pass. Anchors are the strong left-corner candidates
// (their windows double as instrument controls); the right-corner windows
// cover the full perspective-compressed range. Temporary diagnostic, not part
// of any gate.
func TestFinderScanWindowExperiment(t *testing.T) {
	img := scanLoadImage(t, "display_camera/8c_side_rot45_normal.webp")
	lvl := scanPyramidLevel(t, img, 2)
	bm := RotateToBitmap(lvl, 45)
	BalanceRGB(bm)
	ch := BinarizerRGB(bm, nil)
	d := &PrimaryDetector{BM: bm, Ch: ch, Mode: IntensiveDetect}
	status := d.findPrimarySymbol()
	var g [4]int
	for _, c := range d.Candidates {
		if c.Typ >= 0 && c.Typ < 4 {
			g[c.Typ]++
		}
	}
	t.Logf("raw pass: status %d, cands %d/%d/%d/%d", status, g[0], g[1], g[2], g[3])

	fp0c, ok0 := bestOfType(d.Candidates, 0)
	fp3c, ok3 := bestOfType(d.Candidates, 3)
	if !ok0 || !ok3 {
		t.Fatalf("left anchors missing: type0=%v type3=%v", ok0, ok3)
	}
	t.Logf("anchors: type0 (%.0f,%.0f) ms %.1f found %d; type3 (%.0f,%.0f) ms %.1f found %d",
		fp0c.Center.X, fp0c.Center.Y, fp0c.ModuleSize, fp0c.FoundCount,
		fp3c.Center.X, fp3c.Center.Y, fp3c.ModuleSize, fp3c.FoundCount)

	ms := fp0c.ModuleSize
	mkWin := func(cx, cy, x0m, x1m, ym float64) image.Rectangle {
		return image.Rect(int(cx+x0m*ms), int(cy-ym*ms), int(cx+x1m*ms), int(cy+ym*ms))
	}
	// Controls: the found left corners.
	scanWindow(t, d, mkWin(fp0c.Center.X, fp0c.Center.Y, -8, 8, 8), "ctrl-TL")
	scanWindow(t, d, mkWin(fp3c.Center.X, fp3c.Center.Y, -8, 8, 8), "ctrl-BL")
	// Targets: the right corners, wide enough for perspective compression
	// (true side 85 modules, centre distance 78 modules at the LEFT module
	// size; the right side is compressed, so start well short of that).
	trWin := mkWin(fp0c.Center.X, fp0c.Center.Y, 40, 95, 10)
	brWin := mkWin(fp3c.Center.X, fp3c.Center.Y, 40, 95, 10)
	scanWindow(t, d, trWin, "target-TR")
	scanWindow(t, d, brWin, "target-BR")
	scanDumpWindow(t, bm, d.Ch, trWin, "8c_side_L2_rot45_TR")
	scanDumpWindow(t, bm, d.Ch, brWin, "8c_side_L2_rot45_BR")
	scanDumpWindow(t, bm, d.Ch, mkWin(fp0c.Center.X, fp0c.Center.Y, -8, 8, 8), "8c_side_L2_rot45_TL_ctrl")
	// Zooms on the TRUE right corners, placed from the wide-window crops (the
	// code's edges tilt, so the corner rows sit off the anchor rows).
	trZoom := image.Rect(1600, 630, 1790, 790)
	brZoom := image.Rect(1760, 1510, 1950, 1670)
	scanWindow(t, d, trZoom, "zoom-trueTR")
	scanWindow(t, d, brZoom, "zoom-trueBR")
	scanDumpWindow(t, bm, d.Ch, trZoom, "8c_side_L2_rot45_trueTR_zoom")
	scanDumpWindow(t, bm, d.Ch, brZoom, "8c_side_L2_rot45_trueBR_zoom")

	// Existence check: complete the quad from the strong left corners with
	// synthesized right corners around the visually located true positions,
	// keep completions measuring the true 85x85 grid, decode them all. On
	// plain-sample failure, replay production's escape hatches: Part I on the
	// sample (attributing the second blocker), the alignment-pattern
	// resample, and the seeded route (the same quad mapped onto the base
	// level's rotation canvas - uniform scaling about the canvas centre,
	// since both canvases share the rotation).
	baseLvl := scanPyramidLevel(t, img, 3)
	bm3 := RotateToBitmap(baseLvl, 45)
	BalanceRGB(bm3)
	sSeed := float64(bm3.Width) / float64(bm.Width)
	cc2x, cc2y := float64(bm.Width)/2, float64(bm.Height)/2
	cc3x, cc3y := float64(bm3.Width)/2, float64(bm3.Height)/2
	tried, decoded := 0, 0
	partIOK, partIFail := 0, 0
	apTried, apDecoded := 0, 0
	seedDecoded := 0
	const apCap = 150
	for _, msR := range []float64{7, 8.5, 10} {
		for dy1 := -24.0; dy1 <= 24; dy1 += 12 {
			for dx1 := -24.0; dx1 <= 24; dx1 += 12 {
				for dy2 := -24.0; dy2 <= 24; dy2 += 12 {
					for dx2 := -24.0; dx2 <= 24; dx2 += 12 {
						fp1s := FinderPattern{Typ: 1, Center: core.PointF{X: 1690 + dx1, Y: 700 + dy1}, ModuleSize: msR, FoundCount: 3}
						fp2s := FinderPattern{Typ: 2, Center: core.PointF{X: 1840 + dx2, Y: 1590 + dy2}, ModuleSize: msR, FoundCount: 3}
						quad := []FinderPattern{fp0c, fp1s, fp2s, fp3c}
						side := CalculateSideSize(bm, quad)
						if side.X != 85 || side.Y != 85 {
							continue
						}
						tried++
						pt := core.PerspectiveTransform(quad[0].Center, quad[1].Center, quad[2].Center, quad[3].Center, side)
						matrix := SampleSymbol(bm, pt, side)
						if matrix == nil {
							continue
						}
						sym := &core.DecodedSymbol{}
						if decode.DecodePrimary(matrix, sym) == core.Success {
							decoded++
							payload := decode.DecodeData(sym.Data)
							t.Logf("display existence: DECODED, TR off (%+.0f,%+.0f) BR off (%+.0f,%+.0f) ms %.1f: %q",
								dx1, dy1, dx2, dy2, msR, string(payload[:min(48, len(payload))]))
							continue
						}
						// Part I attribution on the failed sample.
						symI := &core.DecodedSymbol{SideSize: side}
						dm := make([]byte, side.X*side.Y)
						x, y, count := spec.PrimaryMetadataX, spec.PrimaryMetadataY, 0
						if partIRet, _ := decode.DecodePrimaryMetadataPartI(matrix, symI, dm, &count, &x, &y); partIRet == core.Success && symI.Meta.NC == 2 {
							partIOK++
						} else {
							partIFail++
						}
						// AP fallback, as detectPrimary would run it.
						sv := sym.Meta.SideVersion
						if apTried < apCap && sv.X >= 1 && sv.X <= 32 && sv.Y >= 1 && sv.Y <= 32 {
							apTried++
							sym.SideSize = image.Pt(spec.VersionToSize(sv.X), spec.VersionToSize(sv.Y))
							for i := range 4 {
								sym.PatternPositions[i] = quad[i].Center
							}
							if apM := SampleSymbolByAlignmentPattern(bm, d.Ch, sym, quad); apM != nil {
								if decode.DecodePrimary(apM, sym) == core.Success {
									apDecoded++
									t.Logf("display existence: AP DECODED, TR off (%+.0f,%+.0f) BR off (%+.0f,%+.0f) ms %.1f",
										dx1, dy1, dx2, dy2, msR)
								}
							}
						}
						// Seeded route: same quad on the base-level canvas.
						var fps3 [4]FinderPattern
						for i := range 4 {
							fps3[i] = FinderPattern{
								Typ: i,
								Center: core.PointF{
									X: sSeed*(quad[i].Center.X-cc2x) + cc3x,
									Y: sSeed*(quad[i].Center.Y-cc2y) + cc3y,
								},
								ModuleSize: quad[i].ModuleSize * sSeed,
								FoundCount: 1,
							}
						}
						pt3 := core.PerspectiveTransform(fps3[0].Center, fps3[1].Center, fps3[2].Center, fps3[3].Center, side)
						if m3 := SampleSymbol(bm3, pt3, side); m3 != nil {
							sym3 := &core.DecodedSymbol{}
							if decode.DecodePrimary(m3, sym3) == core.Success {
								seedDecoded++
								payload := decode.DecodeData(sym3.Data)
								t.Logf("display existence: SEEDED DECODED, TR off (%+.0f,%+.0f) BR off (%+.0f,%+.0f) ms %.1f: %q",
									dx1, dy1, dx2, dy2, msR, string(payload[:min(48, len(payload))]))
							}
						}
					}
				}
			}
		}
	}
	t.Logf("display existence: %d true-grid quads; plain %d, seeded %d, AP %d/%d decoded; Part I ok/fail %d/%d",
		tried, decoded, seedDecoded, apDecoded, apTried, partIOK, partIFail)
}

// TestPrintFinderScanExperiment attributes the print 16c frontal rot45 roi0
// crop's type-0 starvation. Step one: per-pass pool inspection across the
// full LocateFinders retry ladder, with candidate positions, to learn which
// binarization pass produces the three found corners and where the missing
// type-0 corner should sit. Temporary diagnostic, not part of any gate.
func TestPrintFinderScanExperiment(t *testing.T) {
	img := scanLoadImage(t, "print_camera/16c_frontal_rot45.webp")
	rois := ProposeROIs(img, 2)
	if len(rois) == 0 {
		t.Fatal("no ROIs")
	}
	crop := CropImage(img, rois[0].Bounds)
	bm := RotateToBitmap(crop, 135)
	BalanceRGB(bm)
	ch := BinarizerRGB(bm, nil)
	d := &PrimaryDetector{BM: bm, Ch: ch, Mode: IntensiveDetect}
	found := d.LocateFinders()
	t.Logf("crop %v rot135 (%dx%d): LocateFinders=%v printDetected=%v",
		rois[0].Bounds, bm.Width, bm.Height, found, d.PrintDetected())
	t.Logf("consensus work: geometry tuples=%d scores=%d interpolated triples=%d seeks=%d",
		d.Stats.Consensus.GeometryTuples, d.Stats.Consensus.GeometryScores,
		d.Stats.Consensus.InterpolatedTriples, d.Stats.Consensus.InterpolatedSeeks)
	for pi := range d.Stats.Passes {
		p := &d.Stats.Passes[pi]
		var g [4]int
		for _, c := range p.Candidates {
			if c.Typ >= 0 && c.Typ < 4 {
				g[c.Typ]++
			}
		}
		t.Logf("pass %d %-20q status %2d: rawHits %5d branchB/R %d/%d cands %d/%d/%d/%d",
			pi, p.Label, p.Status, p.RawHits, p.BranchBlue, p.BranchRed, g[0], g[1], g[2], g[3])
		if len(p.Candidates) <= 8 {
			for _, c := range p.Candidates {
				t.Logf("    type %d at (%.0f,%.0f) ms %.1f found %d", c.Typ, c.Center.X, c.Center.Y, c.ModuleSize, c.FoundCount)
			}
		}
	}
	if len(d.Stats.Passes) < 2 {
		t.Fatal("expected the avg-RGB retry pass")
	}
	// The avg-RGB pass carries the three true corners (high found counts at
	// ms 18-28); instrument the missing type-0 corner on that pass's channels.
	// In any finder quad types 0/2 and 1/3 are the diagonals, so the missing
	// corner is the parallelogram completion regardless of rotation.
	cands := d.Stats.Passes[1].Candidates
	fp1c, ok1 := bestOfType(cands, 1)
	fp2c, ok2 := bestOfType(cands, 2)
	fp3c, ok3 := bestOfType(cands, 3)
	if !ok1 || !ok2 || !ok3 {
		t.Fatal("anchor finders missing in the avg-RGB pass")
	}
	estX := fp1c.Center.X + fp3c.Center.X - fp2c.Center.X
	estY := fp1c.Center.Y + fp3c.Center.Y - fp2c.Center.Y
	ms := (fp1c.ModuleSize + fp2c.ModuleSize + fp3c.ModuleSize) / 3
	t.Logf("estimated type-0 corner (%.0f,%.0f), anchor ms %.1f", estX, estY, ms)
	avg := d.Stats.RGBAvg
	ch2 := BinarizerRGB(bm, avg[:])
	d2 := &PrimaryDetector{BM: bm, Ch: ch2, Mode: IntensiveDetect}
	ctrl := image.Rect(int(fp3c.Center.X-8*ms), int(fp3c.Center.Y-8*ms), int(fp3c.Center.X+8*ms), int(fp3c.Center.Y+8*ms))
	win := image.Rect(int(estX-10*ms), int(estY-10*ms), int(estX+10*ms), int(estY+10*ms))
	scanWindow(t, d2, ctrl, "print-ctrl-type3")
	scanWindow(t, d2, win, "print-target-type0")
	scanDumpWindow(t, bm, ch2, ctrl, "print16c_roi0_rot135_type3_ctrl")
	scanDumpWindow(t, bm, ch2, win, "print16c_roi0_rot135_type0")
	// The parallelogram estimate lands above the perspective trapezoid's real
	// corner; the type-0-classified seeds at (3057,1260) mark the finder top.
	// Zoom below them to cover the actual finder.
	zoom := image.Rect(2920, 1170, 3250, 1450)
	scanWindow(t, d2, zoom, "print-zoom-type0")
	scanDumpWindow(t, bm, ch2, zoom, "print16c_roi0_rot135_type0_zoom")

	// Existence check for the near-miss recovery: complete the quad with a
	// type-0 candidate synthesized from the measured near-miss band (its
	// median row and x, its module size) and decode from that quad directly.
	fp0c := FinderPattern{
		Typ:        0,
		Center:     core.PointF{X: 3057, Y: 1262},
		ModuleSize: 20.3,
		FoundCount: 6,
	}
	quad := []FinderPattern{fp0c, fp1c, fp2c, fp3c}
	side := CalculateSideSize(bm, quad)
	t.Logf("existence check: quad side %v", side)
	if side.X > 0 && side.Y > 0 {
		pt := core.PerspectiveTransform(quad[0].Center, quad[1].Center, quad[2].Center, quad[3].Center, side)
		samples := map[string]*core.Bitmap{
			"plain":  SampleSymbol(bm, pt, side),
			"offset": SampleSymbolOffset(bm, pt, side, SearchChannelOffsets(bm, pt, side)),
		}
		for name, matrix := range samples {
			if matrix == nil {
				t.Logf("existence check (%s): sample failed", name)
				continue
			}
			sym := &core.DecodedSymbol{}
			res := decode.DecodePrimary(matrix, sym)
			t.Logf("existence check (%s): DecodePrimary => %d, %d data bits", name, res, len(sym.Data))
			if res == core.Success {
				payload := decode.DecodeData(sym.Data)
				t.Logf("existence check (%s): payload %d bytes, head %q", name, len(payload), string(payload[:min(60, len(payload))]))
				continue
			}
			// Where does it die? Replay the metadata ladder on the sample,
			// forcing the known Nc when Part I misreads (non-default layout),
			// to see whether anything BEYOND Part I could decode.
			sym2 := &core.DecodedSymbol{SideSize: side}
			dataMap := make([]byte, side.X*side.Y)
			x, y, count := spec.PrimaryMetadataX, spec.PrimaryMetadataY, 0
			ret, _ := decode.DecodePrimaryMetadataPartI(matrix, sym2, dataMap, &count, &x, &y)
			t.Logf("existence check (%s): Part I => %d, Nc=%d (true 3)", name, ret, sym2.Meta.NC)
			if ret != core.Success || sym2.Meta.NC != 3 {
				sym2.Meta.NC = 3
				x, y, count = spec.PrimaryMetadataX, spec.PrimaryMetadataY, 0
				clear(dataMap)
				for count < spec.PrimaryMetadataPart1ModuleNumber {
					dataMap[y*side.X+x] = 1
					count++
					spec.NextMetadataModuleInPrimary(side.Y, side.X, count, &x, &y)
				}
			}
			if decode.ReadColorPaletteInPrimary(matrix, sym2, dataMap, &count, &x, &y) != core.Success {
				t.Logf("existence check (%s): forced-Nc palette walk failed", name)
				continue
			}
			copies := spec.PaletteCopies(16)
			normPalette := make([]float64, 16*4*copies)
			decode.NormalizeColorPalette(sym2, normPalette, 16)
			palThs := make([]float64, 3*spec.ColorPaletteNumber)
			for i := range copies {
				th := decode.PaletteThreshold(sym2.Palette[16*3*i:], 16)
				palThs[i*3+0], palThs[i*3+1], palThs[i*3+2] = th[0], th[1], th[2]
			}
			ret2, _ := decode.DecodePrimaryMetadataPartII(matrix, sym2, dataMap, normPalette, palThs, &count, &x, &y)
			t.Logf("existence check (%s): forced-Nc Part II => %d (ECL %v mask %d version %v)",
				name, ret2, sym2.Meta.ECL, sym2.Meta.MaskType, sym2.Meta.SideVersion)
			if ret2 <= 0 {
				continue
			}
			res2 := decode.DecodeSymbol(matrix, sym2, dataMap, normPalette, palThs, 0)
			t.Logf("existence check (%s): forced-Nc DecodeSymbol => %d", name, res2)
			if res2 == core.Success {
				payload := decode.DecodeData(sym2.Data)
				t.Logf("existence check (%s): forced-Nc payload %d bytes, head %q",
					name, len(payload), string(payload[:min(60, len(payload))]))
			}
		}
	}
}
