package decode

import (
	"fmt"
	"image"
	"io"
	"math"
	"sort"
	"strings"

	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/spec"
)

// Diagnose measures where primary-symbol finder detection dies on img and writes
// a human-readable report to w. It reproduces Decode's pre-finder chain
// (bitmapFromImage -> balanceRGB -> binarizerRGB) then runs the finder search
// (raw, avg-RGB retry, and descreen passes) through locateFinders, dumping the
// per-pass counters and the retry's flat thresholds; it then replays the
// post-finder chain (side-size -> transform -> sample -> metadata/palette ->
// LDPC, plus the alignment-pattern fallback) so a failure can be attributed to a
// stage. When imageDir is non-empty, each stage additionally writes an annotated
// image there (region boxes, finder candidates and quad, warped sampling grid,
// upscaled sampled matrix, palette swatches), numbered in report order. It is a
// debugging aid for the detector and never influences decoding.
func Diagnose(img image.Image, w io.Writer, imageDir string) {
	sink := newDiagImageSink(imageDir, w)
	bm := bitmapFromImage(img)
	balanceRGB(bm)
	ch := binarizerRGB(bm, nil)
	sink.save("balanced", diagBitmapImage(bm))
	sink.saveBinarized("binarized", ch)

	ppx, ppy := estimatePitch(bm)
	diagLogf(w, "estimatePitch (px,py) = (%d,%d)", ppx, ppy)

	d := &primaryDetector{bm: bm, ch: ch, mode: intensiveDetect}
	ok := d.locateFinders()

	diagLogf(w, "image %dx%d  locateFinders=%v  passes=%d", bm.width, bm.height, ok, len(d.stats.passes))
	if len(d.stats.passes) == 0 {
		diagLogf(w, "no finder pass recorded")
		return
	}
	for i, p := range d.stats.passes {
		logFinderPass(w, passLabel(i), p)
	}
	a := d.stats.rgbAvg
	diagLogf(w, "rgbAvg (avg-RGB retry flat black thresholds, per channel) = [%.2f %.2f %.2f]", a[0], a[1], a[2])
	if d.ch != ch {
		// A retry re-binarized; the detector's final channels differ from the
		// raw pass saved above.
		sink.saveBinarized("binarized_final", d.ch)
	}

	rois := diagROIProposals(w, img)
	sink.saveROIs(img, rois)
	diagROIOrientationProbe(w, sink, img, rois)

	diagFindQuad(w, d.stats.passes[0].candidates)
	diagQuadPaletteScan(w, bm, d.stats.passes[0].candidates)
	var quad []finderPattern
	if ok {
		quad = d.fps
	}
	sink.withPrefix("upright_").saveFinders(bm, d.stats.passes[len(d.stats.passes)-1].candidates, quad)
	if ok {
		func() {
			defer func() {
				if r := recover(); r != nil {
					diagLogf(w, "downstream: panicked (decoder not robust to this geometry): %v", r)
				}
			}()
			diagDownstream(w, sink.withPrefix("upright_"), d)
		}()
	}

	if data, err := Decode(img); err != nil {
		diagLogf(w, "Decode: FAILED: %v", err)
	} else {
		diagLogf(w, "Decode: OK (%d bytes): %q", len(data), string(data))
	}
}

// diagLogf writes one newline-terminated report line to w.
func diagLogf(w io.Writer, format string, args ...any) {
	fmt.Fprintf(w, format+"\n", args...)
}

// diagROIOrientationProbe compares the coarse orientation probe on the full frame
// against the probe run on each proposed region's crop at its own scale - the
// measurement for whether per-region probing recovers orientation families the
// whole-frame probe's downscale loses (the recall gate). For each region it then
// attempts the full decode at the retained rungs, reporting how far wiring the
// retry into Decode would actually get.
func diagROIOrientationProbe(w io.Writer, sink *diagImageSink, img image.Image, rois []roiCandidate) {
	s := sink.withPrefix("full_")
	rungs := diagProbeReport(w, s, "full frame", img)
	if diagRungDecodes(w, s, "full frame", img, rungs) {
		return
	}
	for i, r := range rois {
		crop := cropImage(img, r.bounds)
		label := fmt.Sprintf("ROI %d", i)
		sr := sink.withPrefix(fmt.Sprintf("roi%d_", i))
		sr.save("crop", crop)
		rungs := diagProbeReport(w, sr, label, crop)
		if diagRungDecodes(w, sr, label, crop, rungs) {
			return
		}
	}
}

// diagRungDecodes attempts the full decode of img pre-rotated to each retained
// rung, replaying the stage chain on every failure, and reports whether one read.
func diagRungDecodes(w io.Writer, sink *diagImageSink, label string, img image.Image, rungs []float64) bool {
	for _, deg := range rungs {
		rot := rotateImage(img, deg)
		data, ok, _ := decodeImage(rot)
		if ok {
			diagLogf(w, "  %s decode at %v deg: OK (%d bytes): %q", label, deg, len(data), string(data))
			return true
		}
		diagLogf(w, "  %s decode at %v deg: failed", label, deg)
		s := sink.withPrefix(fmt.Sprintf("rung%03.0f_", deg))
		s.save("input", rot)
		diagRungReplay(w, s, fmt.Sprintf("  rung %v", deg), rot)
	}
	return false
}

// diagRungReplay re-runs the finder chain on one rotated region crop and replays
// the downstream stages, attributing where a retained orientation rung's full
// decode dies - the per-region successor of the whole-frame stage replay.
func diagRungReplay(w io.Writer, sink *diagImageSink, prefix string, img image.Image) {
	bm := bitmapFromImage(img)
	balanceRGB(bm)
	ch := binarizerRGB(bm, nil)
	d := &primaryDetector{bm: bm, ch: ch, mode: intensiveDetect}
	ok := d.locateFinders()
	if len(d.stats.passes) == 0 {
		diagLogf(w, "%s: no finder pass recorded", prefix)
		return
	}
	p := d.stats.passes[0]
	diagLogf(w, "%s: locateFinders=%v passes=%d pass1 cross FP0=%d FP1=%d FP2=%d FP3=%d missing=%d",
		prefix, ok, len(d.stats.passes),
		p.crossSurvivors[0], p.crossSurvivors[1], p.crossSurvivors[2], p.crossSurvivors[3], p.missing)
	var quad []finderPattern
	if ok {
		quad = d.fps
	}
	sink.saveBinarized("binarized", d.ch)
	sink.saveFinders(bm, d.stats.passes[len(d.stats.passes)-1].candidates, quad)
	if !ok {
		return
	}
	defer func() {
		if r := recover(); r != nil {
			diagLogf(w, "%s: downstream panicked: %v", prefix, r)
		}
	}()
	diagDownstream(w, sink, d)
}

// diagProbeReport dumps the per-angle probe evidence and the retained rungs for
// one image, returning the rungs.
func diagProbeReport(w io.Writer, sink *diagImageSink, label string, img image.Image) []float64 {
	sink.save("probe_input", downscaleToMax(img, coarseMaxDim))
	fams := coarseProbeFamilies(img)
	var b strings.Builder
	for _, f := range fams {
		fmt.Fprintf(&b, "  %v:types=%d,sum=%d", f.deg, f.types, f.sum)
	}
	diagLogf(w, "orientation probe [%s]:%s", label, b.String())
	rungs := familiesToRungs(fams)
	diagLogf(w, "  retained rungs: %v", rungs)
	return rungs
}

// passLabel names a finder pass by its position in locateFinders' sequence: the
// raw pass, the avg-RGB retry, then one descreen pass per descreen-schedule entry.
func passLabel(i int) string {
	switch {
	case i == 0:
		return "1 raw"
	case i == 1:
		return "2 avg-RGB retry"
	default:
		return fmt.Sprintf("%d descreen", i+1)
	}
}

// logFinderPass prints one finder-detection pass's counters. crossSurvivors,
// preprune and selected are tallied by finder type FP0/FP1/FP2/FP3; remember a
// black-core type (FP0/FP1) is decided by which channel-pair branch fired, not
// by colour, so an absent black type may be a real absence or a mis-bucketing.
func logFinderPass(w io.Writer, label string, p finderPassStats) {
	diagLogf(w, "pass %s:", label)
	diagLogf(w, "  rawHits (n-1-1-1-m, horiz+conditional vert) = %d", p.rawHits)
	diagLogf(w, "  branch routing: blue(->FP0/FP3)=%d  red(->FP1/FP2)=%d", p.branchBlue, p.branchRed)
	diagLogf(w, "  red path: colorOK(fp2found)=%d  classified(fp1/fp2)=%d", p.redColor, p.redClassified)
	diagLogf(w, "  crossCheckPattern survivors  = FP0=%d FP1=%d FP2=%d FP3=%d",
		p.crossSurvivors[0], p.crossSurvivors[1], p.crossSurvivors[2], p.crossSurvivors[3])
	diagLogf(w, "  pre-prune groups (fc>=3)     = FP0=%d FP1=%d FP2=%d FP3=%d",
		p.preprune[0], p.preprune[1], p.preprune[2], p.preprune[3])
	diagLogf(w, "  selected foundCount (post-prune) = FP0=%d FP1=%d FP2=%d FP3=%d",
		p.selected[0], p.selected[1], p.selected[2], p.selected[3])
	diagLogf(w, "  missing=%d  status=%s  interpolated=%v", p.missing, statusName(p.status), p.interpolated)
	for _, c := range p.candidates {
		diagLogf(w, "    cand typ=%d center=(%.0f,%.0f) foundCount=%d moduleSize=%.1f", c.typ, c.center.x, c.center.y, c.foundCount, c.moduleSize)
	}
}

func statusName(s int) string {
	switch s {
	case jabSuccess:
		return "jabSuccess"
	case jabFailure:
		return "jabFailure"
	case fatalError:
		return "fatalError"
	default:
		return fmt.Sprintf("status(%d)", s)
	}
}

// diagDownstream replays detectPrimary's post-finder chain (side-size -> transform
// -> sampleSymbol -> decodePrimary, then the alignment-pattern fallback) by calling
// the real functions, logging which stage stops on the capture. It mirrors
// detectPrimary so a post-finder failure can be attributed to a stage; the bool
// detectPrimary returns hides that.
func diagDownstream(w io.Writer, sink *diagImageSink, d *primaryDetector) {
	bm, ch, fps := d.bm, d.ch, d.fps
	for i := range 4 {
		diagLogf(w, "downstream: selected fp%d typ=%d center=(%.1f,%.1f) foundCount=%d moduleSize=%.2f",
			i, fps[i].typ, fps[i].center.x, fps[i].center.y, fps[i].foundCount, fps[i].moduleSize)
	}
	// Per-edge module-count breakdown, mirroring calculateSideSize (layout FP0 FP1 / FP3 FP2).
	diagEdge(w, "topX  FP0->FP1", fps[0], fps[1])
	diagEdge(w, "botX  FP3->FP2", fps[3], fps[2])
	diagEdge(w, "leftY FP0->FP3", fps[0], fps[3])
	diagEdge(w, "rghtY FP1->FP2", fps[1], fps[2])

	sideSize := calculateSideSize(fps)
	diagLogf(w, "downstream: sideSize=(%d,%d)", sideSize.X, sideSize.Y)
	if sideSize.X == -1 || sideSize.Y == -1 {
		diagLogf(w, "downstream: per-type side-size invalid; trying geometric quad retry")
		if quad, ok := d.selectFinderQuadByGeometry(); ok {
			for i := range 4 {
				diagLogf(w, "downstream: geom fp%d typ=%d center=(%.1f,%.1f) fc=%d ms=%.2f",
					i, quad[i].typ, quad[i].center.x, quad[i].center.y, quad[i].foundCount, quad[i].moduleSize)
			}
			copy(fps, quad[:])
			sideSize = calculateSideSize(fps)
			diagLogf(w, "downstream: after geometric retry sideSize=(%d,%d)", sideSize.X, sideSize.Y)
		} else {
			diagLogf(w, "downstream: geometric quad retry found no valid quad")
		}
		if sideSize.X == -1 || sideSize.Y == -1 {
			diagLogf(w, "downstream: STAGE side-size FAILED")
			return
		}
	}

	pt := getPerspectiveTransform(fps[0].center, fps[1].center, fps[2].center, fps[3].center, sideSize)
	matrix := sampleSymbol(bm, pt, sideSize)
	if matrix == nil {
		diagLogf(w, "downstream: STAGE sample FAILED (nil matrix)")
		return
	}
	diagLogf(w, "downstream: sampled matrix %dx%d", matrix.width, matrix.height)
	sink.saveGrid(bm, pt, sideSize)
	sink.saveMatrix("matrix", matrix)
	diagModulePlacement(w, bm, pt, sideSize)

	var symbol decodedSymbol
	symbol.index = 0
	symbol.hostIndex = 0
	symbol.sideSize = sideSize
	symbol.moduleSize = (fps[0].moduleSize + fps[1].moduleSize + fps[2].moduleSize + fps[3].moduleSize) / 4.0
	for i := range 4 {
		symbol.patternPositions[i] = fps[i].center
	}

	res := diagDecodePrimary(w, matrix, &symbol)
	sink.savePalette("palette", &symbol)
	sink.saveMatrixClassified("matrix_classified", matrix, &symbol)
	diagLogf(w, "downstream: decodePrimary (finder sample) => %s", statusName(res))
	if res == jabSuccess {
		diagLogf(w, "downstream: PRIMARY DECODED, %d data bits, dockedPosition=%04b", len(symbol.data), symbol.meta.dockedPosition)
		diagLogf(w, "downstream: decodeData => %q", string(decodeData(symbol.data)))
		return
	}
	if res < 0 {
		diagLogf(w, "downstream: fatal in decodePrimary; no AP fallback")
		return
	}

	// Alignment-pattern resample fallback, exactly as detectPrimary does.
	symbol.sideSize = image.Pt(spec.VersionToSize(symbol.meta.sideVersion.X), spec.VersionToSize(symbol.meta.sideVersion.Y))
	apMatrix := sampleSymbolByAlignmentPattern(bm, ch, &symbol, fps)
	if apMatrix == nil {
		diagLogf(w, "downstream: STAGE AP-resample FAILED (nil matrix)")
		return
	}
	diagLogf(w, "downstream: AP matrix %dx%d", apMatrix.width, apMatrix.height)
	sink.saveMatrix("matrix_ap", apMatrix)
	res2 := diagDecodePrimary(w, apMatrix, &symbol)
	sink.savePalette("palette_ap", &symbol)
	sink.saveMatrixClassified("matrix_ap_classified", apMatrix, &symbol)
	diagLogf(w, "downstream: decodePrimary (AP sample) => %s", statusName(res2))
}

// diagModulePlacement separates sampling misplacement from colour cast using the
// modules whose colours are known a priori: each finder's 5-module cross-sections
// alternate its two type colours, and the finder centres are the exact anchors of
// the perspective transform. For each such module it samples the centre the way
// sampleSymbol does, plus the four quadrant points offset a quarter module in
// module space, and reports the quadrant spread (max channel range). A uniform
// footprint (small spread) whose colour still deviates from the expected one is a
// cast/classification problem; a large spread means the point straddles module
// boundaries - misplacement. The +-1/+-2 offsets sit away from the anchors, so
// error growing with offset indicates a pitch/scale mismatch rather than a wrong
// anchor.
func diagModulePlacement(w io.Writer, bm *bitmap, pt perspective, side image.Point) {
	names := []string{"blk", "blu", "grn", "cyn", "red", "mag", "yel", "wht"}
	// Core and ring colour indices per finder (8-colour mode): layers alternate
	// the finder's two type colours outward from the core.
	type fp struct {
		label      string
		mx, my     int
		core, ring int
	}
	fps := []fp{
		{"FP0", 3, 3, 0, 3},
		{"FP1", side.X - 4, 3, 0, 6},
		{"FP2", side.X - 4, side.Y - 4, 6, 0},
		{"FP3", 3, side.Y - 4, 3, 0},
	}
	diagLogf(w, "module placement (finder cross-sections; quadSpread high = straddling, low+wrong colour = cast):")
	axes := [2]struct {
		label  byte
		dx, dy int
	}{{'x', 1, 0}, {'y', 0, 1}}
	for _, f := range fps {
		for _, a := range axes {
			for off := -2; off <= 2; off++ {
				if a.dy != 0 && off == 0 {
					continue // centre already printed on the x axis
				}
				mx := f.mx + off*a.dx
				my := f.my + off*a.dy
				if mx < 0 || my < 0 || mx >= side.X || my >= side.Y {
					continue
				}
				exp := f.core
				if off == -1 || off == 1 {
					exp = f.ring
				}
				c := pointF{float64(mx) + 0.5, float64(my) + 0.5}
				ctr, okC := diagSampleAt(bm, pt.warp(c))
				if !okC {
					diagLogf(w, "  %s %c%+d module=(%d,%d) exp=%s OUT OF IMAGE", f.label, a.label, off, mx, my, names[exp])
					continue
				}
				var lo, hi [3]float64
				for i := range 3 {
					lo[i], hi[i] = 255, 0
				}
				quads := 0
				for _, q := range [4]pointF{
					{c.x - 0.25, c.y - 0.25}, {c.x + 0.25, c.y - 0.25},
					{c.x - 0.25, c.y + 0.25}, {c.x + 0.25, c.y + 0.25},
				} {
					s, ok := diagSampleAt(bm, pt.warp(q))
					if !ok {
						continue
					}
					quads++
					for i := range 3 {
						lo[i] = math.Min(lo[i], s[i])
						hi[i] = math.Max(hi[i], s[i])
					}
				}
				spread := 0.0
				if quads == 4 {
					for i := range 3 {
						spread = math.Max(spread, hi[i]-lo[i])
					}
				} else {
					spread = math.NaN()
				}
				p := pt.warp(c)
				diagLogf(w, "  %s %c%+d module=(%d,%d) exp=%s pt=(%.0f,%.0f) rgb=(%3.0f,%3.0f,%3.0f) quadSpread=%.0f",
					f.label, a.label, off, mx, my, names[exp], p.x, p.y, ctr[0], ctr[1], ctr[2], spread)
			}
		}
	}
}

// diagSampleAt returns the 3x3-average RGB at image point p, or ok=false when p
// falls outside the image. It is a deliberately narrow point probe for
// sub-module positions (the placement dump samples quarter-module offsets),
// unlike sampleSymbol's whole-module footprint mean.
func diagSampleAt(bm *bitmap, p pointF) (rgb [3]float64, ok bool) {
	mx, my := int(p.x), int(p.y)
	if mx < 0 || my < 0 || mx >= bm.width || my >= bm.height {
		return rgb, false
	}
	bpp := bm.channels
	row := bm.width * bpp
	for c := range 3 {
		sum := 0.0
		for dy := -1; dy <= 1; dy++ {
			for dx := -1; dx <= 1; dx++ {
				px, py := mx+dx, my+dy
				if px < 0 || px >= bm.width {
					px = mx
				}
				if py < 0 || py >= bm.height {
					py = my
				}
				sum += float64(bm.pix[py*row+px*bpp+c])
			}
		}
		rgb[c] = sum / 9
	}
	return rgb, true
}

// diagDecodePrimary replays decodePrimary's body (metadata part I -> palette read ->
// metadata part II -> decodeSymbol/LDPC) on the real functions, logging each stage,
// so a decode failure points at one sub-stage rather than a single status code.
func diagDecodePrimary(w io.Writer, matrix *bitmap, symbol *decodedSymbol) int {
	if matrix == nil {
		return fatalError
	}
	symbol.sideSize = image.Pt(matrix.width, matrix.height)
	dataMap := make([]byte, matrix.width*matrix.height)
	x, y := spec.PrimaryMetadataX, spec.PrimaryMetadataY
	moduleCount := 0

	partIRet := decodePrimaryMetadataPartI(matrix, symbol, dataMap, &moduleCount, &x, &y)
	diagLogf(w, "  metadata part I => %s", metaRetName(partIRet))
	if partIRet == jabFailure {
		return jabFailure
	}
	if partIRet == decodeMetadataFailed {
		x, y = spec.PrimaryMetadataX, spec.PrimaryMetadataY
		moduleCount = 0
		clear(dataMap)
		loadDefaultPrimaryMetadata(matrix, symbol)
		diagLogf(w, "  metadata part I unreadable -> default metadata loaded")
	}

	if readColorPaletteInPrimary(matrix, symbol, dataMap, &moduleCount, &x, &y) < 0 {
		diagLogf(w, "  STAGE palette read FAILED")
		return jabFailure
	}
	diagLogf(w, "  palette read OK (Nc=%d, %d palette bytes)", symbol.meta.Nc, len(symbol.palette))
	diagPalette(w, symbol.palette, 1<<(symbol.meta.Nc+1))
	if partIRet == decodeMetadataFailed {
		diagAltPaletteAlignment(w, matrix, symbol)
	}

	colorNumber := 1 << (symbol.meta.Nc + 1)
	normPalette := make([]float64, colorNumber*4*spec.ColorPaletteNumber)
	normalizeColorPalette(symbol, normPalette, colorNumber)
	palThs := make([]float64, 3*spec.ColorPaletteNumber)
	for i := range spec.ColorPaletteNumber {
		th := getPaletteThreshold(symbol.palette[colorNumber*3*i:], colorNumber)
		palThs[i*3+0], palThs[i*3+1], palThs[i*3+2] = th[0], th[1], th[2]
	}

	if partIRet == jabSuccess {
		if decodePrimaryMetadataPartII(matrix, symbol, dataMap, normPalette, palThs, &moduleCount, &x, &y) <= 0 {
			diagLogf(w, "  STAGE metadata part II FAILED")
			return jabFailure
		}
		diagLogf(w, "  metadata part II OK")
	}

	res := decodeSymbol(matrix, symbol, dataMap, normPalette, palThs, 0)
	diagLogf(w, "  decodeSymbol (demask/deinterleave/LDPC) => %s", statusName(res))
	return res
}

// diagAltPaletteAlignment tests the walk-misalignment hypothesis after Part I
// falls back to defaults. A default-encoded symbol places the palette at walk
// position 0, a non-default one places 4 Part I modules first, so a non-default
// symbol read under the default assumption has every walk-read palette slot
// shifted by one round. This dumps the four would-be Part I modules (with their
// decodeModuleNc classification, to show why Part I failed) and re-reads the
// palette at the Part-I-consumed alignment; a coherent palette here means the
// symbol is non-default and the Part I gate is the real blocker.
func diagAltPaletteAlignment(w io.Writer, matrix *bitmap, symbol *decodedSymbol) {
	bpp := matrix.channels
	row := matrix.width * bpp
	x, y, count := spec.PrimaryMetadataX, spec.PrimaryMetadataY, 0
	for i := range spec.PrimaryMetadataPart1ModuleNumber {
		off := y*row + x*bpp
		rgb := matrix.pix[off : off+3]
		diagLogf(w, "  altPartI module %d at (%d,%d) rgb=(%3d,%3d,%3d) decodeModuleNc=%d",
			i, x, y, rgb[0], rgb[1], rgb[2], decodeModuleNc(rgb))
		count++
		spec.NextMetadataModuleInPrimary(matrix.height, matrix.width, count, &x, &y)
	}
	scratch := decodedSymbol{meta: symbol.meta, sideSize: symbol.sideSize}
	dm := make([]byte, matrix.width*matrix.height)
	if readColorPaletteInPrimary(matrix, &scratch, dm, &count, &x, &y) < 0 {
		diagLogf(w, "  alt palette read FAILED")
		return
	}
	diagLogf(w, "  palette re-read at non-default alignment (Part I consumed):")
	diagPalette(w, scratch.palette, 1<<(scratch.meta.Nc+1))
	diagLogf(w, "  alt paletteMinDist=%.1f", paletteMinDist(scratch.palette, 1<<(scratch.meta.Nc+1)))
}

// diagPalette dumps the four corner palettes the decoder read from the sampled
// matrix against the canonical 8-colour palette, and reports each corner's mean
// absolute error to canonical plus the cross-corner spread per colour. Consistent,
// uniformly-shifted palettes mean geometry is right and the residual is a colour
// cast (a calibration problem); garbage or mutually-inconsistent palettes mean the
// geometry or sampling is wrong.
func diagPalette(w io.Writer, pal []byte, colorNumber int) {
	if colorNumber != 8 || len(pal) < 8*3*4 {
		diagLogf(w, "  palette dump skipped (colorNumber=%d len=%d)", colorNumber, len(pal))
		return
	}
	names := []string{"blk", "blu", "grn", "cyn", "red", "mag", "yel", "wht"}
	for corner := range 4 {
		base := corner * colorNumber * 3
		var sumErr float64
		var b strings.Builder
		for c := range 8 {
			r := pal[base+c*3+0]
			g := pal[base+c*3+1]
			bl := pal[base+c*3+2]
			fmt.Fprintf(&b, " %s(%3d,%3d,%3d)", names[c], r, g, bl)
			sumErr += math.Abs(float64(r)-float64(palette.Default[c*3+0])) +
				math.Abs(float64(g)-float64(palette.Default[c*3+1])) +
				math.Abs(float64(bl)-float64(palette.Default[c*3+2]))
		}
		diagLogf(w, "  palette corner %d (meanAbsErr=%.0f):%s", corner, sumErr/(8*3), b.String())
	}
	// Cross-corner spread: max-min of each channel of each colour across the 4 corners.
	var spread float64
	for c := range 8 {
		for ch := range 3 {
			lo, hi := 255.0, 0.0
			for corner := range 4 {
				v := float64(pal[corner*colorNumber*3+c*3+ch])
				lo, hi = math.Min(lo, v), math.Max(hi, v)
			}
			spread += hi - lo
		}
	}
	diagLogf(w, "  palette mean cross-corner spread = %.1f", spread/(8*3))
}

// diagQuadPaletteScan enumerates every geometrically-valid candidate quad, samples
// the symbol it implies, reads its palette, and scores the palette by its minimum
// pairwise colour distance (corner 0). A high best score means some quad samples a
// real, distinct-colour palette, so the true symbol is present and the problem is
// quad selection; a uniformly low score means no candidate quad lands on the symbol
// (a recall problem). Diagnostic only.
func diagQuadPaletteScan(w io.Writer, bm *bitmap, cands []finderPattern) {
	var g [4][]finderPattern
	for _, c := range cands {
		if c.typ >= 0 && c.typ < 4 {
			g[c.typ] = append(g[c.typ], c)
		}
	}
	type scored struct {
		dist float64
		side image.Point
		c    [4]pointF
	}
	var best []scored
	for _, p0 := range g[0] {
		for _, p1 := range g[1] {
			for _, p2 := range g[2] {
				for _, p3 := range g[3] {
					if _, ok := scoreFinderQuad(p0, p1, p2, p3); !ok {
						continue
					}
					if dist, ss, ok := diagSampleQuadPalette(bm, p0, p1, p2, p3); ok {
						best = append(best, scored{dist, ss,
							[4]pointF{p0.center, p1.center, p2.center, p3.center}})
					}
				}
			}
		}
	}
	sort.Slice(best, func(i, j int) bool { return best[i].dist > best[j].dist })
	diagLogf(w, "quad palette scan: %d valid quads sampled; best palette min-colour-distances:", len(best))
	for i, q := range best {
		if i >= 8 {
			break
		}
		diagLogf(w, "  paletteMinDist=%.1f side=(%d,%d) TL=(%.0f,%.0f) TR=(%.0f,%.0f) BR=(%.0f,%.0f) BL=(%.0f,%.0f)",
			q.dist, q.side.X, q.side.Y, q.c[0].x, q.c[0].y, q.c[1].x, q.c[1].y, q.c[2].x, q.c[2].y, q.c[3].x, q.c[3].y)
	}
}

// diagSampleQuadPalette samples the symbol implied by a quad, reads its palette, and
// returns the palette's min pairwise colour distance. It recovers from panics: a
// wrong quad can drive the metadata/palette readers to out-of-range module positions
// (a latent decoder-robustness issue), and such quads are simply skipped here.
func diagSampleQuadPalette(bm *bitmap, p0, p1, p2, p3 finderPattern) (dist float64, side image.Point, ok bool) {
	defer func() {
		if recover() != nil {
			ok = false
		}
	}()
	side = calculateSideSize([]finderPattern{p0, p1, p2, p3})
	pt := getPerspectiveTransform(p0.center, p1.center, p2.center, p3.center, side)
	matrix := sampleSymbol(bm, pt, side)
	if matrix == nil {
		return 0, side, false
	}
	var sym decodedSymbol
	sym.sideSize = image.Pt(matrix.width, matrix.height)
	dataMap := make([]byte, matrix.width*matrix.height)
	x, y, mc := spec.PrimaryMetadataX, spec.PrimaryMetadataY, 0
	if decodePrimaryMetadataPartI(matrix, &sym, dataMap, &mc, &x, &y) == jabFailure {
		return 0, side, false
	}
	x, y, mc = spec.PrimaryMetadataX, spec.PrimaryMetadataY, 0
	clear(dataMap)
	loadDefaultPrimaryMetadata(matrix, &sym)
	if readColorPaletteInPrimary(matrix, &sym, dataMap, &mc, &x, &y) < 0 {
		return 0, side, false
	}
	return paletteMinDist(sym.palette, 1<<(sym.meta.Nc+1)), side, true
}

// paletteMinDist returns the minimum pairwise Euclidean distance among the eight
// colours of the first corner palette: ~0 means the colours are indistinguishable
// (a misaligned sample), large means a real, separable palette.
func paletteMinDist(pal []byte, colorNumber int) float64 {
	if colorNumber != 8 || len(pal) < 8*3 {
		return 0
	}
	mind := math.Inf(1)
	for a := range 8 {
		for b := a + 1; b < 8; b++ {
			dr := float64(pal[a*3+0]) - float64(pal[b*3+0])
			dg := float64(pal[a*3+1]) - float64(pal[b*3+1])
			db := float64(pal[a*3+2]) - float64(pal[b*3+2])
			if d := math.Sqrt(dr*dr + dg*dg + db*db); d < mind {
				mind = d
			}
		}
	}
	return mind
}

// diagFindQuad searches the full pre-prune candidate set (all foundCounts) for any
// four finders - one per type, laid out FP0(TL) FP1(TR) / FP3(BL) FP2(BR) - that form
// a convex, roughly-rectangular quad with consistent module size and a valid side
// size. It answers whether the true symbol quad is present among the candidates (a
// selection problem) or absent (a recall problem), and prototypes a geometric
// selection scorer.
func diagFindQuad(w io.Writer, cands []finderPattern) {
	var g [4][]finderPattern
	for _, c := range cands {
		if c.typ >= 0 && c.typ < 4 {
			g[c.typ] = append(g[c.typ], c)
		}
	}
	diagLogf(w, "quad search: candidate counts by type (all foundCounts) FP0=%d FP1=%d FP2=%d FP3=%d",
		len(g[0]), len(g[1]), len(g[2]), len(g[3]))

	type quad struct {
		side     image.Point
		msSpread float64
		edgeDev  float64 // how far opposite edges differ (0 = perfect)
		selfOK   bool    // geometry matches measured module size
		c        [4]pointF
	}
	var found []quad
	tried, validSide, selfConsistent := 0, 0, 0
	for _, p0 := range g[0] {
		for _, p1 := range g[1] {
			for _, p2 := range g[2] {
				for _, p3 := range g[3] {
					tried++
					fps := []finderPattern{p0, p1, p2, p3}
					if !diagConvexQuad(p0.center, p1.center, p2.center, p3.center) {
						continue
					}
					top := math.Hypot(p0.center.x-p1.center.x, p0.center.y-p1.center.y)
					bot := math.Hypot(p3.center.x-p2.center.x, p3.center.y-p2.center.y)
					left := math.Hypot(p0.center.x-p3.center.x, p0.center.y-p3.center.y)
					right := math.Hypot(p1.center.x-p2.center.x, p1.center.y-p2.center.y)
					edgeDev := math.Max(math.Max(top, bot)/math.Min(top, bot), math.Max(left, right)/math.Min(left, right))
					if edgeDev > 1.3 {
						continue
					}
					msMin, msMax := p0.moduleSize, p0.moduleSize
					msSum := 0.0
					for _, p := range fps {
						msMin = math.Min(msMin, p.moduleSize)
						msMax = math.Max(msMax, p.moduleSize)
						msSum += p.moduleSize
					}
					if msMax/msMin > 1.6 {
						continue
					}
					ss := calculateSideSize(fps)
					if ss.X <= 0 || ss.Y <= 0 {
						continue
					}
					validSide++
					// Self-consistency: edge length / side-modules must match the measured
					// module size, or the quad's geometry and its finders disagree.
					ms := msSum / 4
					okX := diagRatio(top/float64(ss.X), ms) < 1.35 && diagRatio(bot/float64(ss.X), ms) < 1.35
					okY := diagRatio(left/float64(ss.Y), ms) < 1.35 && diagRatio(right/float64(ss.Y), ms) < 1.35
					if okX && okY {
						selfConsistent++
					}
					found = append(found, quad{ss, msMax / msMin, edgeDev, okX && okY,
						[4]pointF{p0.center, p1.center, p2.center, p3.center}})
				}
			}
		}
	}
	diagLogf(w, "quad search: tried %d type-correct combinations; convex+rect+side-valid = %d; module-self-consistent = %d",
		tried, validSide, selfConsistent)
	// Self-consistent quads first (the plausible true symbol), then by edge deviation.
	sort.Slice(found, func(i, j int) bool {
		if found[i].selfOK != found[j].selfOK {
			return found[i].selfOK
		}
		return found[i].edgeDev < found[j].edgeDev
	})
	for i, q := range found {
		if i >= 8 {
			break
		}
		diagLogf(w, "  quad selfOK=%v side=(%d,%d) msSpread=%.2f edgeDev=%.2f TL=(%.0f,%.0f) TR=(%.0f,%.0f) BR=(%.0f,%.0f) BL=(%.0f,%.0f)",
			q.selfOK, q.side.X, q.side.Y, q.msSpread, q.edgeDev,
			q.c[0].x, q.c[0].y, q.c[1].x, q.c[1].y, q.c[2].x, q.c[2].y, q.c[3].x, q.c[3].y)
	}
}

// diagRatio returns the larger/smaller ratio of two positive values (1 = equal).
func diagRatio(a, b float64) float64 {
	if a <= 0 || b <= 0 {
		return math.Inf(1)
	}
	return math.Max(a, b) / math.Min(a, b)
}

// diagConvexQuad reports whether p0,p1,p2,p3 (TL,TR,BR,BL order) form a convex,
// non-self-intersecting quad: all consecutive edge cross-products share one sign.
func diagConvexQuad(p0, p1, p2, p3 pointF) bool {
	pts := [4]pointF{p0, p1, p2, p3}
	var sign float64
	for i := range 4 {
		a, b, c := pts[i], pts[(i+1)&3], pts[(i+2)&3]
		cross := (b.x-a.x)*(c.y-b.y) - (b.y-a.y)*(c.x-b.x)
		if cross == 0 {
			return false
		}
		if i == 0 {
			sign = cross
		} else if (cross > 0) != (sign > 0) {
			return false
		}
	}
	return true
}

// diagEdge logs one finder-pair edge the way calculateSideSize reads it: raw module
// count, the +7 finder allowance, and getSideSize's rounded size and reliability.
func diagEdge(w io.Writer, label string, a, b finderPattern) {
	n := calculateModuleNumber(a, b)
	size, flag := getSideSize(n + 7)
	dist := math.Hypot(a.center.x-b.center.x, a.center.y-b.center.y)
	diagLogf(w, "downstream: %s dist=%.1f moduleN=%d size(n+7)=%d flag=%d", label, dist, n, size, flag)
}

// metaRetName names a metadata-stage return, distinguishing the decodeMetadataFailed
// sentinel (which triggers the default-metadata fallback) from a hard jabFailure.
func metaRetName(r int) string {
	if r == decodeMetadataFailed {
		return "decodeMetadataFailed (-> defaults)"
	}
	return statusName(r)
}
