//go:build jabharness

package decode

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"testing"

	_ "golang.org/x/image/webp"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/testutil"
)

// berLoadImage decodes a capture fixture (png/jpeg/webp).
func berLoadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

// berLocateAndSample runs the standard pre-decode chain on img and samples at
// the given side size from the located finder quad, mirroring detectPrimary's
// sampling branch. Returns nil when finders or the sample fail.
func berLocateAndSample(img image.Image, side image.Point) (*core.Bitmap, image.Point) {
	bm := core.BitmapFromImage(img)
	detect.BalanceRGB(bm)
	ch := detect.BinarizerRGB(bm, nil)
	d := &detect.PrimaryDetector{BM: bm, Ch: ch, Mode: detect.IntensiveDetect}
	if !d.LocateFinders() {
		return nil, image.Point{}
	}
	est := detect.CalculateSideSize(bm, d.FPs)
	if side.X <= 0 || side.Y <= 0 {
		side = est
	}
	pt := core.PerspectiveTransform(d.FPs[0].Center, d.FPs[1].Center, d.FPs[2].Center, d.FPs[3].Center, side)
	var matrix *core.Bitmap
	if d.PrintDetected() {
		matrix = detect.SampleSymbolOffset(bm, pt, side, detect.SearchChannelOffsets(bm, pt, side))
	} else {
		matrix = detect.SampleSymbol(bm, pt, side)
	}
	return matrix, est
}

// berSourcePayload decodes a pixel-exact source fixture through the local
// pipeline (locate, sample at the estimated side, DecodePrimary) and returns
// the net payload bytes for re-encoding.
func berSourcePayload(t *testing.T, dir, rel string) []byte {
	img, err := berLoadImage(filepath.Join(dir, filepath.FromSlash(rel)))
	if err != nil {
		t.Skipf("load %s: %v", rel, err)
	}
	matrix, _ := berLocateAndSample(img, image.Point{})
	if matrix == nil {
		t.Fatalf("%s: source did not locate/sample", rel)
	}
	sym := &core.DecodedSymbol{}
	if DecodePrimary(matrix, sym) != core.Success {
		t.Fatalf("%s: source DecodePrimary failed", rel)
	}
	return DecodeData(sym.Data)
}

// berTruth is one colour count's re-encoded ground truth plus the decode-side
// layout derived from its pixel-exact matrix bitmap.
type berTruth struct {
	colors  int
	side    image.Point
	matrix  []byte // masked module colour indices, row-major
	meta    core.Metadata
	dataMap []byte // 1 = reserved, complete (metadata, palette, finder, alignment)
}

// berGroundTruth re-encodes payload and derives mask/ECC metadata and the
// complete dataMap by replaying the metadata walk on the true matrix bitmap.
func berGroundTruth(t *testing.T, colors int, payload []byte, wantSide image.Point) berTruth {
	r, err := encode.Render(encode.Config{Colors: colors, ModuleSize: 5, ECCLevel: 10, SymbolNumber: 1}, payload)
	if err != nil {
		t.Fatalf("re-encode %dc: %v", colors, err)
	}
	if r.SideSize != wantSide {
		t.Fatalf("re-encode %dc side %v, want %v", colors, r.SideSize, wantSide)
	}
	// True-matrix bitmap: one pixel per module, canonical palette colours.
	pal := palette.SetDefault(colors)
	bm := core.NewBitmap(r.SideSize.X, r.SideSize.Y, 4)
	for i, idx := range r.Matrix {
		copy(bm.Pix[i*4:], pal[int(idx)*3:int(idx)*3+3])
		bm.Pix[i*4+3] = 255
	}
	sym := &core.DecodedSymbol{SideSize: r.SideSize}
	dataMap := make([]byte, r.SideSize.X*r.SideSize.Y)
	x, y, count := spec.PrimaryMetadataX, spec.PrimaryMetadataY, 0
	if partIRet, _ := DecodePrimaryMetadataPartI(bm, sym, dataMap, &count, &x, &y); partIRet != core.Success {
		t.Fatalf("%dc: Part I failed on the true matrix", colors)
	}
	if ReadColorPaletteInPrimary(bm, sym, dataMap, &count, &x, &y) != core.Success {
		t.Fatalf("%dc: palette read failed on the true matrix", colors)
	}
	colorNumber := 1 << (sym.Meta.NC + 1)
	if colorNumber != colors {
		t.Fatalf("%dc: true matrix reads Nc=%d", colors, sym.Meta.NC)
	}
	copies := spec.PaletteCopies(colorNumber)
	normPalette := make([]float64, colorNumber*4*copies)
	NormalizeColorPalette(sym, normPalette, colorNumber)
	palThs := make([]float64, 3*spec.ColorPaletteNumber)
	for i := range copies {
		th := PaletteThreshold(sym.Palette[colorNumber*3*i:], colorNumber)
		palThs[i*3+0], palThs[i*3+1], palThs[i*3+2] = th[0], th[1], th[2]
	}
	if partIIRet, _ := DecodePrimaryMetadataPartII(bm, sym, dataMap, normPalette, palThs, &count, &x, &y); partIIRet <= 0 {
		t.Fatalf("%dc: Part II failed on the true matrix", colors)
	}
	fillDataMap(dataMap, r.SideSize.X, r.SideSize.Y, 0)
	return berTruth{colors: colors, side: r.SideSize, matrix: r.Matrix, meta: sym.Meta, dataMap: dataMap}
}

// berCapturedPalette reads the embedded palette copies from a captured sample
// with the colour mode forced to the known truth when Part I misreads (all
// fixtures here are known non-default symbols).
func berCapturedPalette(matrix *core.Bitmap, side image.Point, trueNC int) (*core.DecodedSymbol, bool) {
	sym := &core.DecodedSymbol{SideSize: side}
	dataMap := make([]byte, side.X*side.Y)
	x, y, count := spec.PrimaryMetadataX, spec.PrimaryMetadataY, 0
	ret, _ := DecodePrimaryMetadataPartI(matrix, sym, dataMap, &count, &x, &y)
	partIOK := ret == core.Success && sym.Meta.NC == trueNC
	if !partIOK {
		sym.Meta.NC = trueNC
		x, y, count = spec.PrimaryMetadataX, spec.PrimaryMetadataY, 0
		for count < spec.PrimaryMetadataPart1ModuleNumber {
			count++
			spec.NextMetadataModuleInPrimary(side.Y, side.X, count, &x, &y)
		}
	}
	if ReadColorPaletteInPrimary(matrix, sym, dataMap, &count, &x, &y) != core.Success {
		return nil, partIOK
	}
	return sym, partIOK
}

// berFinderRegistration scores the sample against the true matrix over the
// four finder 5x5 neighbourhoods, classified against the captured palette's
// first copy in absolute RGB (the decoder's own view of the symbol).
func berFinderRegistration(matrix *core.Bitmap, truth berTruth, pal []byte) (mismatch, total int) {
	cx := [4]int{spec.DistanceToBorder - 1, truth.side.X - spec.DistanceToBorder, truth.side.X - spec.DistanceToBorder, spec.DistanceToBorder - 1}
	cy := [4]int{spec.DistanceToBorder - 1, spec.DistanceToBorder - 1, truth.side.Y - spec.DistanceToBorder, truth.side.Y - spec.DistanceToBorder}
	bpp := matrix.Channels
	for corner := range 4 {
		for dy := -2; dy <= 2; dy++ {
			for dx := -2; dx <= 2; dx++ {
				x, y := cx[corner]+dx, cy[corner]+dy
				if x < 0 || y < 0 || x >= truth.side.X || y >= truth.side.Y {
					continue
				}
				off := y*matrix.Width*bpp + x*bpp
				best, bi := math.Inf(1), 0
				for i := range truth.colors {
					dr := float64(matrix.Pix[off+0]) - float64(pal[i*3+0])
					dg := float64(matrix.Pix[off+1]) - float64(pal[i*3+1])
					db := float64(matrix.Pix[off+2]) - float64(pal[i*3+2])
					if d := dr*dr + dg*dg + db*db; d < best {
						best, bi = d, i
					}
				}
				total++
				if byte(bi) != truth.matrix[y*truth.side.X+x] {
					mismatch++
				}
			}
		}
	}
	return mismatch, total
}

// berSampleBER classifies every data module of a sampled matrix with the
// decoder's captured-palette path (colour mode forced to the truth when the
// sample's Part I misreads) and returns the module and bit error rates
// against the true matrix, plus whether Part I read correctly. Returns ok
// false when the palette walk fails.
func berSampleBER(matrix *core.Bitmap, truth berTruth) (moduleER, bitER float64, partIOK, ok bool) {
	sym, pOK := berCapturedPalette(matrix, truth.side, truth.meta.NC)
	if sym == nil {
		return 0, 0, pOK, false
	}
	colorNumber := truth.colors
	copies := spec.PaletteCopies(colorNumber)
	normPalette := make([]float64, colorNumber*4*copies)
	NormalizeColorPalette(sym, normPalette, colorNumber)
	palThs := make([]float64, 3*spec.ColorPaletteNumber)
	for i := range copies {
		th := PaletteThreshold(sym.Palette[colorNumber*3*i:], colorNumber)
		palThs[i*3+0], palThs[i*3+1], palThs[i*3+2] = th[0], th[1], th[2]
	}
	dataMap := make([]byte, len(truth.dataMap))
	copy(dataMap, truth.dataMap)
	capRaw := readRawModuleData(matrix, sym, dataMap, normPalette, palThs)
	trueRaw := make([]byte, 0, len(capRaw))
	for j := 0; j < truth.side.X; j++ {
		for i := 0; i < truth.side.Y; i++ {
			if truth.dataMap[i*truth.side.X+j] == 0 {
				trueRaw = append(trueRaw, truth.matrix[i*truth.side.X+j])
			}
		}
	}
	if len(capRaw) != len(trueRaw) || len(capRaw) == 0 {
		return 0, 0, pOK, false
	}
	bitsPerModule := truth.meta.NC + 1
	capBits := rawModuleData2RawData(capRaw, bitsPerModule)
	trueBits := rawModuleData2RawData(trueRaw, bitsPerModule)
	moduleErr, bitErr := 0, 0
	for i := range capRaw {
		if capRaw[i] != trueRaw[i] {
			moduleErr++
		}
	}
	for i := range capBits {
		if capBits[i] != trueBits[i] {
			bitErr++
		}
	}
	return float64(moduleErr) / float64(len(capRaw)), float64(bitErr) / float64(len(capBits)), pOK, true
}

// TestSideViewBERExperiment measures the BEST achievable data-module BER on
// the display 8c side row across quad completions that read the true 85x85
// grid: the strong left-corner finders anchor the quad, synthesized right
// corners sweep the visually located true positions (the same bracket the
// finder-scan existence check used), and every true-grid sample's BER is
// scored against the re-encoded truth. The minimum against the ~7% LDPC
// ceiling decides whether the side view is geometry-recoverable at all or
// colour-bounded like the print residue. Temporary diagnostic, not part of
// any gate.
func TestSideViewBERExperiment(t *testing.T) {
	dir := testutil.TestdataPath("highcolor_capture")
	payload := berSourcePayload(t, dir, "source/8c_ecc10_v17_lorem_ms4.png")
	truth := berGroundTruth(t, 8, payload, image.Pt(85, 85))
	img, err := berLoadImage(filepath.Join(dir, filepath.FromSlash("display_camera/8c_side_rot45_normal.webp")))
	if err != nil {
		t.Skipf("load: %v", err)
	}
	// The read pyramid's level 2 (halve while the shorter side stays >= 600,
	// coarsest first), rotated by the row's rung.
	base := core.BitmapFromImage(img)
	for i := 3; i < len(base.Pix); i += 4 {
		base.Pix[i] = 255
	}
	levels := []*image.NRGBA{base.NRGBA()}
	for {
		last := levels[len(levels)-1]
		if min(last.Rect.Dx(), last.Rect.Dy()) < 600 {
			break
		}
		levels = append(levels, detect.HalveNRGBA(last))
	}
	// levels runs finest-first here; the read pyramid's L2 (coarsest-first
	// index 2 of 4) is the half-resolution level, finest-first index 1.
	lvl := levels[1]
	bm := detect.RotateToBitmap(lvl, 45)
	detect.BalanceRGB(bm)
	ch := detect.BinarizerRGB(bm, nil)
	d := &detect.PrimaryDetector{BM: bm, Ch: ch, Mode: detect.IntensiveDetect}
	if !d.LocateFinders() {
		t.Fatal("no finders")
	}
	fp0c, fp3c := d.FPs[0], d.FPs[3]
	t.Logf("left anchors: type0 (%.0f,%.0f) ms %.1f; type3 (%.0f,%.0f) ms %.1f",
		fp0c.Center.X, fp0c.Center.Y, fp0c.ModuleSize, fp3c.Center.X, fp3c.Center.Y, fp3c.ModuleSize)
	tried, scored := 0, 0
	bestBit, bestModule := 1.0, 1.0
	var bestDesc string
	partIOKCount := 0
	for _, msR := range []float64{7, 8.5, 10} {
		for dy1 := -24.0; dy1 <= 24; dy1 += 12 {
			for dx1 := -24.0; dx1 <= 24; dx1 += 12 {
				for dy2 := -24.0; dy2 <= 24; dy2 += 12 {
					for dx2 := -24.0; dx2 <= 24; dx2 += 12 {
						fp1s := detect.FinderPattern{Typ: 1, Center: core.PointF{X: 1690 + dx1, Y: 700 + dy1}, ModuleSize: msR, FoundCount: 3}
						fp2s := detect.FinderPattern{Typ: 2, Center: core.PointF{X: 1840 + dx2, Y: 1590 + dy2}, ModuleSize: msR, FoundCount: 3}
						quad := []detect.FinderPattern{fp0c, fp1s, fp2s, fp3c}
						side := detect.CalculateSideSize(bm, quad)
						if side != truth.side {
							continue
						}
						tried++
						pt := core.PerspectiveTransform(quad[0].Center, quad[1].Center, quad[2].Center, quad[3].Center, side)
						matrix := detect.SampleSymbol(bm, pt, side)
						if matrix == nil {
							continue
						}
						mER, bER, pOK, ok := berSampleBER(matrix, truth)
						if !ok {
							continue
						}
						scored++
						if pOK {
							partIOKCount++
						}
						if bER < bestBit {
							bestBit, bestModule = bER, mER
							bestDesc = fmt.Sprintf("TR off (%+.0f,%+.0f) BR off (%+.0f,%+.0f) ms %.1f partI=%v", dx1, dy1, dx2, dy2, msR, pOK)
						}
					}
				}
			}
		}
	}
	t.Logf("side-view BER: %d true-grid quads, %d scored, %d Part I ok", tried, scored, partIOKCount)
	t.Logf("side-view BER: BEST bit %.4f (module %.4f) at %s; hard+soft ceiling ~0.07", bestBit, bestModule, bestDesc)
}

// the 32c+ frontal capture rows on verified-geometry samples: re-encode the
// known payload for the true matrix, mask and layout; force the true side
// onto the located quad; verify finder registration against the captured
// palette; classify every data module with the decoder's own captured-palette
// path; and compare module indices and bits against the truth. The mask XORs
// out identically on both sides, so masked and demasked comparisons are the
// same; the demask call is kept to match the decode path. Sources are the
// zero-error controls; decoding rows calibrate the correctable range.
// Temporary diagnostic, not part of any gate.
func TestDataModuleBERExperiment(t *testing.T) {
	dir := testutil.TestdataPath("highcolor_capture")
	sources := map[int]string{
		32:  "source/32c_ecc10_v13_lorem_ms5.png",
		64:  "source/64c_ecc10_v11_lorem_ms5.png",
		128: "source/128c_ecc10_v10_lorem_ms6.png",
		256: "source/256c_ecc10_v9_lorem_ms6.png",
	}
	wantSides := map[int]image.Point{32: {X: 69, Y: 69}, 64: {X: 61, Y: 61}, 128: {X: 57, Y: 57}, 256: {X: 53, Y: 53}}
	truths := map[int]berTruth{}
	for _, colors := range []int{32, 64, 128, 256} {
		payload := berSourcePayload(t, dir, sources[colors])
		truths[colors] = berGroundTruth(t, colors, payload, wantSides[colors])
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
		{"display_camera/64c_frontal_rot00_redshift.webp", 64},
		{"print_camera/64c_frontal_rot00.webp", 64},
		{"print_scanner/64c_scan_600dpi_pdf.webp", 64},
		{"source/128c_ecc10_v10_lorem_ms6.png", 128},
		{"display_camera/128c_frontal_rot00_normal.webp", 128},
		{"print_camera/128c_frontal_rot00.webp", 128},
		{"print_scanner/128c_scan_600dpi_pdf.webp", 128},
		{"source/256c_ecc10_v9_lorem_ms6.png", 256},
		{"display_camera/256c_frontal_rot00_normal.webp", 256},
		{"print_scanner/256c_scan_600dpi_pdf.webp", 256},
	}
	for _, c := range cases {
		img, err := berLoadImage(filepath.Join(dir, filepath.FromSlash(c.rel)))
		if err != nil {
			t.Skipf("load %s: %v", c.rel, err)
		}
		truth := truths[c.colors]
		matrix, est := berLocateAndSample(img, truth.side)
		if matrix == nil {
			t.Logf("%s: no finders or sample failed", c.rel)
			continue
		}
		sym, partIOK := berCapturedPalette(matrix, truth.side, truth.meta.NC)
		if sym == nil {
			t.Logf("%s: captured palette walk failed", c.rel)
			continue
		}
		mism, tot := berFinderRegistration(matrix, truth, sym.Palette[:c.colors*3])
		t.Logf("%s: est side %v (true %v), partI ok=%v, finder registration %d/%d mismatched",
			c.rel, est, truth.side, partIOK, mism, tot)

		colorNumber := c.colors
		copies := spec.PaletteCopies(colorNumber)
		normPalette := make([]float64, colorNumber*4*copies)
		NormalizeColorPalette(sym, normPalette, colorNumber)
		palThs := make([]float64, 3*spec.ColorPaletteNumber)
		for i := range copies {
			th := PaletteThreshold(sym.Palette[colorNumber*3*i:], colorNumber)
			palThs[i*3+0], palThs[i*3+1], palThs[i*3+2] = th[0], th[1], th[2]
		}
		dataMap := make([]byte, len(truth.dataMap))
		copy(dataMap, truth.dataMap)
		capRaw := readRawModuleData(matrix, sym, dataMap, normPalette, palThs)

		trueRaw := make([]byte, 0, len(capRaw))
		for j := 0; j < truth.side.X; j++ {
			for i := 0; i < truth.side.Y; i++ {
				if truth.dataMap[i*truth.side.X+j] == 0 {
					trueRaw = append(trueRaw, truth.matrix[i*truth.side.X+j])
				}
			}
		}
		if len(capRaw) != len(trueRaw) {
			t.Errorf("%s: raw length %d vs truth %d", c.rel, len(capRaw), len(trueRaw))
			continue
		}
		demaskSymbol(capRaw, dataMap, truth.side, truth.meta.MaskType, colorNumber)
		demaskSymbol(trueRaw, dataMap, truth.side, truth.meta.MaskType, colorNumber)

		bitsPerModule := truth.meta.NC + 1
		capBits := rawModuleData2RawData(capRaw, bitsPerModule)
		trueBits := rawModuleData2RawData(trueRaw, bitsPerModule)
		moduleErr, bitErr := 0, 0
		for i := range capRaw {
			if capRaw[i] != trueRaw[i] {
				moduleErr++
			}
		}
		for i := range capBits {
			if capBits[i] != trueBits[i] {
				bitErr++
			}
		}
		wc, wr := truth.meta.ECL.X, truth.meta.ECL.Y
		t.Logf("%s: %d data modules, module error %.4f (%d), bit error %.4f (%d of %d), ecc wc/wr %d/%d",
			c.rel, len(capRaw),
			float64(moduleErr)/float64(len(capRaw)), moduleErr,
			float64(bitErr)/float64(len(capBits)), bitErr, len(capBits), wc, wr)
	}
}
