package diag

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/read"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/testutil"
	"github.com/srlehn/jabcode/internal/wire"
)

func TestDiagnoseReturnsDecodedPayload(t *testing.T) {
	payload := []byte("diagnose returns its authoritative payload")
	img, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 12, SymbolNumber: 1}, payload)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	var report bytes.Buffer
	got, err := Diagnose(img, &report, "", "fixture.png", nil)
	if err != nil {
		t.Fatalf("Diagnose: %v\n%s", err, report.String())
	}
	want := append([]byte("]j1"), payload...)
	if !bytes.Equal(got, want) {
		t.Fatalf("Diagnose payload = %q, want %q", got, want)
	}
	if !strings.Contains(report.String(), "Decode: OK") {
		t.Fatalf("diagnostic report omitted final decode result:\n%s", report.String())
	}
}

func TestTraceRenderingCoversEveryProbeAngleAndDecodeStage(t *testing.T) {
	payload := []byte("visualize the authoritative pipeline")
	img, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 12, ECCLevel: 10, SymbolNumber: 1}, payload)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	_, cleanTrace, err := read.DecodeWithTrace(img)
	if err != nil {
		t.Fatalf("clean DecodeWithTrace: %v", err)
	}
	cleanNames := renderedImageNames(t, cleanTrace)
	for _, stage := range []string{
		"_input.png", "_balanced.png", "_pass01_input.png", "_binarized.png",
		"_finders.png", "_grid.png", "_sampled.png",
		"_metadata_part_i_modules.png", "_palette_modules.png",
		"_metadata_part_ii_modules.png", "_payload_layout.png",
		"_palette.png", "_classified.png", "_sampled_vs_classified.png",
	} {
		if !containsImageStage(cleanNames, stage) {
			t.Errorf("clean trace omitted %s; names=%v", stage, cleanNames)
		}
	}

	_, rotatedTrace, err := read.DecodeWithTrace(detect.RotateImage(img, 30))
	if err != nil {
		t.Fatalf("rotated DecodeWithTrace: %v", err)
	}
	rotatedNames := renderedImageNames(t, rotatedTrace)
	for pi, probe := range rotatedTrace.Probes {
		for ai, angle := range probe.Probe.Angles {
			prefix := fmt.Sprintf("probe%02d_angle%02d_%03.0f_", pi+1, ai+1, angle.Family.Deg)
			for _, stage := range []string{"balanced.png", "binarized.png"} {
				if !containsImageStage(rotatedNames, prefix+stage) {
					t.Errorf("probe %d angle %d omitted %s", pi, ai, stage)
				}
			}
			// The overlay is written only for the rungs the probe kept: every
			// rung is measured and most are dropped at once, so a candidate
			// cloud from a dropped one is a full-frame image saying nothing.
			want := slices.Contains(probe.Rungs, angle.Family.Deg)
			if got := containsImageStage(rotatedNames, prefix+"finders.png"); got != want {
				t.Errorf("probe %d angle %.0f: finders overlay written=%v, retained=%v",
					pi, angle.Family.Deg, got, want)
			}
		}
	}
}

func TestTraceRenderingSeparatesFinderFamilies(t *testing.T) {
	bm := core.NewBitmap(96, 96, 4)
	finders := []detect.FinderPattern{
		{Typ: 0, Center: core.Pt(12, 12), ModuleSize: 4, FoundCount: 3},
		{Typ: 1, Center: core.Pt(84, 12), ModuleSize: 4, FoundCount: 3},
		{Typ: 2, Center: core.Pt(84, 84), ModuleSize: 4, FoundCount: 3},
		{Typ: 3, Center: core.Pt(12, 84), ModuleSize: 4, FoundCount: 3},
	}
	trace := &read.DiagnosticTrace{Attempts: []read.DiagnosticAttempt{{
		Balanced: bm,
		Detector: detect.DetectorStats{Passes: []detect.FinderPassStats{{
			Label: "raw",
			FinderFamilyPassStats: detect.FinderFamilyPassStats{
				Status: core.Failure,
			},
		}}},
		DetectorTrace: detect.DetectorTrace{FinderPasses: []detect.FinderPassTrace{{
			Families: detect.FinderFamilyCurrent.Mask() | detect.FinderFamilyBSI.Mask(),
			Finders: [2][]detect.FinderPattern{
				nil, finders,
			},
		}}},
	}}}
	names := renderedImageNames(t, trace)
	for _, stage := range []string{"_pass01_finders.png", "_pass01_bsi_finders.png"} {
		if !containsImageStage(names, stage) {
			t.Errorf("mixed-family trace omitted %s; names=%v", stage, names)
		}
	}

	// The sampled quad belongs to one signature. Drawn on the other family's
	// overlay it claims a quad that signature never produced, and a BSI-only
	// read would show no sampled quad at all. The BSI quad here is the only one
	// on the frame, so the two overlays are told apart by which of them carries
	// the heavy final outline.
	trace.Attempts[0].Finders = finders
	trace.Attempts[0].FindersFamily = detect.FinderFamilyBSI
	images := map[string]image.Image{}
	renderTrace(io.Discard, &diagImageSink{seq: new(int), record: func(name string, img image.Image) {
		images[name] = img
	}}, trace)
	var current, bsi image.Image
	for name, img := range images {
		switch {
		case strings.Contains(name, "_pass01_bsi_finders.png"):
			bsi = img
		case strings.Contains(name, "_pass01_finders.png"):
			current = img
		}
	}
	if current == nil || bsi == nil {
		t.Fatalf("one of the two family overlays is missing: %v", images)
	}
	if countColor(bsi, diagColFinal) == 0 {
		t.Error("the BSI overlay does not draw the quad its own signature sampled")
	}
	if n := countColor(current, diagColFinal); n != 0 {
		t.Errorf("the current-family overlay draws %d pixels of the BSI sampled quad", n)
	}
}

// countColor counts exactly-matching pixels, which is how an overlay's own
// marks are told from the frame it is drawn on.
func countColor(img image.Image, want color.NRGBA) int {
	n := 0
	b := img.Bounds()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			r, g, bl, a := img.At(x, y).RGBA()
			if uint8(r>>8) == want.R && uint8(g>>8) == want.G && uint8(bl>>8) == want.B && uint8(a>>8) == want.A {
				n++
			}
		}
	}
	return n
}

func TestTraceRenderingCoversPyramidROIAndGeometryViews(t *testing.T) {
	payload := []byte("visualize every pyramid level")
	img, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 32, SymbolNumber: 1}, payload)
	if err != nil {
		t.Fatalf("encode pyramid fixture: %v", err)
	}
	_, trace, err := read.DecodeWithTrace(img)
	if err != nil {
		t.Fatalf("pyramid DecodeWithTrace: %v", err)
	}
	if len(trace.PyramidImages) == 0 || len(trace.PyramidImages) != len(trace.Pyramid) {
		t.Fatalf("pyramid images=%d dimensions=%d", len(trace.PyramidImages), len(trace.Pyramid))
	}
	names := renderedImageNames(t, trace)
	for i := range trace.PyramidImages {
		stage := fmt.Sprintf("pyramid_level%02d_input.png", i)
		if !containsImageStage(names, stage) {
			t.Errorf("pyramid trace omitted %s", stage)
		}
	}

	bm := core.NewBitmap(96, 96, 4)
	pt := core.PerspectiveTransform(core.Pt(8, 8), core.Pt(88, 8), core.Pt(88, 88), core.Pt(8, 88), image.Pt(21, 21))
	mapTrace := &read.DiagnosticTrace{
		ROIs: []read.DiagnosticROIs{{
			Image: image.NewNRGBA(image.Rect(0, 0, 96, 96)),
			TileMap: detect.ROITileMap{
				Score: []float64{0.1, 1, 0.4, 0.2}, Chroma: []float64{0.2, 1, 0.5, 0.3},
				Grad: []float64{0.5, 0.8, 1, 0.2}, GX: 2, GY: 2, Tile: 48, W: 96, H: 96,
			},
			Candidates: []detect.ROICandidate{{Bounds: image.Rect(0, 0, 96, 96), Score: 1}},
		}},
		Attempts: []read.DiagnosticAttempt{{
			Route:    read.DiagnosticRoute{Kind: "upright", Level: -1, ROI: -1},
			Balanced: bm, Side: image.Pt(21, 21), Transform: pt, HasTransform: true,
			PrintDetected: true,
			Alignments: []*detect.AlignmentTrace{
				{
					Attempted: true, Grid: image.Pt(1, 1),
					Expected: []detect.FinderPattern{{Center: core.Pt(48, 48), ModuleSize: 4}},
					Patterns: []detect.FinderPattern{{Center: core.Pt(49, 48), ModuleSize: 4, FoundCount: 1}},
					Matrix:   core.NewBitmap(21, 21, 4),
				},
				{
					Attempted: true, Grid: image.Pt(1, 1),
					Expected: []detect.FinderPattern{{Center: core.Pt(44, 44), ModuleSize: 4}},
					Patterns: []detect.FinderPattern{{Center: core.Pt(45, 44), ModuleSize: 4, FoundCount: 1}},
					Matrix:   core.NewBitmap(21, 21, 4),
				},
			},
		}},
	}
	mapNames := renderedImageNames(t, mapTrace)
	for _, stage := range []string{
		"roi_chroma_map.png", "roi_gradient_map.png", "roi_joint_map.png", "_rois.png",
		"_channel_offsets.png",
		"_" + diagImageSuffixAlignment + ".png",
		"_" + diagImageSuffixSampledAP + ".png",
		"_" + diagImageSuffixAlignment + "02.png",
		"_" + diagImageSuffixAlignment + "02_" + diagImageSuffixSampledAP + ".png",
	} {
		if !containsImageStage(mapNames, stage) {
			t.Errorf("synthetic trace omitted %s; names=%v", stage, mapNames)
		}
	}

	// The repeated stage is the one a name-based selection can get wrong: the
	// second alignment is written as "alignment02", and selecting "alignment"
	// has to take it while still excluding the neighbouring sampled_ap.
	var selected []string
	sink := &diagImageSink{
		seq:   new(int),
		types: map[string]bool{diagImageSuffixAlignment: true},
		record: func(name string, _ image.Image) {
			selected = append(selected, name)
		},
	}
	renderTrace(io.Discard, sink, mapTrace)
	for _, stage := range []string{
		"_" + diagImageSuffixAlignment + ".png",
		"_" + diagImageSuffixAlignment + "02.png",
	} {
		if !containsImageStage(selected, stage) {
			t.Errorf("selecting %q omitted %s; names=%v", diagImageSuffixAlignment, stage, selected)
		}
	}
	for _, name := range selected {
		if strings.Contains(name, diagImageSuffixSampledAP) {
			t.Errorf("selecting %q also wrote %s", diagImageSuffixAlignment, name)
		}
	}

	emptyAlignmentNames := renderedImageNames(t, &read.DiagnosticTrace{
		Attempts: []read.DiagnosticAttempt{{
			Route:    read.DiagnosticRoute{Kind: "upright", Level: -1, ROI: -1},
			Balanced: bm,
			Alignments: []*detect.AlignmentTrace{{
				Attempted: true, Reason: "no drawable geometry",
			}},
		}},
	})
	if containsImageStage(emptyAlignmentNames, "_"+diagImageSuffixAlignment+".png") {
		t.Errorf("geometry-free alignment failure emitted a duplicate image: %v", emptyAlignmentNames)
	}
}

func TestTraceRenderingCoversDockedSecondaryGeometry(t *testing.T) {
	if !read.CompiledCapabilities().Has(wire.CurrentC) {
		t.Skip("current-C decoder not compiled")
	}
	f, err := os.Open(testutil.TestdataPath("c_multi.png"))
	if err != nil {
		t.Fatalf("open c_multi.png: %v", err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatalf("decode c_multi.png: %v", err)
	}
	_, trace, err := read.DecodeWithTraceOnly(img, wire.CurrentC)
	if err != nil {
		t.Fatalf("multi DecodeWithTrace: %v", err)
	}
	var secondaries []read.DiagnosticSecondary
	for _, attempt := range trace.Attempts {
		if len(attempt.Secondaries) > 0 {
			secondaries = attempt.Secondaries
			break
		}
	}
	if len(secondaries) == 0 {
		t.Fatal("multi trace omitted docked secondary")
	}
	for i := range secondaries {
		secondary := &secondaries[i]
		if secondary.Symbol.WireVariant != wire.CurrentC {
			t.Fatalf("secondary %d variant = %d, want current-C", i, secondary.Symbol.WireVariant)
		}
		if secondary.Matrix == nil || len(secondary.Classification.DataMap) == 0 ||
			len(secondary.Classification.Colors) == 0 {
			t.Fatalf("secondary %d omitted authoritative classification", i)
		}
	}
	var report bytes.Buffer
	renderTrace(&report, nil, trace)
	if !strings.Contains(report.String(), "secondary 1: variant=current-c") {
		t.Fatalf("secondary report omitted established current-C variant:\n%s", report.String())
	}
	names := renderedImageNames(t, trace)
	for _, stage := range []string{
		"_secondary01_finders.png", "_secondary01_grid.png", "_secondary01_sampled.png",
		"_secondary01_palette.png", "_secondary01_classified.png",
		"_secondary01_sampled_vs_classified.png",
	} {
		if !containsImageStage(names, stage) {
			t.Errorf("multi trace omitted %s; names=%v", stage, names)
		}
	}
}

func TestTraceRenderingCoversBSISecondaryMetadata(t *testing.T) {
	if !read.CompiledCapabilities().Has(wire.BSI) {
		t.Skip("BSI decoder not compiled")
	}
	f, err := os.Open(testutil.TestdataPath("bsi_tr_03137_8c_docked_custom_3x2_5x2.png"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	_, trace, err := read.DecodeWithTraceOnly(img, wire.BSI)
	if err != nil {
		t.Fatal(err)
	}
	names := renderedImageNames(t, trace)
	if stage := "_secondary01_" + diagImageSuffixSecondaryMetadata + ".png"; !containsImageStage(names, stage) {
		t.Fatalf("BSI trace omitted %s; names=%v", stage, names)
	}
}

func renderedImageNames(t *testing.T, trace *read.DiagnosticTrace) []string {
	t.Helper()
	seq := 0
	var names []string
	sink := &diagImageSink{
		seq: &seq, filePrefix: "fixture",
		record: func(name string, img image.Image) {
			if img == nil {
				t.Errorf("rendered %s with nil image", name)
			}
			names = append(names, name)
		},
	}
	renderTrace(io.Discard, sink, trace)
	return names
}

func containsImageStage(names []string, stage string) bool {
	for _, name := range names {
		if strings.Contains(name, stage) {
			return true
		}
	}
	return false
}

// TestDiagImageTypeSelection pins the selector a reader uses to keep one read
// from writing a gigabyte: only the named types are written, an unknown name is
// reported rather than silently narrowing the run to nothing, and the sequence
// numbers still match an unfiltered run so two runs can be compared by name.
func TestDiagImageTypeSelection(t *testing.T) {
	img, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 12, ECCLevel: 10, SymbolNumber: 1}, []byte("select types"))
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	_, trace, err := read.DecodeWithTrace(img)
	if err != nil {
		t.Fatalf("DecodeWithTrace: %v", err)
	}

	names := func(types []string, report io.Writer) []string {
		var got []string
		sink := &diagImageSink{seq: new(int), record: func(name string, _ image.Image) {
			got = append(got, name)
		}}
		if len(types) > 0 {
			sink.types = map[string]bool{}
			for _, ty := range types {
				sink.types[ty] = true
			}
		}
		sink.w = report
		renderTrace(io.Discard, sink, trace)
		return got
	}

	all := names(nil, io.Discard)
	only := names([]string{"finders"}, io.Discard)
	if len(only) == 0 {
		t.Fatal("selecting finders wrote nothing")
	}
	if len(only) >= len(all) {
		t.Errorf("selection wrote %d of %d images, so it is not selecting", len(only), len(all))
	}
	for _, n := range only {
		if !strings.Contains(n, "finders") {
			t.Errorf("selection wrote %q, which is not a finders image", n)
		}
	}
	// Sequence numbers are the filename's leading counter; a filtered run keeps
	// them so a name from one run names the same stage in the other.
	for _, n := range only {
		if !containsImageStage(all, n) {
			t.Errorf("filtered run renamed %q, so runs cannot be compared", n)
		}
	}

	// Repeated stages are numbered, and the flag advertises the unnumbered type.
	indexed := &diagImageSink{seq: new(int), types: map[string]bool{"alignment": true}}
	for _, name := range []string{"alignment", "alignment02", "alignment10"} {
		if indexed.skipStage(name) {
			t.Errorf("selecting alignment skipped %q", name)
		}
	}
	if !indexed.skipStage("sampled_ap") {
		t.Error("selecting alignment kept sampled_ap")
	}

	var report bytes.Buffer
	newDiagImageSink(t.TempDir(), &report, "fixture.png", []string{"finders", "nonsense"})
	if !strings.Contains(report.String(), `unknown type "nonsense"`) {
		t.Errorf("an unknown type was accepted silently:\n%s", report.String())
	}
}

// TestDiagImageSelectionSkipsRendering pins the selector as a rendering gate
// rather than a write gate. Building the canvas is most of the cost of a
// diagnostic run, so filtering only at the point of writing would save disk and
// nothing else.
//
// The probe is a bitmap with dimensions but no pixels: drawing it panics, so
// completing the call proves no canvas was built. The allowed case asserts the
// probe works, otherwise the skipped case proves nothing.
func TestDiagImageSelectionSkipsRendering(t *testing.T) {
	unreadable := &core.Bitmap{Width: 64, Height: 64, Channels: 4}
	overlay := finderOverlay{cands: []detect.FinderPattern{
		{Typ: 0, Center: core.Pt(8, 8), ModuleSize: 2, FoundCount: 3},
	}}
	render := func(types map[string]bool) (rendered bool) {
		sink := &diagImageSink{seq: new(int), types: types, record: func(string, image.Image) {}}
		defer func() { rendered = recover() != nil }()
		sink.saveFinders(unreadable, overlay)
		if *sink.seq != 1 {
			t.Errorf("sequence advanced to %d, want 1", *sink.seq)
		}
		return false
	}
	if render(map[string]bool{"grid": true}) {
		t.Error("a filtered-out finders stage still built its canvas")
	}
	if !render(map[string]bool{"finders": true}) {
		t.Error("the selected stage did not render, so the probe proves nothing")
	}
}

func TestDiagnoseReturnsDecodeFailureAfterEarlyDiagnosticExit(t *testing.T) {
	var report bytes.Buffer
	_, err := Diagnose(image.NewNRGBA(image.Rect(0, 0, 64, 64)), &report, "", "fixture.png", nil)
	if err == nil {
		t.Fatal("Diagnose returned nil error for a blank image")
	}
	if !strings.Contains(report.String(), "Decode: FAILED") {
		t.Fatalf("diagnostic report omitted final decode failure:\n%s", report.String())
	}
}

func TestTraceRenderingCoversDrawableEarlyExit(t *testing.T) {
	_, trace, err := read.DecodeWithTrace(image.NewNRGBA(image.Rect(0, 0, 64, 64)))
	if err == nil {
		t.Fatal("blank image decoded")
	}
	names := renderedImageNames(t, trace)
	for _, stage := range []string{"_input.png", "_balanced.png", "_pass01_input.png", "_binarized.png", "_finders.png"} {
		if !containsImageStage(names, stage) {
			t.Errorf("early-exit trace omitted %s; names=%v", stage, names)
		}
	}
}

func TestDiagHighColorClassificationUsesEveryPaletteCopy(t *testing.T) {
	for _, colors := range []int{128, 256} {
		img, err := encode.Run(encode.Config{Colors: colors, ModuleSize: 1, Format: wire.EncodeISOHighColor, SymbolNumber: 1}, []byte("diag high color"))
		if err != nil {
			t.Fatalf("colors %d encode: %v", colors, err)
		}
		bm := core.BitmapFromImage(img)
		sym := core.DecodedSymbol{WireVariant: wire.ISOHighColor}
		var trace decode.PrimaryTrace
		obs, ret := decode.ObservePrimaryTraced(bm, &sym, &trace)
		if ret != core.Success || obs == nil {
			t.Fatalf("colors %d ObservePrimary = %d", colors, ret)
		}
		if ret := obs.CorrectPayload(); ret != core.Success {
			t.Fatalf("colors %d CorrectPayload = %d", colors, ret)
		}
		wantLen := colors * 3 * spec.PaletteCopies(colors)
		if len(sym.Palette) != wantLen {
			t.Fatalf("colors %d palette len = %d, want %d", colors, len(sym.Palette), wantLen)
		}
		reserved := -1
		for i, classified := range trace.Classification.Colors {
			if classified == 255 {
				reserved = i
				break
			}
		}
		if reserved < 0 {
			t.Fatalf("colors %d classification trace has no reserved module", colors)
		}
		x, y := reserved%bm.Width, reserved/bm.Width
		off := bm.Offset(x, y)
		bm.Pix[off], bm.Pix[off+1], bm.Pix[off+2] = 17, 83, 149
		got := diagMatrixClassified(bm, &sym, &trace.Classification)
		if got == nil {
			t.Fatalf("colors %d classification image is nil", colors)
		}
		scale := got.Bounds().Dx() / bm.Width
		pixel := got.NRGBAAt(x*scale, y*scale)
		if pixel.R == 17 && pixel.G == 83 && pixel.B == 149 {
			t.Fatalf("colors %d reserved module retained its raw colour", colors)
		}
	}
}

func TestDiagSymbolPaletteLayout(t *testing.T) {
	for _, colors := range []int{8, 128} {
		sym := &core.DecodedSymbol{
			Palette: make([]byte, colors*3*spec.PaletteCopies(colors)),
		}
		sym.Meta.NC = spec.Log2Int(colors) - 1
		gotColors, gotCopies, ok := diagSymbolPaletteLayout(sym)
		if !ok {
			t.Fatalf("colors %d layout rejected", colors)
		}
		if gotColors != colors || gotCopies != spec.PaletteCopies(colors) {
			t.Fatalf("colors %d layout = (%d,%d), want (%d,%d)",
				colors, gotColors, gotCopies, colors, spec.PaletteCopies(colors))
		}
		sym.Palette = sym.Palette[:len(sym.Palette)-1]
		if _, _, ok := diagSymbolPaletteLayout(sym); ok {
			t.Fatalf("colors %d truncated palette accepted", colors)
		}
	}
}
