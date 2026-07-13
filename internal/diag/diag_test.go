package diag

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/read"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/testutil"
)

func TestDiagnoseReturnsDecodedPayload(t *testing.T) {
	payload := []byte("diagnose returns its authoritative payload")
	img, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 12, SymbolNumber: 1}, payload)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	var report bytes.Buffer
	got, err := Diagnose(img, &report, "", "fixture.png")
	if err != nil {
		t.Fatalf("Diagnose: %v\n%s", err, report.String())
	}
	if !bytes.Equal(got, payload) {
		t.Fatalf("Diagnose payload = %q, want %q", got, payload)
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
			for _, stage := range []string{"balanced.png", "binarized.png", "finders.png"} {
				if !containsImageStage(rotatedNames, prefix+stage) {
					t.Errorf("probe %d angle %d omitted %s", pi, ai, stage)
				}
			}
		}
	}
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
			Alignment: &detect.AlignmentTrace{
				Attempted: true, Grid: image.Pt(1, 1),
				Expected: []detect.FinderPattern{{Center: core.Pt(48, 48), ModuleSize: 4}},
				Patterns: []detect.FinderPattern{{Center: core.Pt(49, 48), ModuleSize: 4, FoundCount: 1}},
			},
		}},
	}
	mapNames := renderedImageNames(t, mapTrace)
	for _, stage := range []string{
		"roi_chroma_map.png", "roi_gradient_map.png", "roi_joint_map.png", "_rois.png",
		"_channel_offsets.png", "_alignment.png",
	} {
		if !containsImageStage(mapNames, stage) {
			t.Errorf("synthetic trace omitted %s; names=%v", stage, mapNames)
		}
	}

	emptyAlignmentNames := renderedImageNames(t, &read.DiagnosticTrace{
		Attempts: []read.DiagnosticAttempt{{
			Route:    read.DiagnosticRoute{Kind: "upright", Level: -1, ROI: -1},
			Balanced: bm,
			Alignment: &detect.AlignmentTrace{
				Attempted: true, Reason: "no drawable geometry",
			},
		}},
	})
	if containsImageStage(emptyAlignmentNames, "_alignment.png") {
		t.Errorf("geometry-free alignment failure emitted a duplicate image: %v", emptyAlignmentNames)
	}
}

func TestTraceRenderingCoversDockedSecondaryGeometry(t *testing.T) {
	f, err := os.Open(testutil.TestdataPath("c_multi.png"))
	if err != nil {
		t.Fatalf("open c_multi.png: %v", err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatalf("decode c_multi.png: %v", err)
	}
	_, trace, err := read.DecodeWithTrace(img)
	if err != nil {
		t.Fatalf("multi DecodeWithTrace: %v", err)
	}
	found := false
	for _, attempt := range trace.Attempts {
		if len(attempt.Secondaries) > 0 {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("multi trace omitted docked secondary")
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

func TestDiagnoseReturnsDecodeFailureAfterEarlyDiagnosticExit(t *testing.T) {
	var report bytes.Buffer
	_, err := Diagnose(image.NewNRGBA(image.Rect(0, 0, 64, 64)), &report, "", "fixture.png")
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
		img, err := encode.Run(encode.Config{Colors: colors, ModuleSize: 1, SymbolNumber: 1}, []byte("diag high color"))
		if err != nil {
			t.Fatalf("colors %d encode: %v", colors, err)
		}
		bm := core.BitmapFromImage(img)
		var sym core.DecodedSymbol
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
