//go:build jabcode_bsi

package read

import (
	"bufio"
	"image"
	"image/color"
	"os"
	"strings"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/testutil"
	"github.com/srlehn/jabcode/internal/wire"
)

func TestBSICapabilityReadsAnnexC(t *testing.T) {
	const (
		side       = 21
		moduleSize = 12
	)
	f, err := os.Open(testutil.TestdataPath("bsi_tr_03137_annex_c.golden.txt"))
	if err != nil {
		t.Fatalf("open Annex C golden: %v", err)
	}
	defer f.Close()

	rgb := palette.SetDefaultVariant(8, wire.BSI)
	colors := make(color.Palette, 8)
	for i := range colors {
		colors[i] = color.NRGBA{R: rgb[i*3], G: rgb[i*3+1], B: rgb[i*3+2], A: 255}
	}
	img := image.NewPaletted(image.Rect(0, 0, side*moduleSize, side*moduleSize), colors)
	scanner := bufio.NewScanner(f)
	y := 0
	for scanner.Scan() {
		row := strings.TrimSpace(scanner.Text())
		if row == "" {
			continue
		}
		if y >= side || len(row) != side {
			t.Fatalf("malformed Annex C row %d: %q", y, row)
		}
		for x := range side {
			colorIndex := row[x] - '0'
			for py := y * moduleSize; py < (y+1)*moduleSize; py++ {
				for px := x * moduleSize; px < (x+1)*moduleSize; px++ {
					img.SetColorIndex(px, py, colorIndex)
				}
			}
		}
		y++
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan Annex C golden: %v", err)
	}
	if y != side {
		t.Fatalf("Annex C rows = %d, want %d", y, side)
	}

	bm := core.BitmapFromImage(img)
	detect.BalanceRGB(bm)
	detector := detect.PrimaryDetector{
		BM: bm, Ch: detect.BinarizerRGB(bm, nil), Mode: detect.IntensiveDetect,
	}
	wanted := detect.FinderFamilyCurrent.Mask() | detect.FinderFamilyBSI.Mask()
	if found := detector.LocateFinderFamilies(wanted); !found.Has(detect.FinderFamilyBSI) {
		t.Fatalf("integrated finder families = %#x, want BSI", found)
	}
	if len(detector.Stats.Passes) != 1 {
		t.Fatalf("finder passes = %d, want one shared raw pass", len(detector.Stats.Passes))
	}
	pass := detector.Stats.Passes[0]
	bsi, ok := pass.BSIFamilyStats()
	if !ok || bsi.Status != core.Success {
		t.Fatalf("integrated pass bsiAttempted=%v bsiStatus=%d", ok, bsi.Status)
	}

	got, err := DecodeOnly(img, wire.BSI)
	if err != nil {
		t.Fatal(err)
	}
	if want := "JAB Code 2016!"; string(got) != want {
		t.Fatalf("DecodeOnly = %q, want %q", got, want)
	}
	auto, err := Decode(img)
	if err != nil {
		t.Fatalf("additive Decode: %v", err)
	}
	if want := "JAB Code 2016!"; string(auto) != want {
		t.Fatalf("additive Decode = %q, want %q", auto, want)
	}

	frame := testNRGBA(img)
	var finding finding
	located, stage, _ := decodeBitmapFindingTracedOnly(core.BitmapFromImage(frame), func() bool { return false }, &finding, nil, wire.BSI)
	if want := "JAB Code 2016!"; stage != readDecoded || string(located) != want {
		t.Fatalf("located BSI decode = %q stage=%d, want %q", located, stage, want)
	}
	if finding.family != detect.FinderFamilyBSI {
		t.Fatalf("finding family = %d, want BSI", finding.family)
	}
	seeded, _, ok := decodeSeededTracedOnly([]*image.NRGBA{frame, frame}, finding, func() bool { return false }, nil, wire.BSI)
	if want := "JAB Code 2016!"; !ok || string(seeded) != want {
		t.Fatalf("seeded BSI decode = %q ok=%v, want %q", seeded, ok, want)
	}
}

func TestBSITR03137IndependentFixtures(t *testing.T) {
	tests := []struct {
		fixture string
		want    string
		docked  bool
	}{
		{fixture: "bsi_tr_03137_8c_rect_3x2.png", want: "BSI fixed 3x2 oracle"},
		{fixture: "bsi_tr_03137_8c_rect_5x2.png", want: "BSI fixed 5x2 oracle"},
		{fixture: "bsi_tr_03137_8c_docked_same_3x2.png", want: "BSI fixed two symbol same-size oracle", docked: true},
		{fixture: "bsi_tr_03137_8c_docked_custom_3x2_5x2.png", want: "BSI fixed two symbol custom-side oracle", docked: true},
	}
	for _, tc := range tests {
		t.Run(tc.fixture, func(t *testing.T) {
			img := loadLegacyCReferenceFixture(t, tc.fixture)
			got, err := DecodeOnly(img, wire.BSI)
			if err != nil {
				if tc.docked {
					t.Fatalf("DecodeOnly: %v (%s)", err, bsiSecondaryStageSummary(img))
				}
				t.Fatal(err)
			}
			if string(got) != tc.want {
				t.Fatalf("DecodeOnly = %q, want %q", got, tc.want)
			}
			auto, err := Decode(img)
			if err != nil || string(auto) != tc.want {
				t.Fatalf("additive Decode = %q, %v; want %q", auto, err, tc.want)
			}
			if tc.docked {
				_, trace, err := DecodeWithTraceOnly(img, wire.BSI)
				if err != nil {
					t.Fatal(err)
				}
				assertBSIDockedTrace(t, trace, tc.want)
			}
		})
	}
}

func TestStreamUsesCompiledBSICapability(t *testing.T) {
	for _, tc := range []struct {
		fixture string
		want    string
	}{
		{fixture: "bsi_tr_03137_8c_rect_3x2.png", want: "BSI fixed 3x2 oracle"},
		{fixture: "bsi_tr_03137_8c_docked_custom_3x2_5x2.png", want: "BSI fixed two symbol custom-side oracle"},
	} {
		t.Run(tc.fixture, func(t *testing.T) {
			var stream Stream
			got, frames := requireStreamDecode(t, &stream, loadLegacyCReferenceFixture(t, tc.fixture), 3)
			if string(got) != tc.want {
				t.Fatalf("Stream = %q after %d frames, want %q", got, frames, tc.want)
			}
		})
	}
}

func assertBSIDockedTrace(t *testing.T, trace *DiagnosticTrace, payload string) {
	t.Helper()
	var secondaries []DiagnosticSecondary
	for i := range trace.Attempts {
		if string(trace.Attempts[i].Payload) == payload {
			secondaries = trace.Attempts[i].Secondaries
			break
		}
	}
	if len(secondaries) != 1 {
		t.Fatalf("successful BSI trace secondaries = %d, want 1", len(secondaries))
	}
	secondary := &secondaries[0]
	if secondary.Symbol.WireVariant != wire.BSI || secondary.Result != core.Success {
		t.Fatalf("secondary variant/result = %d/%d, want BSI/success", secondary.Symbol.WireVariant, secondary.Result)
	}
	if secondary.MetadataMatrix == nil || secondary.MetadataMatrix.Width != 5 || secondary.MetadataMatrix.Height != 20 {
		t.Fatalf("secondary metadata matrix = %+v, want 5x20", secondary.MetadataMatrix)
	}
	if secondary.Matrix == nil {
		t.Fatal("secondary trace omitted full sampled matrix")
	}
	moduleCount := secondary.Matrix.Width * secondary.Matrix.Height
	if len(secondary.Classification.DataMap) != moduleCount || len(secondary.Classification.Colors) != moduleCount {
		t.Fatalf("secondary classification map/colors = %d/%d, want %d", len(secondary.Classification.DataMap), len(secondary.Classification.Colors), moduleCount)
	}
}

func bsiSecondaryStageSummary(img image.Image) string {
	bm := core.BitmapFromImage(img)
	detect.BalanceRGB(bm)
	ch := detect.BinarizerRGB(bm, nil)
	d := &detect.PrimaryDetector{BM: bm, Ch: ch, Mode: detect.IntensiveDetect}
	if !d.LocateFinderFamilies(detect.FinderFamilyBSI.Mask()).Has(detect.FinderFamilyBSI) ||
		!d.SelectFinderFamily(detect.FinderFamilyBSI) {
		return "primary finders failed"
	}
	host := core.DecodedSymbol{WireVariant: wire.BSI}
	matrix, stage := sampleLocatedPrimaryTraced(d, detect.FinderFamilyBSI, &host, nil, nil)
	if stage != readSampled {
		return "primary sampling failed"
	}
	if result := decode.DecodeBSIPrimary(matrix, &host); result != core.Success {
		return "primary decode failed"
	}
	dockedPosition := -1
	for position, mask := range [4]int{0x08, 0x04, 0x02, 0x01} {
		if host.Meta.DockedPosition&mask != 0 {
			dockedPosition = position
			break
		}
	}
	if dockedPosition < 0 {
		return "primary metadata has no docked side"
	}
	secondary := core.DecodedSymbol{WireVariant: wire.BSI, Index: 1, HostIndex: 0}
	seed, metadata := detect.PrepareBSISecondary(bm, ch, &host, &secondary, dockedPosition)
	if metadata == nil {
		return "near alignment or metadata sampling failed"
	}
	if result := decode.DecodeBSISecondaryMetadata(metadata, &host, &secondary); result != core.Success {
		return "secondary metadata decode failed"
	}
	secondaryMatrix := detect.FinishBSISecondary(bm, ch, &secondary, seed)
	if secondaryMatrix == nil {
		return "far alignment or full sampling failed"
	}
	if result := decode.DecodeBSISecondary(secondaryMatrix, &secondary); result != core.Success {
		return "secondary payload decode failed"
	}
	return "secondary stages succeeded but assembly failed"
}

func TestBSIDockedTraversalRejectsMissingImage(t *testing.T) {
	symbols := make([]core.DecodedSymbol, maxSymbolNumber)
	symbols[0].WireVariant = wire.BSI
	symbols[0].Meta.DockedPosition = 0x08
	detail := &DiagnosticAttempt{}
	if data, ok := decodeSymbolsTraced(nil, [3]*core.Bitmap{}, symbols, 1, detail); ok || data != nil {
		t.Fatalf("docked BSI traversal = %q, %v; want unavailable", messageTransmission(data), ok)
	}
	if len(detail.Secondaries) != 1 || detail.Secondaries[0].Result != core.Failure ||
		detail.Secondaries[0].Symbol.WireVariant != wire.BSI {
		t.Fatalf("docked BSI trace = %+v, want one explicit BSI failure", detail.Secondaries)
	}
}
