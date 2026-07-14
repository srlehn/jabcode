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
