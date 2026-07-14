//go:build jabcode_bsi

package read

import (
	"bufio"
	"image"
	"image/color"
	"os"
	"strings"
	"testing"

	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/testutil"
	"github.com/srlehn/jabcode/internal/wire"
)

func TestBSIProfileReadsAnnexC(t *testing.T) {
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
}
