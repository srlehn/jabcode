//go:build jabcode_bsi

package decode

import (
	"bufio"
	"os"
	"strings"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/testutil"
	"github.com/srlehn/jabcode/internal/wire"
)

func TestDecodeBSIAnnexCPrimary(t *testing.T) {
	f, err := os.Open(testutil.TestdataPath("bsi_tr_03137_annex_c.golden.txt"))
	if err != nil {
		t.Fatalf("open Annex C golden: %v", err)
	}
	defer f.Close()

	const side = 21
	matrix := core.NewBitmap(side, side, 4)
	colors := palette.SetDefaultProfile(8, wire.BSI)
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
			colorIndex := int(row[x] - '0')
			if colorIndex < 0 || colorIndex >= 8 {
				t.Fatalf("invalid color index at (%d,%d): %q", x, y, row[x])
			}
			off := matrix.Offset(x, y)
			copy(matrix.Pix[off:off+3], colors[colorIndex*3:colorIndex*3+3])
			matrix.Pix[off+3] = 255
		}
		y++
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan Annex C golden: %v", err)
	}
	if y != side {
		t.Fatalf("Annex C rows = %d, want %d", y, side)
	}

	var symbol core.DecodedSymbol
	if got := DecodeBSIPrimary(matrix, &symbol); got != core.Success {
		t.Fatalf("DecodeBSIPrimary = %d, want %d", got, core.Success)
	}
	if symbol.Meta.NC != 2 {
		t.Errorf("Nc = %d, want 2", symbol.Meta.NC)
	}
	if symbol.Meta.SideVersion.X != 1 || symbol.Meta.SideVersion.Y != 1 {
		t.Errorf("side version = %v, want (1,1)", symbol.Meta.SideVersion)
	}
	if symbol.Meta.ECL.X != 8 || symbol.Meta.ECL.Y != 9 {
		t.Errorf("ECC weights = %v, want (8,9)", symbol.Meta.ECL)
	}
	if symbol.Meta.DockedPosition != 0 {
		t.Errorf("docked position = %d, want 0", symbol.Meta.DockedPosition)
	}
	got, ok := DecodeDataProfile(symbol.Data, wire.BSI)
	if !ok {
		t.Fatal("DecodeDataProfile rejected the corrected Annex C payload")
	}
	if want := "JAB Code 2016!"; string(got) != want {
		t.Fatalf("payload = %q, want %q", got, want)
	}
}
