//go:build jabcode_non_iso_encode

package encode

import (
	"bufio"
	"bytes"
	"os"
	"strings"
	"testing"

	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/testutil"
	"github.com/srlehn/jabcode/internal/wire"
)

func TestEncodeBSIAnnexCMatrix(t *testing.T) {
	f, err := os.Open(testutil.TestdataPath("bsi_tr_03137_annex_c.golden.txt"))
	if err != nil {
		t.Fatalf("open Annex C golden: %v", err)
	}
	defer f.Close()

	var want []byte
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		row := strings.TrimSpace(scanner.Text())
		if row == "" {
			continue
		}
		for i := range row {
			want = append(want, row[i]-'0')
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan Annex C golden: %v", err)
	}

	e := &encoder{
		colors: 8, moduleSize: 1, symbolNumber: 1, format: wire.EncodeBSI,
		palette: palette.SetDefaultVariant(8, wire.BSI),
	}
	if err := e.generate([]byte("JAB Code 2016!")); err != nil {
		t.Fatal(err)
	}
	symbol := &e.symbols[0]
	if symbol.sideSize.X != 21 || symbol.sideSize.Y != 21 {
		t.Fatalf("side size = %v, want 21x21", symbol.sideSize)
	}
	if !bytes.Equal(symbol.matrix, want) {
		for i := range want {
			if symbol.matrix[i] != want[i] {
				t.Fatalf("module[%d] (x=%d,y=%d) = %d, want %d", i, i%21, i/21, symbol.matrix[i], want[i])
			}
		}
		t.Fatalf("matrix length = %d, want %d", len(symbol.matrix), len(want))
	}
}
