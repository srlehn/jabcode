package encode

import (
	"bufio"
	"bytes"
	"encoding/hex"
	"os"
	"strings"
	"testing"

	"github.com/srlehn/jabcode/internal/palette"
	"github.com/srlehn/jabcode/internal/testutil"
	"github.com/srlehn/jabcode/internal/wire"
)

// TestEncodeMatrixGolden runs the full default-mode encoder and checks that the
// final masked module matrix matches the reference library byte for byte. This
// exercises the whole encode pipeline: data encoding, LDPC, interleaving,
// pattern/palette placement, data layout, and masking.
func TestEncodeMatrixGolden(t *testing.T) {
	f, err := os.Open(testutil.TestdataPath("matrix_golden.txt"))
	if err != nil {
		t.Fatalf("open golden: %v", err)
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) != 4 {
			t.Fatalf("malformed golden line: %q", line)
		}
		input, err := hex.DecodeString(fields[0])
		if err != nil {
			t.Fatalf("decode input hex: %v", err)
		}
		wantW := testutil.MustAtoi(t, fields[1])
		wantH := testutil.MustAtoi(t, fields[2])
		wantMatrix, err := hex.DecodeString(fields[3])
		if err != nil {
			t.Fatalf("decode matrix hex: %v", err)
		}

		e := &encoder{
			colors: 8, moduleSize: 12, symbolNumber: 1, format: wire.EncodeCurrentC,
			palette: palette.SetDefaultVariant(8, wire.CurrentC),
		}
		if err := e.generate(input); err != nil {
			t.Errorf("%q: generate: %v", input, err)
			continue
		}
		s := &e.symbols[0]
		if s.sideSize.X != wantW || s.sideSize.Y != wantH {
			t.Errorf("%q: side size %dx%d, want %dx%d", input, s.sideSize.X, s.sideSize.Y, wantW, wantH)
			continue
		}
		if !bytes.Equal(s.matrix, wantMatrix) {
			for i := range wantMatrix {
				if s.matrix[i] != wantMatrix[i] {
					t.Errorf("%q: module[%d] (x=%d,y=%d) = %d, want %d", input, i, i%wantW, i/wantW, s.matrix[i], wantMatrix[i])
					break
				}
			}
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan golden: %v", err)
	}
}
