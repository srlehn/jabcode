package jabcode

import (
	"bufio"
	"encoding/hex"
	"os"
	"strings"
	"testing"
)

// TestEncodeDataGolden verifies the data-encoding stage (analyzeInputData +
// encodeData) reproduces the reference's chosen length and output bitstream for
// a range of inputs exercising the different encoding modes.
func TestEncodeDataGolden(t *testing.T) {
	f, err := os.Open("testdata/encdata_golden.txt")
	if err != nil {
		t.Fatalf("open golden: %v", err)
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) != 3 {
			t.Fatalf("malformed golden line: %q", line)
		}
		input, err := hex.DecodeString(fields[0])
		if err != nil {
			t.Fatalf("decode input hex: %v", err)
		}
		wantLen := mustAtoi(t, fields[1])
		wantBits, err := hex.DecodeString(fields[2])
		if err != nil {
			t.Fatalf("decode bitstream hex: %v", err)
		}

		seq, gotLen := analyzeInputData(input)
		if gotLen != wantLen {
			t.Errorf("%q: encoded length %d, want %d", input, gotLen, wantLen)
			continue
		}
		got, err := encodeData(input, gotLen, seq)
		if err != nil {
			t.Errorf("%q: encodeData: %v", input, err)
			continue
		}
		for i := range gotLen {
			wantBit := (wantBits[i/8] >> (7 - uint(i%8))) & 1
			if got[i] != wantBit {
				t.Errorf("%q: bit[%d]=%d, want %d", input, i, got[i], wantBit)
				break
			}
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan golden: %v", err)
	}
}
