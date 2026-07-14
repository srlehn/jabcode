package ecc

import (
	"bufio"
	"encoding/hex"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/srlehn/jabcode/internal/testutil"
	"github.com/srlehn/jabcode/internal/wire"
)

// ldpcInputBit reproduces the deterministic input pattern used by the C oracle
// harness so the test feeds EncodeLDPC the same message bits the C reference was
// given.
func ldpcInputBit(i int) byte { return byte((uint32(i) * 2654435761) >> 31) }

// TestEncodeLDPCGolden checks that EncodeLDPC reproduces the reference library's
// output bit for bit, across several lengths and code rates. Behaviour parity is
// required for cross-compatibility: the reference decoder reconstructs the same
// matrices from the same seeds and expects this exact codeword layout.
func TestEncodeLDPCGolden(t *testing.T) {
	f, err := os.Open(testutil.TestdataPath("ldpc_golden.txt"))
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
		if len(fields) != 5 {
			t.Fatalf("malformed golden line: %q", line)
		}
		Pn := mustAtoi(t, fields[0])
		wc := mustAtoi(t, fields[1])
		wr := mustAtoi(t, fields[2])
		Pg := mustAtoi(t, fields[3])
		want, err := hex.DecodeString(fields[4]) // Pg bits packed 8 per byte, MSB first
		if err != nil {
			t.Fatalf("decode golden hex: %v", err)
		}

		in := make([]byte, Pn)
		for i := range in {
			in[i] = ldpcInputBit(i)
		}
		got := EncodeLDPCProfile(in, wc, wr, wire.Legacy)

		if len(got) != Pg {
			t.Errorf("Pn=%d wc=%d wr=%d: length %d, want %d", Pn, wc, wr, len(got), Pg)
			continue
		}
		for i := range Pg {
			wantBit := (want[i/8] >> (7 - uint(i%8))) & 1
			if got[i] != wantBit {
				t.Errorf("Pn=%d wc=%d wr=%d: bit[%d]=%d, want %d", Pn, wc, wr, i, got[i], wantBit)
				break
			}
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan golden: %v", err)
	}
}

func mustAtoi(t *testing.T, s string) int {
	t.Helper()
	n, err := strconv.Atoi(s)
	if err != nil {
		t.Fatalf("atoi %q: %v", s, err)
	}
	return n
}
