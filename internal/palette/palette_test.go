package palette

import (
	"bufio"
	"bytes"
	"encoding/hex"
	"os"
	"strings"
	"testing"

	"github.com/srlehn/jabcode/internal/testutil"
)

// TestSetDefaultPaletteGolden checks that the generated default palettes match
// the reference library byte for byte, across every supported color count.
func TestSetDefaultPaletteGolden(t *testing.T) {
	f, err := os.Open(testutil.TestdataPath("palette_golden.txt"))
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
		if len(fields) != 2 {
			t.Fatalf("malformed golden line: %q", line)
		}
		cn := testutil.MustAtoi(t, fields[0])
		want, err := hex.DecodeString(fields[1])
		if err != nil {
			t.Fatalf("decode golden hex: %v", err)
		}
		got := SetDefault(cn)
		if !bytes.Equal(got, want) {
			t.Errorf("colors=%d: palette mismatch\n got %x\nwant %x", cn, got, want)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan golden: %v", err)
	}
}
