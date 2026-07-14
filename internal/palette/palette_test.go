package palette

import (
	"bufio"
	"bytes"
	"encoding/hex"
	"os"
	"strings"
	"testing"

	"github.com/srlehn/jabcode/internal/tables"
	"github.com/srlehn/jabcode/internal/testutil"
	"github.com/srlehn/jabcode/internal/wire"
)

func TestISOFourColorPalette(t *testing.T) {
	want := []byte{
		0, 0, 0,
		0, 255, 255,
		255, 0, 255,
		255, 255, 0,
	}
	if got := SetDefaultProfile(4, wire.ISO23634); !bytes.Equal(got, want) {
		t.Fatalf("ISO four-color palette = %v, want %v", got, want)
	}
}

// TestNcMetadataColorIsBlackCyanYellow checks the end-to-end contract behind the
// Part I color-mode marker: the palette index tables.NcMetadataColorIndex chooses
// for each mode must actually carry black, cyan and yellow, so the pre-palette
// DecodeModuleNC reads the marker back by color pattern. This ties the marker
// mapping to the real palettes SetDefault builds, across the modes that embed
// every color.
func TestNcMetadataColorIsBlackCyanYellow(t *testing.T) {
	ncOf := map[int]int{4: 1, 8: 2, 16: 3, 32: 4, 64: 5, 128: 6, 256: 7}
	rgb := func(pal []byte, idx int) [3]byte {
		return [3]byte{pal[idx*3], pal[idx*3+1], pal[idx*3+2]}
	}
	for _, colors := range []int{4, 8, 16, 32, 64, 128, 256} {
		pal := SetDefaultProfile(colors, wire.Legacy)
		nc := ncOf[colors]
		want := map[int][3]byte{
			0: {0, 0, 0},     // black
			3: {0, 255, 255}, // cyan
			6: {255, 255, 0}, // yellow
		}
		for value, wantRGB := range want {
			idx := tables.NcMetadataColorIndexProfile(value, nc, wire.Legacy)
			if got := rgb(pal, idx); got != wantRGB {
				t.Errorf("colors=%d value=%d -> index %d = %v, want %v", colors, value, idx, got, wantRGB)
			}
		}
	}
}

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
		got := SetDefaultProfile(cn, wire.Legacy)
		if !bytes.Equal(got, want) {
			t.Errorf("colors=%d: palette mismatch\n got %x\nwant %x", cn, got, want)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan golden: %v", err)
	}
}
