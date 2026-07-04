package detect

import (
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/testutil"
)

// TestDetectPrimarySnapshot pins findPrimarySymbol's outputs (the four selected
// finder patterns and the status) on a clean, in-memory-encoded symbol run
// through the real pre-finder chain (BitmapFromImage -> BalanceRGB ->
// BinarizerRGB -> findPrimarySymbol). It exists so a behaviour-preserving
// refactor can be verified by Go self-invariance: the snapshot must be
// byte-identical before and after. Regenerate the golden by deleting
// testdata/detect_primary_snapshot.golden.
func TestDetectPrimarySnapshot(t *testing.T) {
	golden := testutil.TestdataPath("detect_primary_snapshot.golden")

	img, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 12, SymbolNumber: 1}, []byte("Just Another Bar Code 2024"))
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	bm := core.BitmapFromImage(img)
	BalanceRGB(bm)
	ch := BinarizerRGB(bm, nil)

	d := &PrimaryDetector{BM: bm, Ch: ch, Mode: IntensiveDetect}
	status := d.findPrimarySymbol()
	got := snapshotFinderPatterns(d.FPs, status)

	want, err := os.ReadFile(golden)
	if err != nil {
		if werr := os.WriteFile(golden, []byte(got), 0o644); werr != nil {
			t.Fatalf("write golden: %v", werr)
		}
		t.Logf("wrote golden %s", golden)
		return
	}
	if got != string(want) {
		t.Errorf("finder-pattern snapshot changed:\n--- got ---\n%s--- want ---\n%s", got, want)
	}
}

// snapshotFinderPatterns renders the four selected finder patterns and the
// detection status as deterministic text. Floats use shortest exact form, so an
// arithmetic-preserving refactor produces identical bytes.
func snapshotFinderPatterns(fps []FinderPattern, status int) string {
	var b strings.Builder
	b.WriteString("status=")
	b.WriteString(strconv.Itoa(status))
	b.WriteString("\n")
	for i := range 4 {
		fp := fps[i]
		b.WriteString("fp")
		b.WriteString(strconv.Itoa(i))
		b.WriteString(" typ=")
		b.WriteString(strconv.Itoa(fp.Typ))
		b.WriteString(" cx=")
		b.WriteString(strconv.FormatFloat(fp.Center.X, 'g', -1, 64))
		b.WriteString(" cy=")
		b.WriteString(strconv.FormatFloat(fp.Center.Y, 'g', -1, 64))
		b.WriteString(" ms=")
		b.WriteString(strconv.FormatFloat(fp.ModuleSize, 'g', -1, 64))
		b.WriteString(" fc=")
		b.WriteString(strconv.Itoa(fp.FoundCount))
		b.WriteString(" dir=")
		b.WriteString(strconv.Itoa(fp.direction))
		b.WriteString("\n")
	}
	return b.String()
}
