package detect

import (
	"image"
	"math"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/encode"
)

// A pass sweeps up to six scan directions, so the selection counters cannot be
// pass-level assignments: whichever direction ran last would overwrite the one
// that produced the answer. Each direction records its own entry and the pass
// summary mirrors the published entry, which is what this pins.
func TestPassSummaryDescribesThePublishedScan(t *testing.T) {
	r, err := encode.Render(encode.Config{
		Colors: 8, ModuleSize: 1, SymbolNumber: 1,
		SymbolVersions: []image.Point{{X: 4, Y: 4}},
	}, []byte("the pass summary must name the direction that won"))
	if err != nil {
		t.Fatal(err)
	}
	for _, deg := range []float64{0, 35, 65} {
		rgba, _ := renderRotatedRGBA(r, 8, deg*math.Pi/180)
		bm := core.BitmapFromImage(rgba)
		BalanceRGB(bm)
		ch := BinarizerRGB(bm, nil)
		d := &PrimaryDetector{BM: bm, Ch: ch, Mode: IntensiveDetect}
		if !d.LocateFinders() {
			t.Errorf("deg=%.0f: no finders", deg)
			continue
		}
		published := 0
		for _, p := range d.Stats.Passes {
			if len(p.Scans) == 0 {
				t.Errorf("deg=%.0f: a pass recorded no scan directions", deg)
			}
			for _, s := range p.Scans {
				if !s.Published {
					continue
				}
				published++
				if s.Missing != p.Missing || s.Status != p.Status || s.Interpolated != p.Interpolated {
					t.Errorf("deg=%.0f: summary missing=%d status=%d interpolated=%v does not mirror dir=%g",
						deg, p.Missing, p.Status, p.Interpolated, s.Degrees)
				}
				if s.Preprune != p.Preprune || s.Selected != p.Selected {
					t.Errorf("deg=%.0f: summary groups/selection do not mirror dir=%g", deg, s.Degrees)
				}
			}
		}
		if published != 1 {
			t.Errorf("deg=%.0f: %d scans marked published, want exactly one", deg, published)
		}
	}
}

// A direction that fails must still leave its numbers behind. Without this the
// only visible entry would be the winner, and the question these counters exist
// to answer - why the other directions did not win - would be unanswerable.
func TestFailedScanDirectionsAreStillRecorded(t *testing.T) {
	r, err := encode.Render(encode.Config{
		Colors: 8, ModuleSize: 1, SymbolNumber: 1,
		SymbolVersions: []image.Point{{X: 4, Y: 4}},
	}, []byte("losing directions must leave their numbers behind"))
	if err != nil {
		t.Fatal(err)
	}
	rgba, _ := renderRotatedRGBA(r, 8, 35*math.Pi/180)
	bm := core.BitmapFromImage(rgba)
	BalanceRGB(bm)
	ch := BinarizerRGB(bm, nil)
	d := &PrimaryDetector{BM: bm, Ch: ch, Mode: IntensiveDetect}
	if !d.LocateFinders() {
		t.Fatal("no finders")
	}
	var swept, degrees int
	for _, p := range d.Stats.Passes {
		swept = max(swept, len(p.Scans))
		for i, s := range p.Scans {
			if i > 0 && s.Degrees == 0 {
				t.Errorf("scan %d records the row walk's 0 degrees", i)
			}
			degrees++
		}
	}
	if swept < 2 {
		t.Fatalf("no pass swept past the row walk (max %d scans), so this proves nothing", swept)
	}
	if degrees == 0 {
		t.Fatal("no scan directions recorded at all")
	}
}
