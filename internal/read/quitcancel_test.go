package read

import (
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/encode"
)

// TestDecodeLocatedDetectorStopsAfterLosing pins the cancellation window that
// only opens once a finder walk has already returned: a route that lost while
// it was locating must abandon the consensus, sampling and correction tail
// whose result can no longer be used, and the identical detector must still
// decode when nothing cancelled it. Without the poll every losing route in the
// ladder runs that tail to completion in the background.
func TestDecodeLocatedDetectorStopsAfterLosing(t *testing.T) {
	msg := []byte("cancel the payload tail")
	img, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 12, SymbolNumber: 1}, msg)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	locate := func(quit func() bool) (*detect.PrimaryDetector, detect.FinderFamilySet) {
		bm := core.BitmapFromImage(img)
		detect.BalanceRGB(bm)
		d := &detect.PrimaryDetector{
			BM:   bm,
			Ch:   detect.BinarizerRGB(bm, nil),
			Mode: detect.IntensiveDetect,
			Quit: quit,
		}
		return d, d.LocateFinderFamilies(finderFamiliesForCapabilities(compiledCapabilities()))
	}

	lost := false
	d, found := locate(func() bool { return lost })
	lost = true
	data, stage, _ := decodeLocatedDetector(d, found, nil, nil, compiledCapabilities())
	if stage == readDecoded || data != nil {
		t.Fatalf("route that already lost decoded anyway: stage %v", stage)
	}

	d, found = locate(nil)
	data, stage, _ = decodeLocatedDetector(d, found, nil, nil, compiledCapabilities())
	if stage != readDecoded || data == nil {
		t.Fatalf("uncancelled route did not decode: stage %v", stage)
	}
	if got, want := string(messageTransmission(data)), string(isoPayload(msg)); got != want {
		t.Fatalf("payload %q, want %q", got, want)
	}
}
