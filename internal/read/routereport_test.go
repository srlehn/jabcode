package read

import (
	"image"
	"testing"

	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/testutil"
	"github.com/srlehn/jabcode/internal/wire"
)

// TestRouteTraceWinner pins that the win is attributed to the first decoded
// attempt in collected order - the ladders merge in commit order and return at
// the slot that decoded, so a later decoded attempt can only be a straggler.
func TestRouteTraceWinner(t *testing.T) {
	tr := &routeTrace{levels: 3}
	tr.attempts = []routeAttempt{
		{kind: "frame", level: 0, roi: -1, stage: readNoFinders},
		{kind: "seeded", level: 2, roi: -1, stage: readDecoded, side: image.Pt(61, 61)},
		{kind: "roi", level: 1, roi: 0, stage: readDecoded, side: image.Pt(61, 61)},
	}
	win, ok := tr.winner()
	if !ok || win.kind != "seeded" || win.level != 2 {
		t.Fatalf("winner() = %+v, %v; want the seeded attempt", win, ok)
	}
	got := tr.report().String()
	want := "decoded=true kind=seeded deg=0 level=2 levels=3 roi=-1 stage=decoded grid=61x61 attempts=3 by=frame:1,seeded:1,roi:1 at=aborted:0,no-finders:1,no-side-size:0,no-sample:0,sampled:0,decoded:2"
	if got != want {
		t.Fatalf("report() = %q, want %q", got, want)
	}
}

// TestRouteTraceReportFailure pins that a failed read reports the furthest rung
// reached instead of nothing, which is what makes the census usable on the
// frames that do not decode.
func TestRouteTraceReportFailure(t *testing.T) {
	tr := &routeTrace{levels: 2}
	tr.attempts = []routeAttempt{
		{kind: "frame", level: 0, roi: -1, stage: readNoFinders, deg: -1},
		{kind: "roi", level: 1, roi: 2, stage: readSampled, side: image.Pt(53, 69), deg: 30},
		{kind: "roi", level: 1, roi: 3, stage: readNoSideSize, deg: -1},
	}
	if _, ok := tr.winner(); ok {
		t.Fatal("winner() reported a win on a failed read")
	}
	got := tr.report().String()
	want := "decoded=false kind=roi deg=30 level=1 levels=2 roi=2 stage=sampled grid=53x69 attempts=3 by=frame:1,seeded:0,roi:2 at=aborted:0,no-finders:1,no-side-size:1,no-sample:0,sampled:1,decoded:0"
	if got != want {
		t.Fatalf("report() = %q, want %q", got, want)
	}
}

// Deg has to distinguish three states that a float cannot carry on its own:
// a located quad from the row walk, a located quad from a turned scan, and no
// quad at all. Zero is the first of those, so the third has to be -1 - and a
// report built from no attempts at all must not fall through to zero and claim
// the search stayed on image rows.
func TestRouteReportDegStates(t *testing.T) {
	for _, tc := range []struct {
		name     string
		attempts []routeAttempt
		want     float64
	}{
		{"no attempts at all", nil, -1},
		{
			"the best attempt located nothing",
			[]routeAttempt{{kind: "frame", level: 0, roi: -1, stage: readNoFinders, deg: -1}},
			-1,
		},
		{
			"a located row-scan result keeps its zero",
			[]routeAttempt{{kind: "frame", level: 0, roi: -1, stage: readDecoded, deg: 0}},
			0,
		},
		{
			"a located turned result keeps its direction",
			[]routeAttempt{{kind: "frame", level: 0, roi: -1, stage: readDecoded, deg: 60}},
			60,
		},
		{
			"a failed read still reports the best attempt's direction",
			[]routeAttempt{
				{kind: "frame", level: 0, roi: -1, stage: readNoFinders, deg: -1},
				{kind: "roi", level: 0, roi: 1, stage: readSampled, deg: 45},
			},
			45,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tr := &routeTrace{attempts: tc.attempts}
			if got := tr.report().Deg; got != tc.want {
				t.Fatalf("report().Deg = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestDecodeWithRouteAttributesTheLadder checks the reported route against two
// reads whose winning rung is known by construction. Both win on the whole-frame
// route with no extra routes attempted - no route resamples pixels to try an
// angle, so a rotated input takes the same rung and is separated only by the
// direction it reports.
func TestDecodeWithRouteAttributesTheLadder(t *testing.T) {
	msg := []byte("route attribution")
	img, err := encode.Run(encode.Config{Colors: 8, ModuleSize: 8, SymbolNumber: 1}, msg)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	caps := wire.ISO23634.Mask()

	data, report, err := DecodeWithRouteCapabilities(img, caps)
	if err != nil {
		t.Fatalf("DecodeWithRouteCapabilities: %v", err)
	}
	if string(data) != string(isoPayload(msg)) {
		t.Fatalf("unrotated payload = %q, want %q", data, isoPayload(msg))
	}
	if !report.Decoded || report.Kind != "frame" || report.ROI != -1 {
		t.Fatalf("unrotated report = %v; want a whole-frame win", report)
	}
	if report.Attempts != 1 {
		t.Fatalf("unrotated report attempted %d routes, want the single whole-frame route", report.Attempts)
	}

	_, report, err = DecodeWithRouteCapabilities(testutil.RotateImage(img, 35), caps)
	if err != nil {
		t.Fatalf("DecodeWithRouteCapabilities rotated: %v", err)
	}
	// A rotated frame is answered by the finder scan's own directions, so it
	// wins on the whole-frame route without a rotated canvas ever being built.
	if !report.Decoded || report.Kind != "frame" || report.ROI != -1 {
		t.Fatalf("rotated report = %v; want a whole-frame win", report)
	}
	if report.Attempts != 1 {
		t.Fatalf("rotated report attempted %d routes, want the single whole-frame route", report.Attempts)
	}
}
