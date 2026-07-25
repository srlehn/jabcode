package read

import (
	"image"
	"testing"

	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/wire"
)

// TestRouteTraceWinner pins that the win is attributed to the first decoded
// attempt in collected order - the ladders merge in commit order and return at
// the slot that decoded, so a later decoded attempt can only be a straggler.
func TestRouteTraceWinner(t *testing.T) {
	tr := &routeTrace{levels: 3}
	tr.attempts = []routeAttempt{
		{kind: "upright", level: 0, roi: -1, stage: readNoFinders},
		{kind: "seeded", level: 2, roi: -1, stage: readDecoded, side: image.Pt(61, 61)},
		{kind: "rotated", level: 1, deg: 45, roi: -1, stage: readDecoded, side: image.Pt(61, 61)},
	}
	win, ok := tr.winner()
	if !ok || win.kind != "seeded" || win.level != 2 {
		t.Fatalf("winner() = %+v, %v; want the seeded attempt", win, ok)
	}
	got := tr.report().String()
	want := "decoded=true kind=seeded level=2 levels=3 angle=0 roi=-1 stage=decoded grid=61x61 attempts=3 by=upright:1,seeded:1,rotated:1,roi:0 at=aborted:0,no-finders:1,no-side-size:0,no-sample:0,sampled:0,decoded:2"
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
		{kind: "upright", level: 0, roi: -1, stage: readNoFinders},
		{kind: "roi", level: 1, deg: 45, roi: 2, stage: readSampled, side: image.Pt(53, 69)},
		{kind: "rotated", level: 1, deg: 90, roi: -1, stage: readNoSideSize},
	}
	if _, ok := tr.winner(); ok {
		t.Fatal("winner() reported a win on a failed read")
	}
	got := tr.report().String()
	want := "decoded=false kind=roi level=1 levels=2 angle=45 roi=2 stage=sampled grid=53x69 attempts=3 by=upright:1,seeded:0,rotated:1,roi:1 at=aborted:0,no-finders:1,no-side-size:1,no-sample:0,sampled:1,decoded:0"
	if got != want {
		t.Fatalf("report() = %q, want %q", got, want)
	}
}

// TestDecodeWithRouteAttributesTheLadder checks the reported route against two
// reads whose winning rung is known by construction: an upright code must win
// upright with no extra routes attempted, and a rotated one must win on a
// rotated rung at that angle.
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
		t.Fatalf("upright payload = %q, want %q", data, isoPayload(msg))
	}
	if !report.Decoded || report.Kind != "upright" || report.Angle != 0 || report.ROI != -1 {
		t.Fatalf("upright report = %v; want an upright whole-frame win", report)
	}
	if report.Attempts != 1 {
		t.Fatalf("upright report attempted %d routes, want the single upright route", report.Attempts)
	}

	_, report, err = DecodeWithRouteCapabilities(detect.RotateImage(img, 35), caps)
	if err != nil {
		t.Fatalf("DecodeWithRouteCapabilities rotated: %v", err)
	}
	if !report.Decoded || report.Kind != "rotated" || report.Angle == 0 {
		t.Fatalf("rotated report = %v; want a rotated whole-frame win", report)
	}
	if report.Attempts < 2 {
		t.Fatalf("rotated report attempted %d routes, want the upright rung plus a rotated one", report.Attempts)
	}
}
