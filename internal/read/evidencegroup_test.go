package read

import (
	"image"
	"math"
	"slices"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/ecc"
	"github.com/srlehn/jabcode/internal/encode"
)

// groupSnap builds a minimal snapshot with the given layout.
func groupSnap(side image.Point, nc, mask int, ecl image.Point, fix int) *decode.ObservationSnapshot {
	return &decode.ObservationSnapshot{
		Side:         side,
		Meta:         core.Metadata{NC: nc, MaskType: mask, ECL: ecl},
		FixedAgree:   fix,
		FixedChecked: fix,
	}
}

// groupQuad builds a located finding whose corners sit on an axis-aligned
// square of the given origin and span.
func groupQuad(x, y, span float64) finding {
	return finding{
		quad: [4]core.PointF{
			{X: x, Y: y}, {X: x + span, Y: y},
			{X: x + span, Y: y + span}, {X: x, Y: y + span},
		},
		located: true,
	}
}

func TestEvidenceGroupContracts(t *testing.T) {
	side := image.Pt(85, 85)
	ecl := image.Pt(6, 7)
	src := image.Pt(1000, 1000)
	var g evidenceGroup

	// First admitted observation anchors the group.
	if admitted, _ := g.admit(groupSnap(side, 2, 7, ecl, 100), groupQuad(100, 100, 200), src); !admitted {
		t.Fatal("first observation not admitted")
	}
	if len(g.snaps) != 1 || g.anchorFix != 100 {
		t.Fatalf("anchor not established: %d snaps, fix %d", len(g.snaps), g.anchorFix)
	}

	// Layout mismatches are reject-only: wrong side, colour mode, mask, ECC.
	for _, bad := range []*decode.ObservationSnapshot{
		groupSnap(image.Pt(77, 77), 2, 7, ecl, 90),
		groupSnap(side, 3, 7, ecl, 90),
		groupSnap(side, 2, 3, ecl, 90),
		groupSnap(side, 2, 7, image.Pt(4, 9), 90),
	} {
		if admitted, _ := g.admit(bad, groupQuad(100, 100, 200), src); admitted {
			t.Fatalf("incompatible layout admitted: %+v", bad.Meta)
		}
		g.rejects = 0 // isolate the compatibility check from the reset rule
	}
	if len(g.snaps) != 1 {
		t.Fatalf("rejections mutated retained evidence: %d snaps", len(g.snaps))
	}

	// A quad a full span away is another track.
	if admitted, _ := g.admit(groupSnap(side, 2, 7, ecl, 90), groupQuad(500, 500, 200), src); admitted {
		t.Fatal("off-track quad admitted")
	}
	g.rejects = 0

	// Compatible drifting observations admit; the cap holds in input order.
	for i := range 12 {
		if admitted, _ := g.admit(groupSnap(side, 2, 7, ecl, 90), groupQuad(100+float64(i), 100, 200), src); !admitted {
			t.Fatalf("compatible observation %d rejected", i)
		}
	}
	if len(g.snaps) != evidenceGroupCap {
		t.Fatalf("cap not held: %d snaps", len(g.snaps))
	}

	// One strictly better observation re-anchors; a second better one does not.
	if admitted, _ := g.admit(groupSnap(side, 2, 7, ecl, 150), groupQuad(110, 100, 200), src); !admitted {
		t.Fatal("better observation rejected")
	}
	if g.anchorFix != 150 || g.reanchors != 1 {
		t.Fatalf("re-anchor missing: fix %d, reanchors %d", g.anchorFix, g.reanchors)
	}
	if admitted, _ := g.admit(groupSnap(side, 2, 7, ecl, 200), groupQuad(111, 100, 200), src); !admitted {
		t.Fatal("later observation rejected")
	}
	if g.anchorFix != 150 {
		t.Fatalf("second re-anchor happened: fix %d", g.anchorFix)
	}

	// Persistent coherent incompatibility resets the group.
	for range evidenceResetAfter {
		g.admit(groupSnap(side, 3, 7, ecl, 90), groupQuad(111, 100, 200), src)
	}
	if len(g.snaps) != 0 || g.anchorSrc != (image.Point{}) {
		t.Fatalf("group did not reset after %d coherent mismatches", evidenceResetAfter)
	}
}

func TestCalibrateFrameEvidence(t *testing.T) {
	base := []float64{65025, -32512.5, 650.25, 650250}
	got := calibrateFrameEvidence(base, 255, 0)
	want := []float64{1, -0.5, 0.01, 1}
	if len(got) != len(want) {
		t.Fatalf("calibrated length %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Fatalf("calibrated[%d] = %.12f, want %.12f", i, got[i], want[i])
		}
	}

	// Uniform photometric scaling multiplies squared candidate costs and the
	// squared palette separation by the same factor, so it must not redefine
	// the channel evidence.
	scaled := make([]float64, len(base))
	for i, v := range base {
		scaled[i] = v * 0.25
	}
	gotScaled := calibrateFrameEvidence(scaled, 127.5, 0)
	for i := range got {
		if math.Abs(gotScaled[i]-got[i]) > 1e-12 {
			t.Fatalf("scaled evidence[%d] = %.12f, want %.12f", i, gotScaled[i], got[i])
		}
	}

	// Palette disagreement reduces a frame's authority without changing its
	// signs. Equal disagreement and separation give one-half weight.
	noisy := calibrateFrameEvidence(base, 255, 255)
	for i := range got {
		if math.Abs(noisy[i]-got[i]/2) > 1e-12 {
			t.Fatalf("noisy evidence[%d] = %.12f, want %.12f", i, noisy[i], got[i]/2)
		}
	}
	if got := calibrateFrameEvidence(base, 0, 0); got != nil {
		t.Fatalf("zero palette separation produced %d evidence values", len(got))
	}
}

func TestEvidenceAccumulatorContracts(t *testing.T) {
	var a evidenceAccumulator
	first := []float64{0.75, -0.5, 0}
	if !a.add(first) {
		t.Fatal("first evidence frame rejected")
	}
	near := []float64{0.75 + evidenceDuplicateTolerance/2, -0.5, 0}
	if a.add(near) {
		t.Fatal("near-duplicate evidence frame admitted")
	}
	if a.duplicates != 1 || a.frames != 1 {
		t.Fatalf("duplicate accounting = %d duplicates, %d frames", a.duplicates, a.frames)
	}

	// A materially different agreeing observation increases confidence.
	second := []float64{0.5, -0.25, 0.2}
	if !a.add(second) {
		t.Fatal("complementary agreeing frame rejected")
	}
	if a.signed[0] <= first[0] || math.Abs(a.signed[1]) <= math.Abs(first[1]) {
		t.Fatalf("agreeing evidence did not grow: %v", a.signed)
	}
	if a.samples[2] != 1 || math.Abs(a.weight[2]-0.2) > 1e-12 || math.Abs(a.weightSquared[2]-0.04) > 1e-12 {
		t.Fatalf("effective-sample statistics not recorded: samples=%d weight=%g squared=%g",
			a.samples[2], a.weight[2], a.weightSquared[2])
	}
	if got := a.effectiveSamples(0); math.Abs(got-1.9230769230769231) > 1e-12 {
		t.Fatalf("effective samples = %.12f, want 1.923076923077", got)
	}

	// Opposing valid evidence cancels instead of manufacturing confidence.
	var conflict evidenceAccumulator
	if !conflict.add([]float64{0.8}) || !conflict.add([]float64{-0.6}) {
		t.Fatal("conflicting observations were not retained")
	}
	if math.Abs(conflict.signed[0]-0.2) > 1e-12 {
		t.Fatalf("conflicting evidence summed to %.12f, want 0.2", conflict.signed[0])
	}
}

// TestEvidenceGroupAccumulatedSnapshot exercises the complete retained path:
// sampled snapshot, trusted layout, mask and deinterleave, frame calibration,
// duplicate suppression, accumulation, and direct signed LDPC consumption.
func TestEvidenceGroupAccumulatedSnapshot(t *testing.T) {
	payload := []byte("retained evidence group")
	r, err := encode.Render(encode.Config{Colors: 8, ModuleSize: 1, ECCLevel: 10, SymbolNumber: 1}, payload)
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	bm := core.NewBitmap(r.SideSize.X, r.SideSize.Y, 4)
	for i, idx := range r.Matrix {
		copy(bm.Pix[i*4:], r.Palette[int(idx)*3:int(idx)*3+3])
		bm.Pix[i*4+3] = 255
	}
	sym := &core.DecodedSymbol{}
	obs, ret := decode.ObservePrimary(bm, sym)
	if ret != core.Success || obs == nil {
		t.Fatalf("observe: %d", ret)
	}
	snap := obs.Snapshot()
	if !snap.Admitted {
		t.Fatal("clean snapshot not admitted")
	}
	untrusted := *snap
	untrusted.PartISyndromeOK = false

	g := evidenceGroup{snaps: []*decode.ObservationSnapshot{snap, snap, &untrusted}}
	a := g.accumulatedEvidence()
	if a.frames != 1 || a.duplicates != 1 {
		t.Fatalf("accumulated %d frames and %d duplicates, want 1 and 1", a.frames, a.duplicates)
	}
	if len(a.signed) == 0 {
		t.Fatal("snapshot produced no accumulated evidence")
	}
	hard := make([]byte, len(a.signed))
	for i, v := range a.signed {
		if v < 0 {
			hard[i] = 1
		}
	}
	want, hardOK := ecc.DecodeLDPCHard(hard, snap.Meta.ECL.X, snap.Meta.ECL.Y)
	got, signedOK := ecc.DecodeLDPCSigned(a.signed, snap.Meta.ECL.X, snap.Meta.ECL.Y)
	if !hardOK || !signedOK || !slices.Equal(got, want) {
		t.Fatalf("signed accumulated decode mismatch: hardOK=%v signedOK=%v", hardOK, signedOK)
	}
}

func TestEvidenceCorrectionSchedule(t *testing.T) {
	const pn, wc, wr = 100, 4, 9
	message := make([]byte, pn)
	for i := range message {
		message[i] = byte((i*7 + i/3) & 1)
	}
	codeword := ecc.EncodeLDPC(message, wc, wr)
	frame := func(magnitude float64) []float64 {
		out := make([]float64, len(codeword))
		for i, bit := range codeword {
			out[i] = magnitude
			if bit != 0 {
				out[i] = -magnitude
			}
		}
		return out
	}

	var a evidenceAccumulator
	a.add(frame(0.6))
	a.add(frame(0.4))
	g := evidenceGroup{meta: layoutKey{ecl: image.Pt(wc, wr)}, version: 2}
	if !g.correctionDue(&a) {
		t.Fatal("first two-frame aggregate was not scheduled")
	}
	g.recordAttempt(&a)
	if g.correctionDue(&a) {
		t.Fatal("unchanged aggregate was rescheduled")
	}

	// A unique but weak agreeing frame does not cross a whole evidence unit.
	a.add(frame(0.3))
	g.version++
	if g.correctionDue(&a) {
		t.Fatal("sub-material evidence gain was scheduled")
	}

	// More independent agreement crosses the next bounded unit and schedules.
	a.add(frame(0.8))
	g.version++
	if !g.correctionDue(&a) {
		t.Fatal("material evidence gain was not scheduled")
	}
}
