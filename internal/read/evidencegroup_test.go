package read

import (
	"image"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
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
	if !g.admit(groupSnap(side, 2, 7, ecl, 100), groupQuad(100, 100, 200), src) {
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
		if g.admit(bad, groupQuad(100, 100, 200), src) {
			t.Fatalf("incompatible layout admitted: %+v", bad.Meta)
		}
		g.rejects = 0 // isolate the compatibility check from the reset rule
	}
	if len(g.snaps) != 1 {
		t.Fatalf("rejections mutated retained evidence: %d snaps", len(g.snaps))
	}

	// A quad a full span away is another track.
	if g.admit(groupSnap(side, 2, 7, ecl, 90), groupQuad(500, 500, 200), src) {
		t.Fatal("off-track quad admitted")
	}
	g.rejects = 0

	// Compatible drifting observations admit; the cap holds in input order.
	for i := range 12 {
		if !g.admit(groupSnap(side, 2, 7, ecl, 90), groupQuad(100+float64(i), 100, 200), src) {
			t.Fatalf("compatible observation %d rejected", i)
		}
	}
	if len(g.snaps) != evidenceGroupCap {
		t.Fatalf("cap not held: %d snaps", len(g.snaps))
	}

	// One strictly better observation re-anchors; a second better one does not.
	if !g.admit(groupSnap(side, 2, 7, ecl, 150), groupQuad(110, 100, 200), src) {
		t.Fatal("better observation rejected")
	}
	if g.anchorFix != 150 || g.reanchors != 1 {
		t.Fatalf("re-anchor missing: fix %d, reanchors %d", g.anchorFix, g.reanchors)
	}
	if !g.admit(groupSnap(side, 2, 7, ecl, 200), groupQuad(111, 100, 200), src) {
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
