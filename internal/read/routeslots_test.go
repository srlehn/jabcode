package read

import (
	"bytes"
	"image"
	"testing"
)

// TestRunRouteSlotsOrderedCommit pins the scheduler's contract: the lowest
// decoded slot wins even when a higher slot decodes first, findings publish
// under the sequential ladder's rule (a decode always overwrites, a
// locate-only result never displaces an earlier locate), and traces merge in
// slot order up to the winner only.
func TestRunRouteSlotsOrderedCommit(t *testing.T) {
	var f finding
	tr := &routeTrace{level: 3}
	release := make(chan struct{})
	data, deg, ok := runRouteSlots(nil, tr, &f, 4,
		func(slot int, slotQuit func() bool, slotTr *routeTrace) routeSlotResult {
			switch slot {
			case 0: // locates without decoding
				<-release
				slotTr.add(routeAttempt{deg: 10, roi: -1, stage: readSampled})
				return routeSlotResult{
					deg: 10, stage: readSampled,
					rf:     finding{located: true, side: image.Pt(1, 1)},
					canvas: image.Pt(100, 100), srcW: 100, srcH: 100,
				}
			case 1: // decodes second in wall time, but owns the lowest decoded slot
				<-release
				slotTr.add(routeAttempt{deg: 20, roi: -1, stage: readDecoded})
				return routeSlotResult{
					data: &Message{Data: []byte("winner"), ReaderTransmission: []byte("winner")}, deg: 20, stage: readDecoded,
					rf:     finding{located: true, side: image.Pt(2, 2)},
					canvas: image.Pt(100, 100), srcW: 100, srcH: 100,
				}
			case 2: // fails outright
				<-release
				slotTr.add(routeAttempt{deg: 30, roi: -1, stage: readNoFinders})
				return routeSlotResult{deg: 30, stage: readNoFinders}
			default: // decodes first in wall time from the highest slot
				slotTr.add(routeAttempt{deg: 40, roi: -1, stage: readDecoded})
				close(release)
				return routeSlotResult{
					data: &Message{Data: []byte("outranked"), ReaderTransmission: []byte("outranked")}, deg: 40, stage: readDecoded,
					rf:     finding{located: true, side: image.Pt(4, 4)},
					canvas: image.Pt(100, 100), srcW: 100, srcH: 100,
				}
			}
		})
	if !ok || !bytes.Equal(messageTransmission(data), []byte("winner")) || deg != 20 {
		t.Fatalf("route slots returned %q deg %v ok %v, want winner at 20", messageTransmission(data), deg, ok)
	}
	if !f.located || f.side != image.Pt(2, 2) || !bytes.Equal(messageTransmission(f.payload), []byte("winner")) {
		t.Fatalf("published finding = %+v, want the winning slot's decode finding", f)
	}
	want := []routeAttempt{
		{level: 3, deg: 10, roi: -1, stage: readSampled},
		{level: 3, deg: 20, roi: -1, stage: readDecoded},
	}
	if len(tr.attempts) != len(want) {
		t.Fatalf("merged %d attempts %+v, want the slots up to the winner in order", len(tr.attempts), tr.attempts)
	}
	for index, attempt := range want {
		if tr.attempts[index] != attempt {
			t.Fatalf("attempt %d = %+v, want %+v", index, tr.attempts[index], attempt)
		}
	}
}

// TestRunRouteSlotsNoWinner pins the exhaustion path: every slot merges in
// order and the first located finding stays published under the
// no-displacement rule.
func TestRunRouteSlotsNoWinner(t *testing.T) {
	var f finding
	tr := &routeTrace{level: -1}
	data, deg, ok := runRouteSlots(nil, tr, &f, 3,
		func(slot int, slotQuit func() bool, slotTr *routeTrace) routeSlotResult {
			slotTr.add(routeAttempt{deg: float64(slot), roi: -1, stage: readSampled})
			return routeSlotResult{
				deg: float64(slot), stage: readSampled,
				rf:     finding{located: slot > 0, side: image.Pt(slot, slot)},
				canvas: image.Pt(10, 10), srcW: 10, srcH: 10,
			}
		})
	if ok || data != nil || deg != 0 {
		t.Fatalf("exhausted slots returned %q deg %v ok %v, want no winner", messageTransmission(data), deg, ok)
	}
	if !f.located || f.side != image.Pt(1, 1) {
		t.Fatalf("published finding = %+v, want the first located slot kept", f)
	}
	if len(tr.attempts) != 3 {
		t.Fatalf("merged %d attempts, want all three in order", len(tr.attempts))
	}
	for index, attempt := range tr.attempts {
		if attempt.deg != float64(index) {
			t.Fatalf("attempt %d has deg %v, want slot order preserved", index, attempt.deg)
		}
	}
}

// TestRunRouteSlotsOuterQuit pins cancellation: a quit hook that already
// fired aborts every slot before its route body runs.
func TestRunRouteSlotsOuterQuit(t *testing.T) {
	var f finding
	tr := &routeTrace{level: 0}
	ran := false
	data, _, ok := runRouteSlots(func() bool { return true }, tr, &f, 2,
		func(slot int, slotQuit func() bool, slotTr *routeTrace) routeSlotResult {
			ran = true
			return routeSlotResult{stage: readDecoded, data: &Message{Data: []byte("unreachable"), ReaderTransmission: []byte("unreachable")}}
		})
	if ok || data != nil || ran {
		t.Fatalf("cancelled slots returned %q ok %v ran %v, want aborted without running", messageTransmission(data), ok, ran)
	}
	if f.located || len(tr.attempts) != 0 {
		t.Fatalf("cancelled slots published finding %+v with %d attempts, want none", f, len(tr.attempts))
	}
}
