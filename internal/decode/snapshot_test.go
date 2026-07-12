package decode

import (
	"bytes"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
)

// TestSnapshotIsDeeplyOwned pins the retention ownership rule: a snapshot
// shares no state with its live observation, so mutating the sampled matrix,
// the symbol, or running a payload correction afterwards cannot corrupt what
// was banked.
func TestSnapshotIsDeeplyOwned(t *testing.T) {
	bm, _, _ := softPathSymbol(t, []byte("snapshot ownership"))
	sym := &core.DecodedSymbol{}
	obs, ret := ObservePrimary(bm, sym)
	if ret != core.Success || obs == nil {
		t.Fatalf("observation failed: %d", ret)
	}

	snap := obs.Snapshot()
	if !snap.Admitted || snap.FixedChecked == 0 || snap.FixedAgree != snap.FixedChecked {
		t.Fatalf("clean observation snapshot not admitted/perfect: %+v", snap)
	}
	modules := append([]byte(nil), snap.Modules...)
	palette := append([]byte(nil), snap.Palette...)
	meta := snap.Meta

	// Corrupt the live state: pixels, palette, symbol metadata, and spend a
	// payload correction (which mutates the symbol's data buffers).
	for i := range obs.Matrix.Pix {
		obs.Matrix.Pix[i] ^= 0xFF
	}
	for i := range obs.Symbol.Palette {
		obs.Symbol.Palette[i] ^= 0xFF
	}
	obs.Symbol.Meta.NC = 7
	obs.CorrectPayload()

	if !bytes.Equal(snap.Modules, modules) {
		t.Error("snapshot modules changed with the live matrix")
	}
	if !bytes.Equal(snap.Palette, palette) {
		t.Error("snapshot palette changed with the live symbol")
	}
	if snap.Meta != meta {
		t.Error("snapshot metadata changed with the live symbol")
	}
}
