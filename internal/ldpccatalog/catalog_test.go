package ldpccatalog

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/srlehn/jabcode/internal/ecc"
)

// legalKeys enumerates the complete key space the catalog claims to cover.
func legalKeys() [][3]int {
	var all [][3]int
	for wr := MinRowWeight; wr <= MaxRowWeight; wr++ {
		for wc := MinColWeight; wc < wr; wc++ {
			for capacity := wr; capacity <= MaxCapacity; capacity += wr {
				all = append(all, [3]int{wc, wr, capacity})
			}
		}
	}
	return all
}

func TestSlotIsAPerfectHash(t *testing.T) {
	all := legalKeys()
	if len(all) != SlotCount() {
		t.Fatalf("legal keys = %d, SlotCount = %d", len(all), SlotCount())
	}
	seen := make([]bool, SlotCount())
	for _, key := range all {
		slot, ok := Slot(key[0], key[1], key[2])
		if !ok {
			t.Fatalf("legal key wc=%d wr=%d capacity=%d has no slot", key[0], key[1], key[2])
		}
		if slot < 0 || slot >= SlotCount() {
			t.Fatalf("key wc=%d wr=%d capacity=%d maps outside the directory at %d", key[0], key[1], key[2], slot)
		}
		if seen[slot] {
			t.Fatalf("slot %d is claimed twice, at wc=%d wr=%d capacity=%d", slot, key[0], key[1], key[2])
		}
		seen[slot] = true
	}
}

func TestSlotRejectsUnsupportedShapes(t *testing.T) {
	for _, key := range [][3]int{
		{3, 3, 12}, {2, 4, 8}, {3, 12, 24}, {11, 11, 22},
		{3, 4, 0}, {3, 4, MaxCapacity + 4}, {3, 4, 6},
	} {
		if _, ok := Slot(key[0], key[1], key[2]); ok {
			t.Errorf("wc=%d wr=%d capacity=%d was admitted", key[0], key[1], key[2])
		}
	}
}

func TestCatalogIsWellformed(t *testing.T) {
	if !Wellformed(GeneratorISO) {
		t.Error("the ISO catalog is missing or malformed")
	}
}

// TestPivotsMatchTheBuilder is the parity gate: a reconstructed transcript has
// to be what the sweep it replaced would have produced. The sample is fixed by
// seed so a failure reproduces, and it covers every row weight through the
// boundary keys; the exhaustive form of the same comparison is behind the
// jabcatalog_exhaustive tag because sweeping the whole space costs minutes.
func TestPivotsMatchTheBuilder(t *testing.T) {
	all := legalKeys()
	var sample [][3]int
	for wr := MinRowWeight; wr <= MaxRowWeight; wr++ {
		sample = append(sample,
			[3]int{MinColWeight, wr, wr},
			[3]int{MinColWeight, wr, MaxCapacity / wr * wr},
			[3]int{wr - 1, wr, MaxCapacity / wr * wr},
		)
	}
	random := rand.New(rand.NewPCG(1, 2))
	for range 60 {
		sample = append(sample, all[random.IntN(len(all))])
	}
	for _, key := range sample {
		if err := checkKey(GeneratorISO, key[0], key[1], key[2]); err != nil {
			t.Fatal(err)
		}
	}
}

// checkKey reports rather than fails, because the exhaustive form runs it from
// worker goroutines and only the test goroutine may call FailNow.
func checkKey(g Generator, wc, wr, capacity int) error {
	got, ok := Pivots(g, wc, wr, capacity)
	if !ok {
		return fmt.Errorf("wc=%d wr=%d capacity=%d is not in the catalog", wc, wr, capacity)
	}
	want := ecc.PivotSweep(wc, wr, capacity, g.Variant())
	if len(got) != len(want) {
		return fmt.Errorf("wc=%d wr=%d capacity=%d: %d pivots, want %d", wc, wr, capacity, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			return fmt.Errorf("wc=%d wr=%d capacity=%d row %d: pivot %d, want %d",
				wc, wr, capacity, i, got[i], want[i])
		}
	}
	return nil
}

// TestUnselectableGeneratorIsNotBuilt pins what keeps a catalog out of a build
// that cannot read it. The embedded build leaves the artifact out of the binary
// and the runtime build must not compute it either: Combined asks for both
// generators to assemble the device upload, so reachability through a decode is
// not what decides this.
func TestUnselectableGeneratorIsNotBuilt(t *testing.T) {
	if lcgSelectable {
		t.Skip("this build compiles a C-family wire variant")
	}
	if blob := Blob(GeneratorLCG); blob != nil {
		t.Fatalf("C-family catalog is %d bytes in a build that cannot select it", len(blob))
	}
	if Wellformed(GeneratorLCG) {
		t.Fatal("C-family catalog reports itself usable in a build that cannot select it")
	}
}
