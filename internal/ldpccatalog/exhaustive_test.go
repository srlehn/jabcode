//go:build jabcatalog_exhaustive

package ldpccatalog

import (
	"runtime"
	"sync"
	"testing"

	"github.com/srlehn/jabcode/internal/ecc"
)

// TestEveryKeyMatchesTheBuilder is the completeness and parity gate in full: it
// sweeps the whole legal key space and compares every pivot against the builder
// the catalog replaces. It is tagged because that is minutes of elimination, so
// it belongs in a deliberate run rather than in the baseline suite.
func TestEveryKeyMatchesTheBuilder(t *testing.T) {
	all := legalKeys()
	for _, g := range []Generator{GeneratorISO, GeneratorLCG} {
		if !Wellformed(g) {
			t.Fatalf("generator %d has no catalog compiled in", g)
		}
		work := make(chan [3]int, len(all))
		for _, key := range all {
			work <- key
		}
		close(work)
		failures := make(chan error, len(all))
		var wg sync.WaitGroup
		for range runtime.NumCPU() {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for key := range work {
					if err := checkKey(g, key[0], key[1], key[2]); err != nil {
						failures <- err
					}
				}
			}()
		}
		wg.Wait()
		close(failures)
		for err := range failures {
			t.Errorf("generator %d: %v", g, err)
		}
	}
}

// TestLeadingBlockNeedsNoStorage is the identity the record area relies on: the
// rows of the leading Gallager block have pairwise disjoint support, so none is
// modified before the sweep reaches it and each pivots at its own first column.
func TestLeadingBlockNeedsNoStorage(t *testing.T) {
	for _, g := range []Generator{GeneratorISO, GeneratorLCG} {
		for _, key := range legalKeys() {
			wc, wr, capacity := key[0], key[1], key[2]
			pivots := ecc.PivotSweep(wc, wr, capacity, g.Variant())
			for i := range capacity / wr {
				if int(pivots[i]) != ecc.LeadingPivot(i, wr) {
					t.Fatalf("generator %d wc=%d wr=%d capacity=%d row %d pivots at %d, want %d",
						g, wc, wr, capacity, i, pivots[i], ecc.LeadingPivot(i, wr))
				}
			}
		}
	}
}
