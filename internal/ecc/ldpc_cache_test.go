package ecc

import "testing"

// TestSystematicEntryLRUKeepsHotKey floods the parity-matrix cache with more
// distinct keys than it holds while periodically re-requesting one hot key,
// and asserts the hot key stays resident while the flood evicts only among
// itself. This pins the eviction policy: a camera stream's real code,
// requested every frame, must not be crowded out by junk keys minted from
// garbage detections.
func TestSystematicEntryLRUKeepsHotKey(t *testing.T) {
	sysMu.Lock()
	saved := sysCache
	sysCache = map[sysKey]*sysEntry{}
	sysMu.Unlock()
	t.Cleanup(func() {
		sysMu.Lock()
		sysCache = saved
		sysMu.Unlock()
	})

	const wc, wr = 3, 4
	hot := sysKey{wc: wc, wr: wr, capacity: 24, encode: false}
	systematicEntry(hot.wc, hot.wr, hot.capacity, false)

	for i := range sysCacheMax + 16 {
		systematicEntry(wc, wr, 28+4*i, false)
		if i%8 == 0 {
			systematicEntry(hot.wc, hot.wr, hot.capacity, false)
		}
	}

	sysMu.Lock()
	defer sysMu.Unlock()
	if len(sysCache) > sysCacheMax {
		t.Fatalf("cache holds %d entries, cap is %d", len(sysCache), sysCacheMax)
	}
	if _, ok := sysCache[hot]; !ok {
		t.Fatalf("hot key %+v evicted by the junk flood", hot)
	}
}
