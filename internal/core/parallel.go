package core

import (
	"runtime"
	"sync"
)

// minRowsPerBand keeps ParallelRows from splitting a small pixel loop into
// bands whose goroutine overhead outweighs the loop body.
const minRowsPerBand = 64

// ParallelRows splits the half-open row range [0, rows) into contiguous
// bands and runs fn on each band, concurrently when the range is large
// enough to pay for the goroutines. Each band's writes must stay inside its
// own rows; bands are disjoint, so the combined result is identical to the
// sequential loop regardless of scheduling.
func ParallelRows(rows int, fn func(lo, hi int)) {
	ParallelChunks(rows, minRowsPerBand, fn)
}

// ParallelChunks is ParallelRows with an explicit minimum chunk size, for
// loops whose iterations are much heavier than one pixel row (e.g. block
// rows). The same disjoint-writes contract applies.
func ParallelChunks(n, minPerChunk int, fn func(lo, hi int)) {
	workers := min(runtime.GOMAXPROCS(0), n/max(minPerChunk, 1))
	if workers <= 1 {
		fn(0, n)
		return
	}
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := range workers {
		lo := n * w / workers
		hi := n * (w + 1) / workers
		go func() {
			defer wg.Done()
			fn(lo, hi)
		}()
	}
	wg.Wait()
}
