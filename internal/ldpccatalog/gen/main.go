// Command gen writes the precomputed pivot transcripts package ldpccatalog
// embeds. It is invoked through the go:generate directive in catalog.go and
// should otherwise be left alone: the artifacts it produces are checked in, and
// a rebuild is only interesting when the Gallager construction or the sweep
// changes, in which case a byte difference is the signal that it did.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/srlehn/jabcode/internal/ecc"
	"github.com/srlehn/jabcode/internal/ldpccatalog"
	"github.com/srlehn/jabcode/internal/wire"
)

// key is one code the catalog covers.
type key struct {
	wc, wr, capacity int
}

func main() {
	dir := flag.String("out", ".", "directory to write the catalog artifacts into")
	flag.Parse()

	for _, g := range []struct {
		generator ldpccatalog.Generator
		name      string
	}{
		{ldpccatalog.GeneratorISO, "iso.bin"},
		{ldpccatalog.GeneratorLCG, "lcg.bin"},
	} {
		blob, rows, err := build(g.generator)
		if err != nil {
			fmt.Fprintln(os.Stderr, "jabcode: "+err.Error())
			os.Exit(1)
		}
		at := filepath.Join(*dir, g.name)
		if err := os.WriteFile(at, blob, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "jabcode: "+err.Error())
			os.Exit(1)
		}
		fmt.Printf("%s: %d slots, %d stored pivots, %d bytes\n",
			g.name, ldpccatalog.SlotCount(), rows, len(blob))
	}
}

// keys enumerates the complete legal key space in slot order, so the directory
// is filled by construction rather than by sorting afterwards.
func keys() []key {
	all := make([]key, ldpccatalog.SlotCount())
	for wr := ldpccatalog.MinRowWeight; wr <= ldpccatalog.MaxRowWeight; wr++ {
		for wc := ldpccatalog.MinColWeight; wc < wr; wc++ {
			for capacity := wr; capacity <= ldpccatalog.MaxCapacity; capacity += wr {
				slot, ok := ldpccatalog.Slot(wc, wr, capacity)
				if !ok {
					panic(fmt.Sprintf("jabcode: legal key wc=%d wr=%d capacity=%d has no slot", wc, wr, capacity))
				}
				if all[slot] != (key{}) {
					panic(fmt.Sprintf("jabcode: slot %d is claimed twice", slot))
				}
				all[slot] = key{wc, wr, capacity}
			}
		}
	}
	return all
}

// build sweeps every key and packs the result. Keys are independent, so the
// sweeps run across the machine while the packing stays in slot order.
func build(g ldpccatalog.Generator) ([]byte, int, error) {
	all := keys()
	starts := make([]uint32, len(all))
	total := uint32(0)
	for slot, k := range all {
		starts[slot] = total
		total += uint32(ldpccatalog.StoredRows(k.wc, k.wr, k.capacity))
	}

	stored := make([]uint16, total)
	variant := g.Variant()
	var failed error
	var mu sync.Mutex
	var wg sync.WaitGroup
	work := make(chan int, len(all))
	for slot := range all {
		work <- slot
	}
	close(work)
	for range runtime.NumCPU() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for slot := range work {
				k := all[slot]
				if err := sweepInto(stored[starts[slot]:], k, variant); err != nil {
					mu.Lock()
					if failed == nil {
						failed = err
					}
					mu.Unlock()
					return
				}
			}
		}()
	}
	wg.Wait()
	if failed != nil {
		return nil, 0, failed
	}
	return pack(starts, stored), int(total), nil
}

// sweepInto runs one key's sweep and writes the rows the catalog keeps. The
// leading Gallager block is checked rather than stored: the reconstruction
// depends on those rows pivoting at their own first column, so a construction
// change that broke it has to fail here and not silently in a decode.
func sweepInto(into []uint16, k key, variant wire.Variant) error {
	pivots := ecc.PivotSweep(k.wc, k.wr, k.capacity, variant)
	blockRows := k.capacity / k.wr
	if len(pivots) != blockRows*k.wc {
		return fmt.Errorf("wc=%d wr=%d capacity=%d swept %d rows, want %d",
			k.wc, k.wr, k.capacity, len(pivots), blockRows*k.wc)
	}
	for i := range blockRows {
		if int(pivots[i]) != ecc.LeadingPivot(i, k.wr) {
			return fmt.Errorf("wc=%d wr=%d capacity=%d leading row %d pivots at %d, want %d",
				k.wc, k.wr, k.capacity, i, pivots[i], ecc.LeadingPivot(i, k.wr))
		}
	}
	for i, pivot := range pivots[blockRows:] {
		if pivot == ldpccatalog.SweepNone {
			into[i] = ldpccatalog.PivotNone
			continue
		}
		if int(pivot) >= k.capacity {
			return fmt.Errorf("wc=%d wr=%d capacity=%d row %d pivots outside the code at %d",
				k.wc, k.wr, k.capacity, blockRows+i, pivot)
		}
		into[i] = pivot
	}
	return nil
}

// pack lays down the header, the directory and the twelve-bit record area.
func pack(starts []uint32, stored []uint16) []byte {
	const pivotBits = 12
	words := 4 + len(starts) + (len(stored)*pivotBits+31)/32
	blob := make([]byte, words*4)
	binary.LittleEndian.PutUint32(blob[0:], 0x4A424C50)
	binary.LittleEndian.PutUint32(blob[4:], 1)
	binary.LittleEndian.PutUint32(blob[8:], uint32(len(starts)))
	binary.LittleEndian.PutUint32(blob[12:], uint32(len(stored)))
	for slot, start := range starts {
		binary.LittleEndian.PutUint32(blob[(4+slot)*4:], start)
	}

	base := (4 + len(starts)) * 4
	for i, pivot := range stored {
		bit := uint64(i) * pivotBits
		at := base + int(bit/32)*4
		value := uint64(pivot) << (bit % 32)
		binary.LittleEndian.PutUint32(blob[at:],
			binary.LittleEndian.Uint32(blob[at:])|uint32(value))
		if high := value >> 32; high != 0 {
			binary.LittleEndian.PutUint32(blob[at+4:],
				binary.LittleEndian.Uint32(blob[at+4:])|uint32(high))
		}
	}
	return blob
}
