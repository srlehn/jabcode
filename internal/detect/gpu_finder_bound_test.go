//go:build !js

package detect

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestGPUFinderFamilyPoolCoversCompleteLocate(t *testing.T) {
	const oldSlots = 8192
	want := gpuFinderFamilyPoolMaxShares * (maxFinderPatterns - 1)
	if gpuFinderFamilyPoolSlots != want {
		t.Fatalf("family pool slots = %d, want complete locate bound %d",
			gpuFinderFamilyPoolSlots, want)
	}
	if want != 20958 {
		t.Fatalf("complete locate bound = %d, want 20958", want)
	}
	if got := len(descreenSchedule(4, 4, 100)); got != maxFinderDescreenPasses {
		t.Fatalf("maximum descreen schedule has %d passes, bound says %d",
			got, maxFinderDescreenPasses)
	}
	if len(scanDirections) != finderScanDirectionCount {
		t.Fatalf("scan directions = %d, bound says %d",
			len(scanDirections), finderScanDirectionCount)
	}
	if gpuFinderDirectionalBatchMax != finderScanDirectionCount {
		t.Fatalf("resident locate batch directions = %d, want column plus retries %d",
			gpuFinderDirectionalBatchMax, finderScanDirectionCount)
	}
	if gpuFinderPoolSharesPerPass != 1+gpuFinderDirectionalBatchMax {
		t.Fatalf("finder pool shares per pass = %d, want row plus batch %d",
			gpuFinderPoolSharesPerPass, 1+gpuFinderDirectionalBatchMax)
	}
	delta := (gpuFinderFamilyPoolSlots - oldSlots) * gpuFinderFoldPatternWords * 4
	if delta != 306384 {
		t.Fatalf("family pool allocation delta = %d bytes, want 306384", delta)
	}
	var resident gpuResidentBinarizer
	for fold := 0; fold < gpuFinderFamilyPoolMaxShares; fold++ {
		if !resident.claimFinderPoolShare() {
			t.Fatalf("family pool declined fold %d inside the proven bound", fold)
		}
	}
	if resident.claimFinderPoolShare() {
		t.Fatal("family pool accepted a fold beyond the proven bound")
	}

	// The first entry the old allocation could not hold must remain a valid
	// resident record. This checks the host boundary without executing a GPU.
	count := oldSlots + 1
	record := make([]byte, gpuFinderPoolWords*4)
	binary.LittleEndian.PutUint32(record[gpuFinderPoolCount*4:], uint32(count))
	pool := make([]byte, count*gpuFinderFoldPatternWords*4)
	at := (count - 1) * gpuFinderFoldPatternWords
	binary.LittleEndian.PutUint32(pool[at*4:], 3)
	binary.LittleEndian.PutUint32(pool[(at+1)*4:], math.MaxUint32)
	binary.LittleEndian.PutUint32(pool[(at+2)*4:], math.Float32bits(123.5))
	binary.LittleEndian.PutUint32(pool[(at+3)*4:], math.Float32bits(456.5))
	binary.LittleEndian.PutUint32(pool[(at+4)*4:], math.Float32bits(7.25))
	binary.LittleEndian.PutUint32(pool[(at+5)*4:], 9)

	entries, dropped, _, err := parseGPUFinderPool(
		record, pool, gpuFinderFamilyPoolSlots,
	)
	if err != nil {
		t.Fatalf("parse former-overflow pool: %v", err)
	}
	if dropped != 0 || len(entries) != count {
		t.Fatalf("former-overflow pool parsed %d entries with %d drops, want %d and zero",
			len(entries), dropped, count)
	}
	last := entries[len(entries)-1]
	if last.Typ != 3 || last.direction != -1 || last.Center.X != 123.5 ||
		last.Center.Y != 456.5 || last.ModuleSize != 7.25 || last.FoundCount != 9 {
		t.Fatalf("former-overflow sentinel changed: %+v", last)
	}
}
