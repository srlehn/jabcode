package detect

import (
	"encoding/binary"
	"math"
	"testing"
)

// writeRowSummaryChannel fills one channel's summary block the way
// finder_chain_row.wgsl does.
func writeRowSummaryChannel(summary []byte, channel, compacted, rawHits int) {
	block := channel * gpuRowSummaryWords
	put := func(index int, value uint32) {
		binary.LittleEndian.PutUint32(summary[(block+index)*4:], value)
	}
	put(gpuRowSummaryCompacted, uint32(compacted))
	put(gpuRowSummaryRawHits, uint32(rawHits))
}

// writeRowCompactedRecord fills one compacted record: the six words the fold
// reads, then the walk fields the host replay needs.
func writeRowCompactedRecord(compact []byte, channel, index, y, seq int) {
	base := (channel*gpuRowCompactCapacity + index) * gpuRowCompactWords * 4
	record := compact[base:]
	put := func(word int, value uint32) {
		binary.LittleEndian.PutUint32(record[word*4:], value)
	}
	put(0, chainFlagSurvivor)
	put(1, uint32(fp0))
	put(2, 1)
	put(3, math.Float32bits(float32(seq)))
	put(4, math.Float32bits(float32(y)))
	put(5, math.Float32bits(3))
	put(6, uint32(y))
	put(7, uint32(seq))
	put(8, 12) // endPos
	put(9, 3)  // s2
	put(10, 3) // s3
	put(11, 3) // s4
	put(12, 9) // inside
}

// TestFinderRowHitsStayResidentUntilRead pins the door the row fold depends on:
// a summarized pass reports what it compacted without fetching it, and the
// candidates cross exactly once, only for a consumer that reads them.
func TestFinderRowHitsStayResidentUntilRead(t *testing.T) {
	summary := make([]byte, gpuRowSummaryBytes)
	compact := make([]byte, gpuRowCompactBytes)
	writeRowSummaryChannel(summary, 1, 2, 40)
	// Written in the reverse of the walk's order, because device lanes append
	// unordered and the replay is what restores it.
	writeRowCompactedRecord(compact, 1, 0, 24, 1)
	writeRowCompactedRecord(compact, 1, 1, 12, 0)

	fetches := 0
	hits := parseFinderRowSummary(summary, 1<<1, 1<<1, func(channel, count int) ([]byte, bool) {
		fetches++
		if channel != 1 || count != 2 {
			t.Errorf("fetch(%d, %d), want fetch(1, 2)", channel, count)
		}
		return compact, true
	})
	if !hits.valid || !hits.scanned(1) || !hits.chained(1) {
		t.Fatal("summarized pass did not come back valid, scanned and chained")
	}
	if got := hits.compactedCount(1); got != 2 {
		t.Fatalf("compacted count = %d, want 2", got)
	}
	if summarized := hits.summary(1); summarized == nil || summarized.rawHits != 40 {
		t.Fatalf("summary = %+v, want one carrying 40 raw hits", summarized)
	}
	if fetches != 0 {
		t.Fatalf("reading the counters fetched %d times, want 0", fetches)
	}

	first := hits.hitsFor(1)
	if len(first) != 2 {
		t.Fatalf("hitsFor returned %d hits, want 2", len(first))
	}
	if first[0].y != 12 || first[1].y != 24 {
		t.Fatalf("hits arrived at rows %d and %d, want the walk's 12 then 24",
			first[0].y, first[1].y)
	}
	if outcome := hits.outcomes[first[0].rec]; outcome.flags&chainFlagSurvivor == 0 {
		t.Fatalf("hit at row %d lost its outcome record", first[0].y)
	}
	if again := hits.hitsFor(1); len(again) != 2 || fetches != 1 {
		t.Fatalf("second read returned %d hits after %d fetches, want 2 after 1",
			len(again), fetches)
	}
}

// TestFinderRowHitsFailedFetchReportsNoHits holds the failing fetch to reporting
// nothing rather than a short list: a consumer that acted on part of a channel
// would select from candidates the pass never fully saw, and nothing downstream
// could tell that from a genuine miss.
func TestFinderRowHitsFailedFetchReportsNoHits(t *testing.T) {
	summary := make([]byte, gpuRowSummaryBytes)
	writeRowSummaryChannel(summary, 1, 2, 40)
	hits := parseFinderRowSummary(summary, 1<<1, 1<<1, func(int, int) ([]byte, bool) {
		return nil, false
	})
	if got := hits.hitsFor(1); got != nil {
		t.Fatalf("a failed fetch reported %d hits, want none", len(got))
	}
	if hits.valid {
		t.Fatal("a failed fetch left the device pass authoritative")
	}
	if got := hits.hitsFor(1); got != nil {
		t.Fatalf("a retried failed fetch reported %d hits, want none", len(got))
	}
}

func TestFinderRowSummaryMaterializesOnlyAfterFoldDecline(t *testing.T) {
	summary := make([]byte, gpuRowSummaryBytes)
	compact := make([]byte, gpuRowCompactBytes)
	writeRowSummaryChannel(summary, 1, 1, 20)
	writeRowCompactedRecord(compact, 1, 0, 12, 0)

	materialized := 0
	hits := &finderPassRowHits{
		channelMask:     1 << 1,
		outcomeChannels: 1 << 1,
		outcomes:        []finderChainOutcome{},
		valid:           true,
		summaryResident: 1 << 1,
	}
	hits.materialize = func(target *finderPassRowHits) bool {
		materialized++
		parsed := parseFinderRowSummary(
			summary, 1<<1, 1<<1,
			func(int, int) ([]byte, bool) { return compact, true },
		)
		*target = *parsed
		return true
	}
	if !hits.summaryOnDevice(1) || hits.summary(1) != nil || materialized != 0 {
		t.Fatal("observing a resident summary materialized it")
	}
	got := hits.hitsFor(1)
	if len(got) != 1 || got[0].y != 12 || materialized != 1 {
		t.Fatalf("decline materialized %d hits at %+v after %d calls, want row 12 once",
			len(got), got, materialized)
	}
	if summarized := hits.summary(1); summarized == nil || summarized.rawHits != 20 {
		t.Fatalf("materialized summary = %+v, want 20 raw hits", summarized)
	}
}
