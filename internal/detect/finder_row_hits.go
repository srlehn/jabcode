package detect

import (
	"cmp"
	"encoding/binary"
	"math"
	"slices"
)

// finderRowHit is one raw run-length hit of the finder row scan, in the exact
// integer terms of the five-state machine, so the float centre and module
// size derive from it with the CPU scan's own float64 expressions. rec is the
// hit's slot in the device record buffer, which is also its slot in the
// chain-outcome buffer.
type finderRowHit struct {
	y      int
	seq    int
	endPos int
	s2     int
	s3     int
	s4     int
	inside int
	rec    int
}

// center is the hit's scanline centre, the seekPatternHorizontal expression.
func (hit finderRowHit) center() float64 {
	return float64(hit.endPos-hit.s4-hit.s3) - float64(hit.s2)/2.0
}

// moduleSize is the hit's layer-size estimate, the checkPatternCross expression.
func (hit finderRowHit) moduleSize() float64 {
	return float64(hit.inside) / 3.0
}

// finderChainOutcome is one raw hit's device cross-check chain outcome: the
// per-hit stat flags and, for a surviving hit, the refined finder pattern in
// the CPU chain's exact float64 values.
type finderChainOutcome struct {
	flags      uint32
	typ        int
	direction  int
	centerX    float64
	centerY    float64
	moduleSize float64
}

// Device finder records use this fixed byte layout. The CPU replay parser and
// the native GPU producer share these constants so platform selection cannot
// change record interpretation.
const (
	gpuFinderScanRecordWords   = 8
	gpuFinderScanHeaderBytes   = 16
	gpuFinderChainOutcomeWords = 10
)

// Outcome flag bits, mirroring the per-hit stat counters of the CPU chain.
const (
	chainFlagBranchBlue    = 1 << 0
	chainFlagBranchRed     = 1 << 1
	chainFlagRedColor      = 1 << 2
	chainFlagRedClassified = 1 << 3
	chainFlagSurvivor      = 1 << 4
)

// finderPassRowHits carries one prepared pass's device row-scan output: the
// per-channel raw hits in scan order and the per-record chain outcomes of
// the channels whose chain kernel ran (a pass before the background kernel
// compilation finishes has none, and the consumer runs the bit-identical CPU
// per-hit chain instead). A pass that overflowed the record buffer is
// invalid and the consumer runs the CPU row walk.
type finderPassRowHits struct {
	channels        [3][]finderRowHit
	outcomes        []finderChainOutcome
	channelMask     uint32
	outcomeChannels uint32
	valid           bool
}

// scanned reports whether the pass scanned the given channel on the device.
func (hits *finderPassRowHits) scanned(channel int) bool {
	return hits != nil && hits.valid && hits.channelMask&(1<<channel) != 0
}

// chained reports whether the pass also ran the given channel's cross-check
// chain on the device, making its outcome records authoritative.
func (hits *finderPassRowHits) chained(channel int) bool {
	return hits.scanned(channel) && hits.outcomes != nil &&
		hits.outcomeChannels&(1<<channel) != 0
}

// bsiFamilyFinderCoreColors are the default-palette color indexes of the four
// BSI TR-03137 primary finder cores. The table lives untagged because the
// chain kernel parameter block always carries both classification tables;
// the BSI chain kernel itself is compiled in only by the BSI decoder tags.
var bsiFamilyFinderCoreColors = [4]int{1, 2, 5, 6}

// finderScanChannelMask maps the requested finder families to the channels
// their row scans seed on: the current family seeks on green, the BSI-era
// family on red.
func finderScanChannelMask(wantCurrent, wantBSI bool) uint32 {
	var mask uint32
	if wantCurrent {
		mask |= 1 << 1
	}
	if wantBSI {
		mask |= 1 << 0
	}
	return mask
}

// parseFinderScanRecords decodes the downloaded record and chain-outcome
// buffers into per-channel hits ordered like the CPU row walk: ascending row,
// then scan order within the row. Device lanes append records unordered, so
// the order is restored here; a truncated (overflowed) buffer parses as
// invalid. chainOutcomes is nil when no chain kernel ran this pass.
func parseFinderScanRecords(records, chainOutcomes []byte, channelMask, chainChannels uint32) *finderPassRowHits {
	hits := &finderPassRowHits{channelMask: channelMask, outcomeChannels: chainChannels}
	count := int(binary.LittleEndian.Uint32(records))
	if count > (len(records)-gpuFinderScanHeaderBytes)/(gpuFinderScanRecordWords*4) {
		return hits
	}
	var perChannel [3]int
	for index := range count {
		record := records[gpuFinderScanHeaderBytes+index*gpuFinderScanRecordWords*4:]
		channel := int(binary.LittleEndian.Uint32(record))
		if channel > 2 {
			return hits
		}
		perChannel[channel]++
	}
	for channel, n := range perChannel {
		if n > 0 {
			hits.channels[channel] = make([]finderRowHit, 0, n)
		}
	}
	for index := range count {
		record := records[gpuFinderScanHeaderBytes+index*gpuFinderScanRecordWords*4:]
		channel := int(binary.LittleEndian.Uint32(record))
		hits.channels[channel] = append(hits.channels[channel], finderRowHit{
			y:      int(binary.LittleEndian.Uint32(record[4:])),
			seq:    int(binary.LittleEndian.Uint32(record[8:])),
			endPos: int(binary.LittleEndian.Uint32(record[12:])),
			s2:     int(binary.LittleEndian.Uint32(record[16:])),
			s3:     int(binary.LittleEndian.Uint32(record[20:])),
			s4:     int(binary.LittleEndian.Uint32(record[24:])),
			inside: int(binary.LittleEndian.Uint32(record[28:])),
			rec:    index,
		})
	}
	for channel := range hits.channels {
		slices.SortFunc(hits.channels[channel], func(a, b finderRowHit) int {
			if c := cmp.Compare(a.y, b.y); c != 0 {
				return c
			}
			return cmp.Compare(a.seq, b.seq)
		})
	}
	if count > 0 && chainOutcomes != nil {
		hits.outcomes = make([]finderChainOutcome, count)
		for index := range count {
			slot := chainOutcomes[index*gpuFinderChainOutcomeWords*4:]
			hits.outcomes[index] = finderChainOutcome{
				flags:     binary.LittleEndian.Uint32(slot),
				typ:       int(binary.LittleEndian.Uint32(slot[4:])),
				direction: int(int32(binary.LittleEndian.Uint32(slot[8:]))),
				centerX: math.Float64frombits(
					uint64(binary.LittleEndian.Uint32(slot[12:]))<<32 |
						uint64(binary.LittleEndian.Uint32(slot[16:]))),
				centerY: math.Float64frombits(
					uint64(binary.LittleEndian.Uint32(slot[20:]))<<32 |
						uint64(binary.LittleEndian.Uint32(slot[24:]))),
				moduleSize: math.Float64frombits(
					uint64(binary.LittleEndian.Uint32(slot[28:]))<<32 |
						uint64(binary.LittleEndian.Uint32(slot[32:]))),
			}
		}
	}
	hits.valid = true
	return hits
}
