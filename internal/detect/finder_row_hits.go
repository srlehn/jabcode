package detect

import (
	"cmp"
	"encoding/binary"
	"slices"
)

// finderRowHit is one raw run-length hit of the finder row scan, in the exact
// integer terms of the five-state machine, so the float centre and module
// size derive from it with the CPU scan's own float64 expressions.
type finderRowHit struct {
	y      int
	seq    int
	endPos int
	s2     int
	s3     int
	s4     int
	inside int
}

// center is the hit's scanline centre, the seekPatternHorizontal expression.
func (hit finderRowHit) center() float64 {
	return float64(hit.endPos-hit.s4-hit.s3) - float64(hit.s2)/2.0
}

// moduleSize is the hit's layer-size estimate, the checkPatternCross expression.
func (hit finderRowHit) moduleSize() float64 {
	return float64(hit.inside) / 3.0
}

// finderPassRowHits carries one prepared pass's device row-scan output: the
// per-channel raw hits in scan order. A pass that overflowed the record
// buffer is invalid and the consumer runs the CPU row walk instead.
type finderPassRowHits struct {
	channels    [3][]finderRowHit
	channelMask uint32
	valid       bool
}

// scanned reports whether the pass scanned the given channel on the device.
func (hits *finderPassRowHits) scanned(channel int) bool {
	return hits != nil && hits.valid && hits.channelMask&(1<<channel) != 0
}

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

// parseFinderScanRecords decodes the downloaded record buffer into per-channel
// hits ordered like the CPU row walk: ascending row, then scan order within
// the row. Device lanes append records unordered, so the order is restored
// here; a truncated (overflowed) buffer parses as invalid.
func parseFinderScanRecords(records []byte, channelMask uint32) *finderPassRowHits {
	hits := &finderPassRowHits{channelMask: channelMask}
	count := int(binary.LittleEndian.Uint32(records))
	if count > gpuFinderScanCapacity {
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
	hits.valid = true
	return hits
}
