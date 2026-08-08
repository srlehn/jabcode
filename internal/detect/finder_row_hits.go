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

func parseFinderChainOutcome(slot []byte) finderChainOutcome {
	single := func(at int) float64 {
		return float64(math.Float32frombits(binary.LittleEndian.Uint32(slot[at:])))
	}
	return finderChainOutcome{
		flags:      binary.LittleEndian.Uint32(slot),
		typ:        int(binary.LittleEndian.Uint32(slot[4:])),
		direction:  int(int32(binary.LittleEndian.Uint32(slot[8:]))),
		centerX:    single(12),
		centerY:    single(16),
		moduleSize: single(20),
	}
}

// Device finder records use this fixed byte layout. The CPU replay parser and
// the native GPU producer share these constants so platform selection cannot
// change record interpretation.
const (
	gpuFinderScanRecordWords   = 8
	gpuFinderScanHeaderBytes   = 16
	gpuFinderChainOutcomeWords = 6
)

// Outcome flag bits, mirroring the per-hit stat counters of the CPU chain.
// The colour bits are the device's verdict on the source-level signal an FP1 or
// FP2 candidate must show: evaluated says the kernel had the balanced image and
// decided, ok says the candidate passed. A hit without the evaluated bit is one
// the host still has to judge for itself.
const (
	chainFlagBranchBlue     = 1 << 0
	chainFlagBranchRed      = 1 << 1
	chainFlagRedColor       = 1 << 2
	chainFlagRedClassified  = 1 << 3
	chainFlagSurvivor       = 1 << 4
	chainFlagContextualSeed = 1 << 5
	chainFlagColorEvaluated = 1 << 6
	chainFlagColorOK        = 1 << 7
)

// gpuFinderChainFlagColorSource is the parameter bit that tells the chain
// kernel its colour source is bound.
const gpuFinderChainFlagColorSource = 1 << 1

const (
	// The row chain folds every hit into one counter block per scan channel and
	// carries back only the candidates the host can act on, so a pass costs a
	// summary and a short list instead of every raw record and every outcome.
	// The module-size histogram is not among them: it accumulates in the shared
	// seed histogram across every scan of a locate, because its one consumer
	// reads it once and a per-channel copy of a thousand buckets was most of
	// what this block cost.
	gpuRowSummaryWords    = 7
	gpuRowSummaryChannels = 3
	gpuRowSummaryBytes    = gpuRowSummaryChannels * gpuRowSummaryWords * 4

	// gpuRowCompactCapacity bounds one channel's compacted candidates. A pass
	// retains at most maxFinderPatterns survivors and maxContextualFinderSeeds
	// weak candidates, so a longer list could not be acted on in full; a channel
	// that exceeds it says so and the consumer reads the raw records instead.
	gpuRowCompactCapacity = maxFinderPatterns + maxContextualFinderSeeds
	gpuRowCompactWords    = 13
	gpuRowCompactBytes    = gpuRowSummaryChannels * gpuRowCompactCapacity * gpuRowCompactWords * 4

	// Summary word offsets within a channel's block, matching
	// finder_chain_row.wgsl.
	gpuRowSummaryCompacted     = 0
	gpuRowSummaryRawHits       = 1
	gpuRowSummaryBranchBlue    = 2
	gpuRowSummaryBranchRed     = 3
	gpuRowSummaryRedColor      = 4
	gpuRowSummaryRedClassified = 5
	gpuRowSummaryOverflow      = 6
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

	// summaries holds a channel's device-folded counters when the pass came
	// back as a summary and a compacted candidate list rather than as every raw
	// record. A summarized channel's hits carry no per-hit counter work,
	// because the counters already describe every hit the walk would have seen.
	summaries [3]*finderDirSummary
}

// summary returns the channel's device-folded counters, or nil when the pass
// handed back its raw records and the consumer must count them itself.
func (hits *finderPassRowHits) summary(channel int) *finderDirSummary {
	if hits == nil || !hits.valid || channel < 0 || channel >= len(hits.summaries) {
		return nil
	}
	return hits.summaries[channel]
}

// parseFinderRowSummary restores a pass whose chain folded its counters on the
// device and compacted the candidates the consumer can act on. Ordering is the
// walk's own: every compacted record carries the row and sequence the host
// sorts by, so a replayed list is the same sequence the raw records would have
// produced after their sort.
func parseFinderRowSummary(summaryBytes, compact []byte, channelMask, covered uint32) *finderPassRowHits {
	hits := &finderPassRowHits{
		channelMask:     channelMask,
		outcomeChannels: covered,
		outcomes:        []finderChainOutcome{},
	}
	for channel := range gpuRowSummaryChannels {
		if channelMask&(1<<channel) == 0 {
			continue
		}
		if covered&(1<<channel) == 0 {
			return &finderPassRowHits{channelMask: channelMask}
		}
		block := channel * gpuRowSummaryWords
		word := func(index int) int {
			return int(binary.LittleEndian.Uint32(summaryBytes[(block+index)*4:]))
		}
		count := word(gpuRowSummaryCompacted)
		hits.summaries[channel] = &finderDirSummary{
			rawHits:       word(gpuRowSummaryRawHits),
			branchBlue:    word(gpuRowSummaryBranchBlue),
			branchRed:     word(gpuRowSummaryBranchRed),
			redColor:      word(gpuRowSummaryRedColor),
			redClassified: word(gpuRowSummaryRedClassified),
		}
		if count == 0 {
			continue
		}
		hits.channels[channel] = make([]finderRowHit, count)
		for index := range count {
			base := (channel*gpuRowCompactCapacity + index) * gpuRowCompactWords * 4
			if base+gpuRowCompactWords*4 > len(compact) {
				return &finderPassRowHits{channelMask: channelMask}
			}
			record := compact[base:]
			hits.channels[channel][index] = finderRowHit{
				y:      int(binary.LittleEndian.Uint32(record[24:])),
				seq:    int(binary.LittleEndian.Uint32(record[28:])),
				endPos: int(binary.LittleEndian.Uint32(record[32:])),
				s2:     int(binary.LittleEndian.Uint32(record[36:])),
				s3:     int(binary.LittleEndian.Uint32(record[40:])),
				s4:     int(binary.LittleEndian.Uint32(record[44:])),
				inside: int(binary.LittleEndian.Uint32(record[48:])),
				rec:    len(hits.outcomes),
			}
			hits.outcomes = append(hits.outcomes, parseFinderChainOutcome(record))
		}
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
			hits.outcomes[index] = parseFinderChainOutcome(slot)
		}
	}
	hits.valid = true
	return hits
}
