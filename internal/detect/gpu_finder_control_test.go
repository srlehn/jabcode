//go:build !js

package detect

import (
	"encoding/binary"
	"regexp"
	"strconv"
	"testing"
)

func wgslUintConstant(t *testing.T, source, name string) uint32 {
	t.Helper()
	match := regexp.MustCompile(`const ` + regexp.QuoteMeta(name) + `: u32 = ([0-9]+)u;`).FindStringSubmatch(source)
	if len(match) != 2 {
		t.Fatalf("WGSL constant %s is absent", name)
	}
	value, err := strconv.ParseUint(match[1], 10, 32)
	if err != nil {
		t.Fatalf("parse WGSL constant %s: %v", name, err)
	}
	return uint32(value)
}

func TestGPUFinderDirectionalControlConstants(t *testing.T) {
	wants := map[string]uint32{
		"DIRECTION_COUNT":       gpuFinderDirectionalBatchMax,
		"SCAN_CAPACITY":         gpuFinderDirectionalCapacity,
		"COMPACT_CAPACITY":      gpuFinderDirectionalCompactCapacity,
		"SCAN_SKIP_CROSS_CHECK": finderScanSkipCrossCheck,
	}
	chain := gpuFinderChainParams(1, 1, 1, false)
	wants["CHAIN_CLASSIFY_CURRENT"] = binary.LittleEndian.Uint32(chain[16:])
	wants["CHAIN_CLASSIFY_BSI"] = binary.LittleEndian.Uint32(chain[20:])
	wants["CHAIN_CROSS_COLOR_BITS"] = binary.LittleEndian.Uint32(chain[24:])
	for name, want := range wants {
		if got := wgslUintConstant(t, finderDirectionalControlWGSL, name); got != want {
			t.Errorf("%s = %d, want %d", name, got, want)
		}
	}
}

func TestGPUPrimaryBinarizerControlConstants(t *testing.T) {
	chain := gpuFinderChainParams(1, 1, 1, false)
	wants := map[string]uint32{
		"BLOCK_DIVISOR":          binThresholdDivisor,
		"BLOCK_MIN":              binMinBlock,
		"BLOCK_MAX":              binMaxBlock,
		"SCAN_CHANNELS":          1 << currentFamilySeekChannel,
		"CHAIN_CLASSIFY_CURRENT": binary.LittleEndian.Uint32(chain[16:]),
		"CHAIN_CLASSIFY_BSI":     binary.LittleEndian.Uint32(chain[20:]),
		"CHAIN_CROSS_COLOR_BITS": binary.LittleEndian.Uint32(chain[24:]),
		"CHAIN_COMPACT_CAPACITY": gpuRowCompactCapacity,
		"CHAIN_ROW_STRIDE_SHIFT": gpuChainFlagStrideShift,
	}
	for name, want := range wants {
		if got := wgslUintConstant(t, binarizerPrimaryControlWGSL, name); got != want {
			t.Errorf("%s = %d, want %d", name, got, want)
		}
	}
}

func TestGPUFinderFoldControlConstants(t *testing.T) {
	wants := map[string]uint32{
		"ROW_CHANNEL":                 currentFamilySeekChannel,
		"ROW_CAPACITY":                gpuRowCompactCapacity,
		"ROW_STRIDE":                  gpuRowCompactWords,
		"ROW_SUMMARY_WORDS":           gpuRowSummaryWords,
		"ROW_SUMMARY_COMPACTED":       gpuRowSummaryCompacted,
		"ROW_SUMMARY_OVERFLOW":        gpuRowSummaryOverflow,
		"DIRECTION_CAPACITY":          gpuFinderDirectionalCompactCapacity,
		"DIRECTION_STRIDE":            gpuFinderChainOutcomeWords,
		"DIRECTION_SUMMARY_WORDS":     gpuFinderDirectionalSummaryWords,
		"DIRECTION_SUMMARY_COMPACTED": gpuFinderDirectionalSummaryCompacted,
		"DIRECTION_SUMMARY_REQUIRED":  gpuFinderDirectionalSummaryRequired,
		"DIRECTION_RECORD_CAPACITY":   gpuFinderDirectionalCapacity,
		"FAMILY_POOL_CAPACITY":        gpuFinderFamilyPoolSlots,
		"PATTERN_STOP":                maxFinderPatterns - 1,
		"CONTEXTUAL_CAPACITY":         maxContextualFinderSeeds,
		"MIN_CONTEXTUAL_FOUND":        minFinderCrossings,
	}
	for name, want := range wants {
		if got := wgslUintConstant(t, finderFoldControlWGSL, name); got != want {
			t.Errorf("%s = %d, want %d", name, got, want)
		}
	}
}

func TestGPUResidentBinarizerParamsCarryRowStride(t *testing.T) {
	params, _, _ := gpuResidentBinarizerParams(17, 19, nil, false, 7)
	if got := binary.LittleEndian.Uint32(params[36:]); got != 7 {
		t.Fatalf("row stride = %d, want 7", got)
	}
}
