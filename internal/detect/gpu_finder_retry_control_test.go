//go:build !js

package detect

import (
	"strings"
	"testing"
)

func TestGPUFinderRetryControlLayoutMatchesHost(t *testing.T) {
	constants := map[string]uint32{
		"STAGE_CAPTURE_AVERAGE_ROW":      gpuFinderRetryStageCaptureAverageRow,
		"STAGE_CAPTURE_AVERAGE_DECISION": gpuFinderRetryStageCaptureAverageDecision,
		"STAGE_CAPTURE_PASS_RESULT":      gpuFinderRetryStageCapturePassResult,
		"STAGE_CAPTURE_SURVIVORS":        gpuFinderRetryStageCaptureSurvivors,
		"STAGE_SELECT_AVERAGE":           gpuFinderRetryStageSelectAverage,
		"STAGE_SELECT_PITCH":             gpuFinderRetryStageSelectPitch,
		"AVERAGE_WORDS":                  gpuFinderAverageParamsSize / 4,
		"CONTROL_STAGE":                  gpuFinderRetryControlStage / 4,
		"CONTROL_MAX_SURVIVORS":          gpuFinderRetryControlMaxSurvivors / 4,
		"INDIRECT_AVERAGE":               gpuFinderRetryIndirectAverage / 4,
		"INDIRECT_REDUCE":                gpuFinderRetryIndirectReduce / 4,
		"INDIRECT_CANVAS":                gpuFinderRetryIndirectCanvas / 4,
		"INDIRECT_BLOCKS":                gpuFinderRetryIndirectBlocks / 4,
		"INDIRECT_PACK":                  gpuFinderRetryIndirectPack / 4,
		"INDIRECT_SCAN":                  gpuFinderRetryIndirectScan / 4,
		"INDIRECT_CHAIN":                 gpuFinderRetryIndirectChain / 4,
		"INDIRECT_PITCH_SAMPLES":         gpuFinderRetryIndirectPitchSample / 4,
		"INDIRECT_PITCH_ONE":             gpuFinderRetryIndirectPitchOne / 4,
		"INDIRECT_PITCH_CENTER":          gpuFinderRetryIndirectPitchCenter / 4,
		"INDIRECT_PITCH_ACF":             gpuFinderRetryIndirectPitchACF / 4,
		"INDIRECT_PITCH_SELECT":          gpuFinderRetryIndirectPitchSelect / 4,
		"BIN_SCAN_CAPACITY":              gpuBinarizerScanCapacityOffset / 4,
		"SCAN_CHANNELS":                  1 << currentFamilySeekChannel,
		"CHAIN_COMPACT_CAPACITY":         gpuRowCompactCapacity,
		"DECISION_DIAGNOSTIC":            gpuFinderDecisionDiagnostic,
		"DECISION_PASS_INPUT":            gpuFinderDecisionPassInput,
		"DIAGNOSTIC_ADMITTED_SHIFT":      gpuFinderDiagnosticAdmittedShift,
		"DIAGNOSTIC_LOCATED_SHIFT":       gpuFinderDiagnosticLocatedShift,
		"PREPARED_PASSES":                maxFinderPreparedPasses,
	}
	for name, want := range constants {
		if got := wgslUintConstant(t, finderRetryControlWGSL, name); got != want {
			t.Errorf("%s = %d, want %d", name, got, want)
		}
	}
	if gpuFinderRetryControlWords != gpuFinderRetryIndirectPitchSelect/4+3 {
		t.Fatalf(
			"finder retry control has %d words, want %d",
			gpuFinderRetryControlWords,
			gpuFinderRetryIndirectPitchSelect/4+3,
		)
	}
}

// TestGPUFinderDirectionalControlContinuesUntilConsistent pins what binding 1 of
// the directional control actually is: the indirect dispatch block the preceding
// decision writes, carrying [1,1,1] to continue and [0,0,0] once a quad settles.
//
// An earlier version of this test asserted the opposite expression and so
// codified a defect. Reading words 1 and 2 as the decision record's consistent
// and declined fields inverted the gate: an enabled [1,1,1] scanned zero lines
// and a stopped [0,0,0] scanned every line. Every rotated direction dispatched
// no workgroups, so the resident batch could not see a rotated symbol at all,
// and it reported authoritative no-finders rather than failing loudly.
func TestGPUFinderDirectionalControlContinuesUntilConsistent(t *testing.T) {
	if strings.Contains(finderDirectionalControlWGSL, "DECISION_CONSISTENT") ||
		strings.Contains(finderDirectionalControlWGSL, "DECISION_DECLINED") {
		t.Fatal("the directional gate reads decision record fields, not dispatch dimensions")
	}
	if !strings.Contains(finderDirectionalControlWGSL, "gate[0] != 0u") {
		t.Fatal("directional sweeps are not gated on the dispatch dimension the decision writes")
	}
}

func TestGPUFinderAssemblyAcceptsEmptyDeviceSource(t *testing.T) {
	if strings.Contains(finderCandidatesWGSL, "required == 0u") {
		t.Fatal("an empty device-counted direction is valid, not a missing source")
	}
	gate := "required > params[PARAM_REQUIRED_LIMIT] || resident_count > count"
	if !strings.Contains(finderCandidatesWGSL, gate) {
		t.Fatal("device-counted assembly is not bounded by producer and compacted capacities")
	}
}
