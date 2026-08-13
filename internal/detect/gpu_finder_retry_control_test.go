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

func TestGPUFinderDirectionalControlContinuesUntilConsistent(t *testing.T) {
	constants := map[string]uint32{
		"DECISION_CONSISTENT": gpuFinderDecisionConsistent,
		"DECISION_DECLINED":   gpuFinderDecisionDeclined,
	}
	for name, want := range constants {
		if got := wgslUintConstant(t, finderDirectionalControlWGSL, name); got != want {
			t.Errorf("%s = %d, want %d", name, got, want)
		}
	}
	gate := "decision[DECISION_CONSISTENT] == 0u && decision[DECISION_DECLINED] == 0u"
	if !strings.Contains(finderDirectionalControlWGSL, gate) {
		t.Fatal("directional retries are not gated by unresolved, trusted finder evidence")
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
