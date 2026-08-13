//go:build !js

package detect

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/wire"
)

func TestGPUPrimaryBatchParsesOneActiveFixedSlot(t *testing.T) {
	out := make([]byte, gpuPrimaryResultBatchWords*4)
	put := func(index int, value uint32) {
		binary.LittleEndian.PutUint32(out[index*4:], value)
	}
	put(0, gpuPrimaryResultMagic)
	put(1, gpuPrimaryResultVersion)
	put(gpuPrimaryResultMetaStatus, gpuMetadataStatusDefault)
	put(gpuPrimaryResultPayloadStatus, gpuPrimaryPayloadOK)
	put(gpuPrimaryResultMetaModules, 64)
	put(gpuPrimaryResultNC, 1)
	put(gpuPrimaryResultColors, 4)
	put(gpuPrimaryResultPaletteLen, uint32(4*spec.PaletteCopies(4)*3))
	put(gpuPrimaryResultNetBits, 1)
	put(gpuPrimaryResultPaletteSeparation, math.Float32bits(12))
	put(gpuPrimaryResultPaletteDisagreement, math.Float32bits(1))
	put(gpuPrimaryResultFixedAgreements, 20)
	put(gpuPrimaryResultFixedChecks, 20)
	put(gpuPrimaryResultEvidenceFlags, 1)
	put(gpuPrimaryResultPayload, 1)
	put(gpuPrimaryResultSideX, 21)
	put(gpuPrimaryResultSideY, 21)
	put(gpuPrimaryResultProfile, gpuMetadataProfileISO)
	put(gpuPrimaryResultSlot, 0)
	put(gpuPrimaryResultGeometry, 0)
	put(gpuPrimaryResultCorner, uint32(CornerFound))
	put(gpuPrimaryResultDegrees, 0)
	put(gpuPrimaryResultPrint, 1)
	put(gpuPrimaryResultBatchStatus+gpuFinderDecisionHave, 1)
	put(gpuPrimaryResultBatchStatus+gpuFinderDecisionConsistent, 1)
	put(gpuPrimaryResultBatchStatus+gpuFinderDecisionMode,
		4|gpuFinderDiagnosticGeometrySeen|gpuFinderDiagnosticGeometryValid|
			0x1f<<gpuFinderDiagnosticAdmittedShift|
			0x10<<gpuFinderDiagnosticLocatedShift)
	for pattern := range 4 {
		base := gpuPrimaryResultPatterns + pattern*gpuFinderFoldPatternWords
		put(base, uint32(pattern))
		put(base+2, math.Float32bits(float32(10+pattern*20)))
		put(base+3, math.Float32bits(float32(15+pattern*20)))
		put(base+4, math.Float32bits(4))
		put(base+5, 1)
	}

	attempts, state, err := gpuPrimaryBatchResults(out, []wire.Variant{wire.ISO23634})
	if err != nil {
		t.Fatal(err)
	}
	if !state.Have || !state.Consistent || state.Decline != 0 || state.Pass != 4 ||
		state.Geometry != gpuFinderDiagnosticGeometrySeen|gpuFinderDiagnosticGeometryValid ||
		state.Admitted != 0x1f || state.Located != 0x10 ||
		state.Slots != 1 ||
		state.MetadataFailed != 0 || state.Unsupported != 0 {
		t.Fatalf("active primary batch state = %+v", state)
	}
	if len(attempts) != 1 {
		t.Fatalf("attempts = %d, want 1", len(attempts))
	}
	if got := attempts[0]; got.Slot != 0 || got.Geometry != 0 ||
		got.Side.X != 21 || got.Side.Y != 21 || !got.Result.PayloadOK ||
		!got.Result.Evidence.Admitted() || got.AlignmentRetry || !got.PrintDetected {
		t.Fatalf("parsed attempt = %+v", got)
	}

	alignmentBase := gpuPrimaryResultDirectSlots * gpuPrimaryResultWords
	copy(out[alignmentBase*4:], out[:gpuPrimaryResultWords*4])
	put(alignmentBase+gpuPrimaryResultSlot, gpuPrimaryResultDirectSlots)
	attempts, state, err = gpuPrimaryBatchResults(out, []wire.Variant{wire.ISO23634})
	if err != nil {
		t.Fatal(err)
	}
	if !state.Have || !state.Consistent || state.Decline != 0 || state.Slots != 2 ||
		state.MetadataFailed != 0 || state.Unsupported != 0 {
		t.Fatalf("paired primary batch state = %+v", state)
	}
	if len(attempts) != 2 {
		t.Fatalf("paired attempts = %d, want 2", len(attempts))
	}
	if got := attempts[1]; !got.AlignmentRetry || got.Slot != 0 || got.Geometry != 0 {
		t.Fatalf("parsed alignment attempt = %+v", got)
	}
}

func TestGPUPrimaryBatchPreservesDeviceDecline(t *testing.T) {
	out := make([]byte, gpuPrimaryResultBatchBytes)
	binary.LittleEndian.PutUint32(
		out[(gpuPrimaryResultBatchStatus+gpuFinderDecisionDeclined)*4:],
		gpuFinderDeclineAssemblyDeferred,
	)
	attempts, state, err := gpuPrimaryBatchResults(out, []wire.Variant{wire.ISO23634})
	if err != nil {
		t.Fatal(err)
	}
	if len(attempts) != 0 || state.Decline != gpuFinderDeclineAssemblyDeferred {
		t.Fatalf("declined batch = (%d, %#x), want (0, %#x)",
			len(attempts), state.Decline, gpuFinderDeclineAssemblyDeferred)
	}
}

func TestGPUPrimaryBatchPreservesPassAdmissionWithoutFinder(t *testing.T) {
	out := make([]byte, gpuPrimaryResultBatchBytes)
	binary.LittleEndian.PutUint32(
		out[(gpuPrimaryResultBatchStatus+gpuFinderDecisionMode)*4:],
		0x2b<<gpuFinderDiagnosticAdmittedShift,
	)
	attempts, state, err := gpuPrimaryBatchResults(out, []wire.Variant{wire.ISO23634})
	if err != nil {
		t.Fatal(err)
	}
	if len(attempts) != 0 || state.Have || state.Pass != -1 ||
		state.Geometry != 0 || state.Admitted != 0x2b || state.Located != 0 {
		t.Fatalf("empty admitted batch = %+v with %d attempts", state, len(attempts))
	}
}

func TestGPUPrimaryResultBatchBoundMatchesShader(t *testing.T) {
	declineReasons := map[string]uint32{
		"DECLINE_ASSEMBLY_INVALID":        gpuFinderDeclineAssemblyInvalid,
		"DECLINE_ASSEMBLY_DEFERRED":       gpuFinderDeclineAssemblyDeferred,
		"DECLINE_FOLD_DROPPED":            gpuFinderDeclineFoldDropped,
		"DECLINE_FAMILY_POOL_DROPPED":     gpuFinderDeclineFamilyPoolDropped,
		"DECLINE_CONTEXTUAL_POOL_DROPPED": gpuFinderDeclineContextualPoolDropped,
	}
	for name, want := range declineReasons {
		if got := wgslUintConstant(t, finderDecisionWGSL, name); got != want {
			t.Errorf("%s = %#x, want %#x", name, got, want)
		}
	}
	diagnostics := map[string]uint32{
		"DIAGNOSTIC_GEOMETRY_SEEN":     gpuFinderDiagnosticGeometrySeen,
		"DIAGNOSTIC_SIDE_INVALID":      gpuFinderDiagnosticSideInvalid,
		"DIAGNOSTIC_FRAME_INVALID":     gpuFinderDiagnosticFrameInvalid,
		"DIAGNOSTIC_TRANSFORM_INVALID": gpuFinderDiagnosticTransformInvalid,
		"DIAGNOSTIC_MODULE_INVALID":    gpuFinderDiagnosticModuleInvalid,
		"DIAGNOSTIC_GEOMETRY_VALID":    gpuFinderDiagnosticGeometryValid,
	}
	for name, want := range diagnostics {
		if got := wgslUintConstant(t, finderGeometryWGSL, name); got != want {
			t.Errorf("%s = %#x, want %#x", name, got, want)
		}
	}
	if gpuFinderDeclineMask != gpuFinderDeclineAssemblyInvalid|
		gpuFinderDeclineAssemblyDeferred|gpuFinderDeclineFoldDropped|
		gpuFinderDeclineFamilyPoolDropped|gpuFinderDeclineContextualPoolDropped {
		t.Fatal("resident finder decline mask omits a reason")
	}
	if gpuPrimaryResultDirectSlots != 2*(1+gpuFinderDecisionMaxAlts) {
		t.Fatalf("direct slots = %d, want two interpretations of every geometry",
			gpuPrimaryResultDirectSlots)
	}
	if gpuPrimaryResultBatchSlots != 2*gpuPrimaryResultDirectSlots {
		t.Fatalf("batch slots = %d, want paired direct and alignment results",
			gpuPrimaryResultBatchSlots)
	}
	if gpuPrimaryResultBatchStatus != gpuPrimaryResultBatchSlots*gpuPrimaryResultWords ||
		gpuPrimaryBatchStateWords != gpuFinderDecisionPatterns ||
		gpuPrimaryResultBatchWords != gpuPrimaryResultBatchStatus+gpuPrimaryBatchStateWords {
		t.Fatalf("primary batch status is outside the fixed result download")
	}
	if gpuPrimaryResultControlPatterns+4*gpuFinderFoldPatternWords !=
		gpuPrimaryResultControlPrint ||
		gpuPrimaryResultControlPrint+1 != gpuPrimaryResultControlWords {
		t.Fatalf("primary result control holds an incomplete finder quad")
	}
	wants := map[string]uint32{
		"RESULT_HEADER_WORDS": gpuPrimaryResultHeaderWords,
		"RESULT_WORDS":        gpuPrimaryResultWords,
		"RESULT_BATCH_SLOTS":  gpuPrimaryResultBatchSlots,
		"RESULT_PRINT":        gpuPrimaryResultPrint,
	}
	for name, want := range wants {
		if got := wgslUintConstant(t, primaryResultWGSL, name); got != want {
			t.Errorf("%s = %d, want %d", name, got, want)
		}
	}
}

func TestGPUPrimaryPrintProvenanceLayoutMatchesShaders(t *testing.T) {
	wants := []struct {
		name   string
		source string
		want   uint32
	}{
		{"finder decision print", finderDecisionWGSL, gpuFinderDecisionPrint},
		{"finder geometry decision print", finderGeometryWGSL, gpuFinderDecisionPrint},
		{"finder geometry control print", finderGeometryWGSL, gpuPrimaryResultControlPrint},
		{"primary result control print", primaryResultWGSL, gpuPrimaryResultControlPrint},
	}
	names := []string{"DECISION_PRINT", "DECISION_PRINT", "CONTROL_PRINT", "CONTROL_PRINT"}
	for index, check := range wants {
		if got := wgslUintConstant(t, check.source, names[index]); got != check.want {
			t.Errorf("%s = %d, want %d", check.name, got, check.want)
		}
	}
	decisionWords := map[string]uint32{
		"DECISION_DIAGNOSTIC": uint32(gpuFinderDecisionDiagnostic),
		"DECISION_PASS_INPUT": uint32(gpuFinderDecisionPassInput),
	}
	for name, want := range decisionWords {
		if got := wgslUintConstant(t, finderDecisionWGSL, name); got != want {
			t.Errorf("finder decision %s = %d, want %d", name, got, want)
		}
	}
	if gpuFinderDecisionPrint+1 != gpuFinderDecisionDiagnostic ||
		gpuFinderDecisionDiagnostic+1 != gpuFinderDecisionPassInput ||
		gpuFinderDecisionPassInput+1 != gpuFinderDecisionWords {
		t.Fatalf("finder decision diagnostics are outside the retained record")
	}
}

func TestGPUPrimaryPrintOffsetControlMatchesResidentSearch(t *testing.T) {
	if gpuFinderGeometryIndirectWords != 12 ||
		gpuFinderGeometryOffsetSelectIndirectOffset != 9*4 {
		t.Fatalf("finder geometry does not reserve both print-offset dispatches")
	}
	checks := []struct {
		name   string
		source string
		want   uint32
	}{
		{"OFFSET_SCORE_SLOTS", finderGeometryWGSL, gpuChannelOffsetSlots},
		{"CANDIDATE_SIDE", channelOffsetSelectWGSL, gpuChannelOffsetSteps},
		{"NOMINAL", channelOffsetSelectWGSL, uint32(len(channelOffsetGrid) / 2 * (len(channelOffsetGrid) + 1))},
	}
	for _, check := range checks {
		if got := wgslUintConstant(t, check.source, check.name); got != check.want {
			t.Errorf("%s = %d, want %d", check.name, got, check.want)
		}
	}
}
