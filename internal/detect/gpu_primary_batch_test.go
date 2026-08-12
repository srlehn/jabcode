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
	for pattern := range 4 {
		base := gpuPrimaryResultPatterns + pattern*gpuFinderFoldPatternWords
		put(base, uint32(pattern))
		put(base+2, math.Float32bits(float32(10+pattern*20)))
		put(base+3, math.Float32bits(float32(15+pattern*20)))
		put(base+4, math.Float32bits(4))
		put(base+5, 1)
	}

	attempts, err := gpuPrimaryBatchResults(out, []wire.Variant{wire.ISO23634})
	if err != nil {
		t.Fatal(err)
	}
	if len(attempts) != 1 {
		t.Fatalf("attempts = %d, want 1", len(attempts))
	}
	if got := attempts[0]; got.Slot != 0 || got.Geometry != 0 ||
		got.Side.X != 21 || got.Side.Y != 21 || !got.Result.PayloadOK ||
		!got.Result.Evidence.Admitted() || got.AlignmentRetry {
		t.Fatalf("parsed attempt = %+v", got)
	}

	alignmentBase := gpuPrimaryResultDirectSlots * gpuPrimaryResultWords
	copy(out[alignmentBase*4:], out[:gpuPrimaryResultWords*4])
	put(alignmentBase+gpuPrimaryResultSlot, gpuPrimaryResultDirectSlots)
	attempts, err = gpuPrimaryBatchResults(out, []wire.Variant{wire.ISO23634})
	if err != nil {
		t.Fatal(err)
	}
	if len(attempts) != 2 {
		t.Fatalf("paired attempts = %d, want 2", len(attempts))
	}
	if got := attempts[1]; !got.AlignmentRetry || got.Slot != 0 || got.Geometry != 0 {
		t.Fatalf("parsed alignment attempt = %+v", got)
	}
}

func TestGPUPrimaryResultBatchBoundMatchesShader(t *testing.T) {
	if gpuPrimaryResultDirectSlots != 2*(1+gpuFinderDecisionMaxAlts) {
		t.Fatalf("direct slots = %d, want two interpretations of every geometry",
			gpuPrimaryResultDirectSlots)
	}
	if gpuPrimaryResultBatchSlots != 2*gpuPrimaryResultDirectSlots {
		t.Fatalf("batch slots = %d, want paired direct and alignment results",
			gpuPrimaryResultBatchSlots)
	}
	if gpuPrimaryResultControlPatterns+4*gpuFinderFoldPatternWords !=
		gpuPrimaryResultControlWords {
		t.Fatalf("primary result control holds an incomplete finder quad")
	}
	wants := map[string]uint32{
		"RESULT_HEADER_WORDS": gpuPrimaryResultHeaderWords,
		"RESULT_WORDS":        gpuPrimaryResultWords,
		"RESULT_BATCH_SLOTS":  gpuPrimaryResultBatchSlots,
	}
	for name, want := range wants {
		if got := wgslUintConstant(t, primaryResultWGSL, name); got != want {
			t.Errorf("%s = %d, want %d", name, got, want)
		}
	}
}
