package read

import (
	"image"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/wire"
)

func admittedBatchEvidence(payloadInitial uint32) core.PrimaryEvidence {
	return core.PrimaryEvidence{
		Available:           true,
		MetadataExplicit:    true,
		PaletteSeparation:   12,
		PaletteDisagreement: 1,
		PayloadInitial:      payloadInitial,
	}
}

func batchCandidate(variant wire.Variant, payload string, evidence core.PrimaryEvidence) gpuPrimaryMessageCandidate {
	return gpuPrimaryMessageCandidate{
		message: &Message{
			Variant:            variant,
			Data:               []byte(payload),
			ReaderTransmission: append([]byte("]j1"), payload...),
		},
		evidence: evidence,
	}
}

func TestSelectGPUPrimaryMessageAcceptsAgreementWithoutRanking(t *testing.T) {
	first := batchCandidate(wire.ISO23634, "payload", admittedBatchEvidence(5))
	second := batchCandidate(wire.ISO23634, "payload", admittedBatchEvidence(3))
	winner, ambiguous := selectGPUPrimaryMessage([]gpuPrimaryMessageCandidate{first, second})
	if ambiguous || winner != 0 {
		t.Fatalf("agreement selected (%d, %t), want first safe message", winner, ambiguous)
	}
}

func TestEmptyGPUPrimaryBatchIsDecisiveNoFinders(t *testing.T) {
	d := &detect.PrimaryDetector{}
	message, stage, evidence, decisive := decodeGPUPrimaryBatch(
		d, nil, nil, wire.ISO23634.Mask(),
	)
	if message != nil || stage != readNoFinders || evidence || !decisive {
		t.Fatalf(
			"empty batch = (%v, %v, %t, %t), want decisive no-finders",
			message, stage, evidence, decisive,
		)
	}
}

// TestEmptyGPUPrimaryBatchKeepsCurrentFamily pins what a decisive batch is
// allowed to narrow when BSI is compiled beside the current family. A batch
// that returned no attempt never weighed the current family, so narrowing on it
// would remove that family from the level's own locate and leave the symbol to
// the region search; a batch that returned attempts did weigh it.
func TestEmptyGPUPrimaryBatchKeepsCurrentFamily(t *testing.T) {
	capabilities := wire.ISO23634.Mask() | wire.BSI.Mask()
	wanted := finderFamiliesForCapabilities(capabilities)
	if !wanted.Has(detect.FinderFamilyCurrent) || !wanted.Has(detect.FinderFamilyBSI) {
		t.Fatalf("case needs both finder families wanted, got %#x", wanted)
	}
	gotWanted, gotRemaining := narrowAfterCurrentBatch(wanted, capabilities, 0)
	if gotWanted != wanted || gotRemaining != capabilities {
		t.Fatalf(
			"empty batch narrowed to (%#x, %#x), want (%#x, %#x) unchanged",
			gotWanted, gotRemaining, wanted, capabilities,
		)
	}
	gotWanted, gotRemaining = narrowAfterCurrentBatch(wanted, capabilities, 1)
	if gotWanted.Has(detect.FinderFamilyCurrent) || !gotWanted.Has(detect.FinderFamilyBSI) {
		t.Fatalf("batch with attempts left families %#x, want BSI alone", gotWanted)
	}
	if gotRemaining.Has(wire.ISO23634) {
		t.Fatalf("batch with attempts left capabilities %#x, want the current wire dropped", gotRemaining)
	}
}

func TestDefaultGPUPrimaryFailureRetainsAlignmentFallback(t *testing.T) {
	attempt := detect.PrimaryBatchAttempt{
		Variant: wire.ISO23634,
		Side:    image.Pt(spec.VersionToSize(6), spec.VersionToSize(6)),
		Result: core.PrimaryDeviceResult{
			Metadata: core.PrimaryMetadata{
				Defaulted: true,
				NC:        spec.DefaultModuleColorMode,
				Colors:    1 << (spec.DefaultModuleColorMode + 1),
				Palette: make([]byte,
					(1<<(spec.DefaultModuleColorMode+1))*
						spec.PaletteCopies(1<<(spec.DefaultModuleColorMode+1))*3),
			},
			PayloadOK: false,
			Evidence: core.PrimaryEvidence{
				Available: true,
			},
		},
	}
	message, stage, evidence, decisive := decodeGPUPrimaryBatch(
		&detect.PrimaryDetector{}, []detect.PrimaryBatchAttempt{attempt}, nil,
		wire.ISO23634.Mask(),
	)
	if message != nil || stage != readSampled || !evidence || decisive {
		t.Fatalf(
			"default failed batch = (%v, %v, %t, %t), want non-decisive sampled evidence",
			message, stage, evidence, decisive,
		)
	}
}

func TestSelectGPUPrimaryMessageRequiresStrictDominance(t *testing.T) {
	weak := admittedBatchEvidence(4)
	weak.PaletteSeparation = 11
	first := batchCandidate(wire.ISO23634, "first", admittedBatchEvidence(5))
	conflict := batchCandidate(wire.ISO23634, "conflict", weak)
	if winner, ambiguous := selectGPUPrimaryMessage(
		[]gpuPrimaryMessageCandidate{first, conflict},
	); !ambiguous || winner != -1 {
		t.Fatalf("incomparable messages selected (%d, %t)", winner, ambiguous)
	}

	dominant := batchCandidate(wire.ISO23634, "dominant", admittedBatchEvidence(3))
	if winner, ambiguous := selectGPUPrimaryMessage(
		[]gpuPrimaryMessageCandidate{first, dominant},
	); ambiguous || winner != 1 {
		t.Fatalf("dominant message selected (%d, %t), want candidate 1", winner, ambiguous)
	}
}

func TestSelectGPUPrimaryMessageTreatsCrossWireParseAsConflict(t *testing.T) {
	evidence := admittedBatchEvidence(5)
	iso := batchCandidate(wire.ISO23634, "same-bits", evidence)
	current := batchCandidate(wire.CurrentC, "same-bits", evidence)
	if winner, ambiguous := selectGPUPrimaryMessage(
		[]gpuPrimaryMessageCandidate{iso, current},
	); !ambiguous || winner != -1 {
		t.Fatalf("cross-wire parse selected (%d, %t), want ambiguity", winner, ambiguous)
	}
}
