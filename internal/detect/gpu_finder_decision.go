//go:build !js

package detect

import (
	"fmt"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/phaseprobe"
	"github.com/srlehn/jabcode/internal/wire"
)

const (
	gpuFinderDecisionHave         = 0
	gpuFinderDecisionConsistent   = 1
	gpuFinderDecisionDeclined     = 2
	gpuFinderDecisionMissing      = 3
	gpuFinderDecisionCornerSource = 4
	gpuFinderDecisionCornerMiss   = 5
	gpuFinderDecisionAlternatives = 6
	gpuFinderDecisionMode         = 7
	gpuFinderDecisionPatterns     = 8
	gpuFinderDecisionAltPatterns  = 32
	gpuFinderDecisionMaxAlts      = 8
	gpuFinderDecisionScan         = gpuFinderDecisionAltPatterns +
		gpuFinderDecisionMaxAlts*gpuFinderFoldPatternWords
	gpuFinderDecisionGeometry = gpuFinderDecisionScan + 1
	gpuFinderDecisionWords    = gpuFinderDecisionGeometry + 1

	gpuFinderDecisionIndirectWords = 3
	gpuFinderDecisionRetainedBytes = (gpuFinderDecisionWords +
		2*gpuFinderDecisionIndirectWords) * 4

	gpuFinderDecisionModeRetry       = 0
	gpuFinderDecisionModeRow         = 1
	gpuFinderDecisionModeRowVertical = 2
)

// ensureFinderDecisionLocked creates the control only when a resident batch is
// ready to consume it. Keeping this lazy avoids adding a pipeline compilation
// and retained allocation to the existing materializing route.
func (resident *gpuResidentBinarizer) ensureFinderDecisionLocked() error {
	if resident.finderDecisionBindings != nil {
		return nil
	}
	if resident.finderFoldCursor == nil {
		return fmt.Errorf("jabcode: resident GPU finder decision has no traversal cursor")
	}
	decision, err := resident.device.NewBuffer(gpuFinderDecisionWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU finder decision: %w", err)
	}
	indirect, err := resident.device.NewBuffer(gpuFinderDecisionIndirectWords * 4)
	if err != nil {
		_ = decision.Close()
		return fmt.Errorf("jabcode: allocate resident GPU finder decision control: %w", err)
	}
	rowIndirect, err := resident.device.NewBuffer(gpuFinderDecisionIndirectWords * 4)
	if err != nil {
		_ = indirect.Close()
		_ = decision.Close()
		return fmt.Errorf("jabcode: allocate resident GPU row decision control: %w", err)
	}
	kernel, err := resident.kernels.finderDecision()
	if err != nil {
		_ = rowIndirect.Close()
		_ = indirect.Close()
		_ = decision.Close()
		return err
	}
	bindings, err := kernel.NewBindings(
		vulki.BindBuffer(0, resident.foldSelection),
		vulki.BindBuffer(1, resident.foldRecord),
		vulki.BindBuffer(2, resident.assemblyRecord),
		vulki.BindBuffer(3, resident.familyPoolRecord),
		vulki.BindBuffer(4, resident.contextualPoolRecord),
		vulki.BindBuffer(5, resident.cornerRecord),
		vulki.BindBuffer(6, decision),
		vulki.BindBuffer(7, indirect),
		vulki.BindBuffer(8, rowIndirect),
		vulki.BindBuffer(9, resident.finderFoldCursor),
	)
	if err != nil {
		_ = rowIndirect.Close()
		_ = indirect.Close()
		_ = decision.Close()
		return fmt.Errorf("jabcode: bind resident GPU finder decision: %w", err)
	}
	resident.finderDecision = decision
	resident.finderDecisionIndirect = indirect
	resident.finderRowIndirect = rowIndirect
	resident.finderDecisionKernel = kernel
	resident.finderDecisionBindings = bindings
	if resident.binarizer != nil && resident.binarizer.onRetainedAllocation != nil {
		resident.binarizer.onRetainedAllocation(gpuFinderDecisionRetainedBytes)
	}
	return nil
}

func (resident *gpuResidentBinarizer) ensureFinderDirectionalControlLocked() error {
	if resident.finderDirectionalRowBindings != nil {
		return nil
	}
	if resident.binarizer == nil || resident.binarizer.colorSource == nil {
		return fmt.Errorf("jabcode: resident GPU directional control has no source image")
	}
	scanKernel, err := resident.kernels.finderWindows(finderScanInterleaved)
	if err != nil {
		return err
	}
	if err := resident.binarizer.ensureDirectionalBuffers(scanKernel); err != nil {
		return err
	}
	if err := resident.binarizer.ensureDirectionalChainBuffers(); err != nil {
		return err
	}
	if err := resident.binarizer.ensureDirectionalBatchBuffers(); err != nil {
		return err
	}
	kernel, err := resident.kernels.finderDirectionalControl()
	if err != nil {
		return err
	}
	cursor, err := resident.device.NewBuffer(4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU directional cursor: %w", err)
	}
	bind := func(decision *vulki.Buffer) (*vulki.BindingSet, error) {
		return kernel.NewBindings(
			vulki.BindBuffer(0, resident.binarizer.params),
			vulki.BindBuffer(1, decision),
			vulki.BindBuffer(2, cursor),
			vulki.BindBuffer(3, resident.binarizer.dirParams),
			vulki.BindBuffer(4, resident.binarizer.dirChainParams),
			vulki.BindBuffer(5, resident.binarizer.dirArgs),
		)
	}
	rowBindings, err := bind(resident.finderRowIndirect)
	if err != nil {
		_ = cursor.Close()
		return fmt.Errorf("jabcode: bind resident GPU column control: %w", err)
	}
	retryBindings, err := bind(resident.finderDecisionIndirect)
	if err != nil {
		_ = rowBindings.Close()
		_ = cursor.Close()
		return fmt.Errorf("jabcode: bind resident GPU retry control: %w", err)
	}
	resident.finderDirectionalCursor = cursor
	resident.finderDirectionalRowBindings = rowBindings
	resident.finderDirectionalRetryBindings = retryBindings
	if resident.binarizer.onRetainedAllocation != nil {
		resident.binarizer.onRetainedAllocation(4)
	}
	return nil
}

func (resident *gpuResidentBinarizer) ensureFinderFoldControlLocked() error {
	if resident.finderFoldControlBindings != nil {
		return nil
	}
	kernel, err := resident.kernels.finderFoldControl()
	if err != nil {
		return err
	}
	cursor, err := resident.device.NewBuffer(4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU finder fold cursor: %w", err)
	}
	bindings, err := kernel.NewBindings(
		vulki.BindBuffer(0, resident.binarizer.params),
		vulki.BindBuffer(1, cursor),
		vulki.BindBuffer(2, resident.foldParams),
		vulki.BindBuffer(3, resident.assemblyParams),
		vulki.BindBuffer(4, resident.familyPoolParams),
		vulki.BindBuffer(5, resident.groupParams),
		vulki.BindBuffer(6, resident.contextualParams),
		vulki.BindBuffer(7, resident.cornerParams),
	)
	if err != nil {
		_ = cursor.Close()
		return fmt.Errorf("jabcode: bind resident GPU finder fold control: %w", err)
	}
	resident.finderFoldCursor = cursor
	resident.finderFoldControlBindings = bindings
	if resident.binarizer.onRetainedAllocation != nil {
		resident.binarizer.onRetainedAllocation(4)
	}
	return nil
}

func (resident *gpuResidentBinarizer) resetFinderDecision(
	recorder *vulki.Recorder,
) error {
	if err := recorder.Fill(
		resident.finderDecision, 0, gpuFinderDecisionWords*4, 0,
	); err != nil {
		return fmt.Errorf("jabcode: clear resident GPU finder decision: %w", err)
	}
	if err := recorder.Fill(
		resident.finderDecisionIndirect, 0, gpuFinderDecisionIndirectWords*4, 1,
	); err != nil {
		return fmt.Errorf("jabcode: start resident GPU finder decision: %w", err)
	}
	if err := recorder.Fill(
		resident.finderRowIndirect, 0, gpuFinderDecisionIndirectWords*4, 0,
	); err != nil {
		return fmt.Errorf("jabcode: clear resident GPU row decision: %w", err)
	}
	if err := recorder.Barrier(
		resident.finderDecision, resident.finderDecisionIndirect,
		resident.finderRowIndirect,
	); err != nil {
		return fmt.Errorf("jabcode: synchronize resident GPU finder decision reset: %w", err)
	}
	return nil
}

func (resident *gpuResidentBinarizer) recordFinderDecision(
	recorder *vulki.Recorder,
	indirect *vulki.Buffer,
) error {
	var err error
	if indirect == nil {
		err = recorder.Dispatch(
			resident.finderDecisionKernel,
			resident.finderDecisionBindings,
			vulki.Workgroups{X: 1, Y: 1, Z: 1},
		)
	} else {
		err = recorder.DispatchIndirect(
			resident.finderDecisionKernel,
			resident.finderDecisionBindings,
			indirect,
			0,
		)
	}
	if err != nil {
		return fmt.Errorf("jabcode: dispatch resident GPU finder decision: %w", err)
	}
	if err := recorder.Barrier(
		resident.finderDecision, resident.finderDecisionIndirect,
		resident.finderRowIndirect,
	); err != nil {
		return fmt.Errorf("jabcode: synchronize resident GPU finder decision: %w", err)
	}
	return nil
}

func (resident *gpuResidentBinarizer) setFinderDecisionMode(
	recorder *vulki.Recorder,
	mode uint32,
) error {
	if err := recorder.Fill(
		resident.finderDecision, gpuFinderDecisionMode*4, 4, mode,
	); err != nil {
		return fmt.Errorf("jabcode: set resident GPU finder decision mode: %w", err)
	}
	if err := recorder.Barrier(resident.finderDecision); err != nil {
		return fmt.Errorf("jabcode: synchronize resident GPU finder decision mode: %w", err)
	}
	return nil
}

func gpuDirectionalFoldSource(
	bindings *vulki.BindingSet,
	slot int,
) gpuFinderFoldSource {
	return gpuFinderFoldSource{
		Bindings:    bindings,
		Base:        slot * gpuFinderDirectionalCompactCapacity,
		Count:       gpuFinderDirectionalCompactCapacity,
		Stride:      gpuFinderChainOutcomeWords,
		DeviceCount: true,
		CountOffset: slot*gpuFinderDirectionalSummaryWords +
			gpuFinderDirectionalSummaryCompacted,
		RequiredAt: slot*gpuFinderDirectionalSummaryWords +
			gpuFinderDirectionalSummaryRequired,
		RequiredMax: gpuFinderDirectionalCapacity,
	}
}

// FoldLocateBatchResident runs the row preview, its conditional row-plus-column
// fold, and the five retry folds in host traversal order without exposing an
// intermediate selection. The row decision suppresses the column scan itself
// when its type counts do not call for one, while the retained consistency
// decision suppresses each later retry scan, chain and fold. Every retained
// geometry and wire interpretation then reaches primary admission before one
// fixed result batch crosses to the host.
func (resident *gpuResidentBinarizer) FoldLocateBatchResident(
	variants []wire.Variant,
	quit func() bool,
) ([]PrimaryBatchAttempt, error) {
	out, err := resident.foldLocateBatchResident(variants, quit, nil, 0)
	if err != nil {
		return nil, err
	}
	return gpuPrimaryBatchResults(out, variants)
}

// foldLocateBatchResidentInto leaves one level's fixed batch in a shared
// device allocation. The pyramid joins all level sections before its only
// host-facing result transfer.
func (resident *gpuResidentBinarizer) foldLocateBatchResidentInto(
	variants []wire.Variant,
	quit func() bool,
	destination *vulki.Buffer,
	offset uint64,
) error {
	_, err := resident.foldLocateBatchResident(variants, quit, destination, offset)
	return err
}

func (resident *gpuResidentBinarizer) foldLocateBatchResident(
	variants []wire.Variant,
	quit func() bool,
	destination *vulki.Buffer,
	offset uint64,
) ([]byte, error) {
	if len(variants) < 1 || len(variants) > 2 {
		return nil, fmt.Errorf("jabcode: resident GPU primary batch needs one or two wire variants")
	}
	for _, variant := range variants {
		switch variant {
		case wire.ISO23634, wire.ISOHighColor, wire.CurrentC:
		default:
			return nil, fmt.Errorf("jabcode: resident GPU primary batch does not cover variant %d", variant)
		}
	}
	if resident == nil {
		return nil, fmt.Errorf("jabcode: resident GPU finder decision needs a complete locate batch")
	}
	resident.mu.Lock()
	defer resident.mu.Unlock()
	if resident.closed || resident.device == nil || resident.device.Closed() ||
		resident.binarizer == nil || resident.binarizer.seedHistogram == nil {
		return nil, fmt.Errorf("jabcode: resident GPU finder decision is unavailable")
	}
	shares := 1 + gpuFinderDirectionalBatchMax
	resident.finderPoolShares = 0
	if shares > gpuFinderFamilyPoolMaxShares {
		return nil, fmt.Errorf("jabcode: GPU finder pool takes up to %d folds per locate",
			gpuFinderFamilyPoolMaxShares)
	}
	rowOutcomes := resident.binarizer.rowCompacted
	rowSummaries := resident.binarizer.rowSummary
	if rowOutcomes == nil || rowSummaries == nil {
		return nil, fmt.Errorf("jabcode: resident GPU finder decision has no batch outcomes")
	}
	if err := resident.ensureFinderFoldControlLocked(); err != nil {
		return nil, err
	}
	if err := resident.ensureFinderDecisionLocked(); err != nil {
		return nil, err
	}
	if err := resident.ensureFinderDirectionalControlLocked(); err != nil {
		return nil, err
	}
	if err := resident.ensureFinderGeometryLocked(); err != nil {
		return nil, err
	}
	directionalOutcomes := resident.binarizer.dirBatchOutcomes
	directionalSummaries := resident.binarizer.dirBatchSummary
	if directionalOutcomes == nil || directionalSummaries == nil {
		return nil, fmt.Errorf("jabcode: resident GPU finder decision has no directional outcomes")
	}
	rowBindings, err := resident.newFinderAssemblyCountBindings(
		rowOutcomes, rowSummaries,
	)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = rowBindings.Close()
	}()
	directionalBindings, err := resident.newFinderAssemblyCountBindings(
		directionalOutcomes, directionalSummaries,
	)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = directionalBindings.Close()
	}()
	rowSource := gpuFinderFoldSource{
		Bindings:    rowBindings,
		Base:        currentFamilySeekChannel * gpuRowCompactCapacity,
		Count:       gpuRowCompactCapacity,
		Stride:      gpuRowCompactWords,
		DeviceCount: true,
		CountOffset: currentFamilySeekChannel*gpuRowSummaryWords +
			gpuRowSummaryCompacted,
		RequiredAt: currentFamilySeekChannel*gpuRowSummaryWords +
			gpuRowSummaryOverflow,
		RequiredMax: 0,
	}
	verticalSource := gpuDirectionalFoldSource(directionalBindings, 0)
	if err := validateGPUFinderFoldSources([]gpuFinderFoldSource{rowSource}); err != nil {
		return nil, err
	}
	if err := validateGPUFinderFoldSources(
		[]gpuFinderFoldSource{rowSource, verticalSource},
	); err != nil {
		return nil, err
	}
	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return nil, fmt.Errorf("jabcode: create resident GPU finder decision recorder: %w", err)
	}
	defer recorder.Abort()
	resident.poolsStale = true
	resident.invalidateFinderPoolMirror()
	for _, buffer := range []*vulki.Buffer{
		resident.familyPoolRecord,
		resident.contextualPoolRecord,
		resident.binarizer.seedHistogram,
	} {
		if err := recorder.Fill(buffer, 0, buffer.Size(), 0); err != nil {
			return nil, fmt.Errorf("jabcode: clear resident GPU locate accumulation: %w", err)
		}
	}
	if err := recorder.Barrier(
		resident.familyPoolRecord,
		resident.contextualPoolRecord,
		resident.binarizer.seedHistogram,
	); err != nil {
		return nil, fmt.Errorf("jabcode: synchronize resident GPU locate reset: %w", err)
	}
	if err := resident.resetFinderDecision(recorder); err != nil {
		return nil, err
	}
	if err := recorder.Fill(resident.finderDirectionalCursor, 0, 4, 0); err != nil {
		return nil, fmt.Errorf("jabcode: reset resident GPU directional cursor: %w", err)
	}
	if err := recorder.Barrier(resident.finderDirectionalCursor); err != nil {
		return nil, fmt.Errorf("jabcode: synchronize resident GPU directional cursor: %w", err)
	}
	if err := recorder.Fill(resident.finderFoldCursor, 0, 4, 0); err != nil {
		return nil, fmt.Errorf("jabcode: reset resident GPU finder fold cursor: %w", err)
	}
	if err := recorder.Barrier(resident.finderFoldCursor); err != nil {
		return nil, fmt.Errorf("jabcode: synchronize resident GPU finder fold cursor: %w", err)
	}
	resident.finderPoolShares += shares
	if err := resident.setFinderDecisionMode(recorder, gpuFinderDecisionModeRow); err != nil {
		return nil, err
	}
	if err := resident.recordResidentFinderOutcomes(
		recorder, []*vulki.BindingSet{rowBindings}, nil,
	); err != nil {
		return nil, err
	}
	if err := resident.recordFinderDecision(recorder, nil); err != nil {
		return nil, err
	}
	if err := resident.setFinderDecisionMode(
		recorder, gpuFinderDecisionModeRowVertical,
	); err != nil {
		return nil, err
	}
	if err := resident.binarizer.recordResidentDirectionalSweep(
		recorder, resident.finderDirectionalRowBindings,
		resident.finderDirectionalCursor, 0,
	); err != nil {
		return nil, err
	}
	if err := resident.recordResidentFinderOutcomes(
		recorder, []*vulki.BindingSet{rowBindings, directionalBindings},
		resident.finderRowIndirect,
	); err != nil {
		return nil, err
	}
	if err := resident.recordFinderDecision(recorder, resident.finderRowIndirect); err != nil {
		return nil, err
	}
	if err := resident.setFinderDecisionMode(recorder, gpuFinderDecisionModeRetry); err != nil {
		return nil, err
	}
	for slot := 1; slot < gpuFinderDirectionalBatchMax; slot++ {
		if quit != nil && quit() {
			return nil, fmt.Errorf("jabcode: GPU primary batch was cancelled while recording finders")
		}
		if err := resident.binarizer.recordResidentDirectionalSweep(
			recorder, resident.finderDirectionalRetryBindings,
			resident.finderDirectionalCursor, slot,
		); err != nil {
			return nil, err
		}
		if err := resident.recordResidentFinderOutcomes(
			recorder, []*vulki.BindingSet{directionalBindings},
			resident.finderDecisionIndirect,
		); err != nil {
			return nil, err
		}
		if err := resident.recordFinderDecision(
			recorder, resident.finderDecisionIndirect,
		); err != nil {
			return nil, err
		}
	}
	resident.sampledGrid = nil
	resident.payloadControlReady = false
	if err := recorder.Fill(
		resident.primaryResult, 0, gpuPrimaryResultBatchWords*4, 0,
	); err != nil {
		return nil, fmt.Errorf("jabcode: clear resident GPU primary result batch: %w", err)
	}
	if err := recorder.Barrier(resident.primaryResult); err != nil {
		return nil, fmt.Errorf("jabcode: synchronize resident GPU primary result batch: %w", err)
	}
	if err := resident.recordMetadataRows(recorder); err != nil {
		return nil, err
	}
	for geometry := 0; geometry <= gpuFinderDecisionMaxAlts; geometry++ {
		if quit != nil && quit() {
			return nil, fmt.Errorf("jabcode: GPU primary batch was cancelled while recording geometry")
		}
		if err := recorder.Fill(
			resident.finderDecision, uint64(gpuFinderDecisionGeometry*4), 4, uint32(geometry),
		); err != nil {
			return nil, fmt.Errorf("jabcode: select resident GPU finder geometry: %w", err)
		}
		if err := recorder.Barrier(resident.finderDecision); err != nil {
			return nil, fmt.Errorf("jabcode: synchronize resident GPU finder geometry selection: %w", err)
		}
		if err := resident.recordFinderGeometrySample(recorder); err != nil {
			return nil, err
		}
		for variantSlot, variant := range variants {
			if quit != nil && quit() {
				return nil, fmt.Errorf("jabcode: GPU primary batch was cancelled while recording payloads")
			}
			slot := geometry*2 + variantSlot
			if err := resident.recordMetadataWalkReady(
				recorder, variant, slot, true, resident.finderGeometryIndirect,
			); err != nil {
				return nil, err
			}
			if err := resident.recordPayloadCorrection(
				recorder, nil, resident.finderGeometryIndirect,
				geometry == 0 && variantSlot == 0,
			); err != nil {
				return nil, err
			}
		}
	}
	if quit != nil && quit() {
		return nil, fmt.Errorf("jabcode: GPU primary batch was cancelled before submission")
	}
	var out []byte
	if destination != nil {
		if offset > destination.Size() ||
			uint64(gpuPrimaryResultBatchBytes) > destination.Size()-offset {
			return nil, fmt.Errorf("jabcode: GPU pyramid primary result section exceeds its buffer")
		}
		if err := recorder.Copy(
			destination, offset, resident.primaryResult, 0, gpuPrimaryResultBatchBytes,
		); err != nil {
			return nil, fmt.Errorf("jabcode: join resident GPU primary result batch: %w", err)
		}
	} else {
		out = make([]byte, gpuPrimaryResultBatchBytes)
		phaseprobe.Count("download.primary_result_batch", len(out))
		if err := recorder.Download(resident.primaryResult, 0, out); err != nil {
			return nil, fmt.Errorf("jabcode: record resident GPU primary result batch download: %w", err)
		}
	}
	if err := recorder.SubmitAndWait(); err != nil {
		// A failed submission can have merged an unknown prefix of the batch.
		// Do not let a later locate treat that partial union as valid evidence.
		resident.poolsStale = true
		resident.permutationCacheDirty = true
		resident.ldpcMatrixCacheDirty = true
		return nil, fmt.Errorf("jabcode: run resident GPU locate and primary batch: %w", err)
	}
	resident.permutationCacheDirty = false
	resident.ldpcMatrixCacheDirty = false
	resident.metadataRowsReady = true
	resident.poolsStale = false
	return out, nil
}
