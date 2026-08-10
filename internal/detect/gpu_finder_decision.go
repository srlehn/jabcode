//go:build !js

package detect

import (
	"fmt"
	"image"

	"github.com/srlehn/vulki"
)

const (
	gpuFinderDecisionHave         = 0
	gpuFinderDecisionConsistent   = 1
	gpuFinderDecisionDeclined     = 2
	gpuFinderDecisionMissing      = 3
	gpuFinderDecisionCornerSource = 4
	gpuFinderDecisionCornerMiss   = 5
	gpuFinderDecisionAlternatives = 6
	gpuFinderDecisionPatterns     = 8
	gpuFinderDecisionAltPatterns  = 32
	gpuFinderDecisionMaxAlts      = 8
	gpuFinderDecisionWords        = gpuFinderDecisionAltPatterns +
		gpuFinderDecisionMaxAlts*gpuFinderFoldPatternWords

	gpuFinderDecisionIndirectWords = 3
	gpuFinderDecisionRetainedBytes = (gpuFinderDecisionWords + gpuFinderDecisionIndirectWords) * 4
)

// ensureFinderDecisionLocked creates the control only when a resident batch is
// ready to consume it. Keeping this lazy avoids adding a pipeline compilation
// and retained allocation to the existing materializing route.
func (resident *gpuResidentBinarizer) ensureFinderDecisionLocked() error {
	if resident.finderDecisionBindings != nil {
		return nil
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
	kernel, err := resident.kernels.finderDecision()
	if err != nil {
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
	)
	if err != nil {
		_ = indirect.Close()
		_ = decision.Close()
		return fmt.Errorf("jabcode: bind resident GPU finder decision: %w", err)
	}
	resident.finderDecision = decision
	resident.finderDecisionIndirect = indirect
	resident.finderDecisionKernel = kernel
	resident.finderDecisionBindings = bindings
	if resident.binarizer != nil && resident.binarizer.onRetainedAllocation != nil {
		resident.binarizer.onRetainedAllocation(gpuFinderDecisionRetainedBytes)
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
	return nil
}

func (resident *gpuResidentBinarizer) recordFinderDecision(
	recorder *vulki.Recorder,
) error {
	if err := recorder.DispatchIndirect(
		resident.finderDecisionKernel,
		resident.finderDecisionBindings,
		resident.finderDecisionIndirect,
		0,
	); err != nil {
		return fmt.Errorf("jabcode: dispatch resident GPU finder decision: %w", err)
	}
	if err := recorder.Barrier(
		resident.finderDecision,
		resident.finderDecisionIndirect,
	); err != nil {
		return fmt.Errorf("jabcode: synchronize resident GPU finder decision: %w", err)
	}
	return nil
}

// FoldDirectionalBatchResident folds retry directions in traversal order and
// retains the first successful or first consistent answer on the device. The
// fold workspace is reused between directions; the decision's indirect words
// suppress every later stage once the same consistency gate as the host has
// settled the traversal.
func (resident *gpuResidentBinarizer) FoldDirectionalBatchResident(
	frame image.Point,
	sweeps []finderDirSweep,
	printPass bool,
) error {
	if resident == nil || len(sweeps) == 0 {
		return fmt.Errorf("jabcode: resident GPU finder decision needs a directional batch")
	}
	resident.mu.Lock()
	defer resident.mu.Unlock()
	if resident.closed || resident.device == nil || resident.device.Closed() ||
		resident.binarizer == nil || resident.poolsStale {
		return fmt.Errorf("jabcode: resident GPU finder decision is unavailable")
	}
	if resident.finderPoolShares+len(sweeps) > gpuFinderFamilyPoolMaxShares {
		return fmt.Errorf("jabcode: GPU finder pool takes up to %d folds per locate",
			gpuFinderFamilyPoolMaxShares)
	}
	var slots [gpuFinderDirectionalBatchMax]bool
	for _, sweep := range sweeps {
		if !sweep.resident || sweep.slot < 0 ||
			sweep.slot >= gpuFinderDirectionalBatchMax {
			return fmt.Errorf("jabcode: resident GPU finder decision has an invalid sweep slot")
		}
		if slots[sweep.slot] {
			return fmt.Errorf("jabcode: resident GPU finder decision repeats sweep slot %d",
				sweep.slot)
		}
		slots[sweep.slot] = true
	}
	outcomes := resident.binarizer.dirBatchOutcomes
	summaries := resident.binarizer.dirBatchSummary
	if outcomes == nil || summaries == nil {
		return fmt.Errorf("jabcode: resident GPU finder decision has no batch outcomes")
	}
	if err := resident.ensureFinderDecisionLocked(); err != nil {
		return err
	}
	bindings, err := resident.newFinderAssemblyCountBindings(outcomes, summaries)
	if err != nil {
		return err
	}
	defer func() {
		_ = bindings.Close()
	}()
	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return fmt.Errorf("jabcode: create resident GPU finder decision recorder: %w", err)
	}
	defer recorder.Abort()
	if err := resident.resetFinderDecision(recorder); err != nil {
		return err
	}
	resident.finderPoolShares += len(sweeps)
	resident.invalidateFinderPoolMirror()
	for _, sweep := range sweeps {
		source := gpuFinderFoldSource{
			Bindings:    bindings,
			Base:        sweep.slot * gpuFinderDirectionalCompactCapacity,
			Count:       gpuFinderDirectionalCompactCapacity,
			Stride:      gpuFinderChainOutcomeWords,
			DeviceCount: true,
			CountOffset: sweep.slot*gpuFinderDirectionalSummaryWords +
				gpuFinderDirectionalSummaryCompacted,
			RequiredAt: sweep.slot*gpuFinderDirectionalSummaryWords +
				gpuFinderDirectionalSummaryRequired,
			RequiredMax: gpuFinderDirectionalCapacity,
		}
		if err := validateGPUFinderFoldSources([]gpuFinderFoldSource{source}); err != nil {
			return err
		}
		if err := resident.recordFinderOutcomes(
			recorder,
			[]gpuFinderFoldSource{source},
			frame,
			printPass,
			[4]bool{},
			resident.finderDecisionIndirect,
		); err != nil {
			return err
		}
		if err := resident.recordFinderDecision(recorder); err != nil {
			return err
		}
	}
	if err := recorder.SubmitAndWait(); err != nil {
		// A failed submission can have merged an unknown prefix of the batch.
		// Do not let a later locate treat that partial union as valid evidence.
		resident.poolsStale = true
		return fmt.Errorf("jabcode: run resident GPU finder decision: %w", err)
	}
	return nil
}
