//go:build !js

package detect

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"image"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/ecc"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/wire"
)

//go:embed shaders/metadata_part1.wgsl
var metadataPart1WGSL string

// Parameter word indices, matching metadata_part1.wgsl.
const (
	gpuMetadataParamSideX = 0
	gpuMetadataParamSideY = 1
	gpuMetadataParamWords = 8
)

// Record word indices, matching metadata_part1.wgsl.
const (
	gpuMetadataRecordStatus       = 0
	gpuMetadataRecordPart1Modules = 1
	gpuMetadataRecordWords        = 16
)

// Metadata record statuses. Anything other than ok means the walk resolved to
// something the host has a ladder for rather than that the read failed.
const (
	gpuMetadataStatusOK      = 0
	gpuMetadataStatusDefault = 1
)

// gpuMetadataPartI is one symbol's Part I result: the colour mode, whether the
// walk fell back to default metadata, and whether the corrected part satisfied
// its parity checks. The syndrome is reported rather than enforced, exactly as
// the host reports it, because metadata has fallback ladders of its own.
type gpuMetadataPartI struct {
	NC          int
	Defaulted   bool
	SyndromeOK  bool
	ModuleCount int
}

// initializeMetadata allocates the metadata walk's buffers and compiles its
// kernel with the rest of the resident stage set, so the compile lands in
// warm-up rather than on the decode call.
func (resident *gpuResidentBinarizer) initializeMetadata() error {
	var err error
	resident.metadataParams, err = resident.device.NewBuffer(gpuMetadataParamWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU metadata parameters: %w", err)
	}
	resident.metadataRecord, err = resident.device.NewBuffer(gpuMetadataRecordWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU metadata record: %w", err)
	}
	resident.metadataPart1Kernel, err = resident.kernels.metadataPart1()
	if err != nil {
		return err
	}
	resident.metadataPart1Bindings, err = resident.metadataPart1Kernel.NewBindings(
		vulki.BindBuffer(0, resident.metadataParams),
		vulki.BindBuffer(1, resident.sampleResult),
		vulki.BindBuffer(2, resident.ldpcBits),
		vulki.BindBuffer(3, resident.metadataRecord),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU metadata walk: %w", err)
	}
	return nil
}

// gpuMetadataLDPCPlan builds the correction plan for one metadata part.
//
// The metadata codes take HardBlockSplit's short-code branch, which discards the
// caller's wc: it sets WC to 2 unless the net length exceeds 36, and the host
// decoder builds its parity-check matrix from that rebound value. A plan built
// from the wc the caller passed solves a different system and returns a
// plausible wrong answer with no error to distinguish it.
func gpuMetadataLDPCPlan(length, wc int, variant wire.Variant) (gpuLDPCPlan, error) {
	split := ecc.HardBlockSplit(length, wc, 0)
	rows, ok := ecc.ParityRows(split.WC, 0, split.GrossSub, variant)
	if !ok {
		return gpuLDPCPlan{}, fmt.Errorf(
			"jabcode: no metadata parity rows for wc=%d gross=%d", split.WC, split.GrossSub)
	}
	plan := gpuLDPCPlan{
		rows:      rows.Rows,
		rowDegree: rows.Degree,
		length:    split.GrossSub,
		height:    rows.Height,
		rank:      rows.Rank,
		net:       split.NetSub,
		blocks:    split.Blocks,
	}
	if !plan.valid() {
		return gpuLDPCPlan{}, fmt.Errorf("jabcode: metadata correction plan is out of range")
	}
	return plan, nil
}

// gpuMetadataPartIPlan is the Part I correction plan. Its wc is the one
// DecodePrimaryMetadataPartI passes, which the split then rebinds.
func gpuMetadataPartIPlan(variant wire.Variant) (gpuLDPCPlan, error) {
	wc := 3
	if spec.PrimaryMetadataPart1Length > 36 {
		wc = 4
	}
	return gpuMetadataLDPCPlan(spec.PrimaryMetadataPart1Length, wc, variant)
}

// WalkMetadataPartI reads Part I off the resident module grid and corrects it
// where it lies, so nothing but the interpreted colour mode leaves the device.
//
// It reports the same three things the host's Part I does - the colour mode,
// the default-metadata fallback, and the parity verdict - and it declines with
// an error, rather than a failed read, for a grid it does not own.
func (resident *gpuResidentBinarizer) WalkMetadataPartI(
	side image.Point,
	variant wire.Variant,
) (gpuMetadataPartI, error) {
	var result gpuMetadataPartI
	if resident == nil || resident.closed || resident.metadataPart1Bindings == nil {
		return result, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	if !spec.ValidSideSize(side.X) || !spec.ValidSideSize(side.Y) ||
		side.X > gpuSampleMaxSide || side.Y > gpuSampleMaxSide {
		return result, fmt.Errorf("jabcode: GPU metadata side %dx%d is out of range", side.X, side.Y)
	}
	plan, err := gpuMetadataPartIPlan(variant)
	if err != nil {
		return result, err
	}

	resident.mu.Lock()
	defer resident.mu.Unlock()

	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return result, fmt.Errorf("jabcode: create GPU metadata recorder: %w", err)
	}
	defer recorder.Abort()

	var params [gpuMetadataParamWords * 4]byte
	binary.LittleEndian.PutUint32(params[gpuMetadataParamSideX*4:], uint32(side.X))
	binary.LittleEndian.PutUint32(params[gpuMetadataParamSideY*4:], uint32(side.Y))
	if err := recorder.Update(resident.metadataParams, 0, params[:]); err != nil {
		return result, fmt.Errorf("jabcode: update GPU metadata parameters: %w", err)
	}
	// The classifier writes only the bits it resolves, so the codeword has to
	// start clear rather than carry the previous symbol's Part I.
	if err := recorder.Fill(resident.ldpcBits, 0, uint64(plan.length*4), 0); err != nil {
		return result, fmt.Errorf("jabcode: clear GPU metadata codeword: %w", err)
	}
	if err := recorder.Barrier(resident.metadataParams, resident.ldpcBits); err != nil {
		return result, fmt.Errorf("jabcode: synchronize GPU metadata inputs: %w", err)
	}
	if err := recorder.Dispatch(
		resident.metadataPart1Kernel,
		resident.metadataPart1Bindings,
		vulki.Workgroups{X: 1, Y: 1, Z: 1},
	); err != nil {
		return result, fmt.Errorf("jabcode: dispatch GPU metadata Part I: %w", err)
	}
	if err := recorder.Barrier(resident.ldpcBits, resident.metadataRecord); err != nil {
		return result, fmt.Errorf("jabcode: synchronize GPU metadata Part I: %w", err)
	}

	if err := gpuLDPCUploadRows(recorder, resident.ldpcRows, plan); err != nil {
		return result, err
	}
	ldpcParams := gpuLDPCParams(plan)
	if err := recorder.Update(resident.ldpcParams, 0, ldpcParams[:]); err != nil {
		return result, fmt.Errorf("jabcode: update GPU metadata correction parameters: %w", err)
	}
	if err := recorder.Barrier(resident.ldpcRows, resident.ldpcParams); err != nil {
		return result, fmt.Errorf("jabcode: synchronize GPU metadata correction inputs: %w", err)
	}
	if err := recorder.Dispatch(
		resident.ldpcKernel,
		resident.ldpcBindings,
		vulki.Workgroups{X: uint32(plan.blocks), Y: 1, Z: 1},
	); err != nil {
		return result, fmt.Errorf("jabcode: dispatch GPU metadata correction: %w", err)
	}
	if err := recorder.Barrier(resident.ldpcNet); err != nil {
		return result, fmt.Errorf("jabcode: synchronize GPU metadata correction: %w", err)
	}

	record := make([]byte, gpuMetadataRecordWords*4)
	if err := recorder.Download(resident.metadataRecord, 0, record); err != nil {
		return result, fmt.Errorf("jabcode: record GPU metadata download: %w", err)
	}
	net := make([]byte, (plan.blocks+plan.net)*4)
	if err := recorder.Download(resident.ldpcNet, 0, net); err != nil {
		return result, fmt.Errorf("jabcode: record GPU metadata correction download: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return result, fmt.Errorf("jabcode: run GPU metadata walk: %w", err)
	}

	result.ModuleCount = int(binary.LittleEndian.Uint32(record[gpuMetadataRecordPart1Modules*4:]))
	if binary.LittleEndian.Uint32(record[gpuMetadataRecordStatus*4:]) == gpuMetadataStatusDefault {
		result.Defaulted = true
		return result, nil
	}
	dec, ok, err := gpuLDPCResult(plan, net, plan.net)
	if err != nil {
		return result, err
	}
	if len(dec) < 3 {
		return result, fmt.Errorf("jabcode: GPU metadata Part I returned %d bits", len(dec))
	}
	result.SyndromeOK = ok
	result.NC = int(dec[0])<<2 + int(dec[1])<<1 + int(dec[2])
	return result, nil
}
