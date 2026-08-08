//go:build !js

package detect

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"image"
	"math"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/ecc"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/tables"
	"github.com/srlehn/jabcode/internal/wire"
)

//go:embed shaders/metadata_part1.wgsl
var metadataPart1WGSL string

//go:embed shaders/metadata_palette.wgsl
var metadataPaletteWGSL string

// Parameter word indices, matching the metadata shaders.
const (
	gpuMetadataParamSideX      = 0
	gpuMetadataParamSideY      = 1
	gpuMetadataParamPlacement4 = 8
	gpuMetadataParamPlacement8 = 24
	gpuMetadataParamWords      = 64
)

// Record word indices, matching the metadata shaders.
const (
	gpuMetadataRecordStatus     = 0
	gpuMetadataRecordModules    = 1
	gpuMetadataRecordWalkX      = 2
	gpuMetadataRecordWalkY      = 3
	gpuMetadataRecordNC         = 4
	gpuMetadataRecordColors     = 5
	gpuMetadataRecordPalette    = 16
	gpuMetadataRecordNormalized = 112
	gpuMetadataRecordThresholds = 240
	gpuMetadataRecordWords      = 256
)

// Metadata record statuses. Anything other than ok means the walk resolved to
// something the host has a ladder for rather than that the read failed.
const (
	gpuMetadataStatusOK          = 0
	gpuMetadataStatusDefault     = 1
	gpuMetadataStatusUnsupported = 2
	gpuMetadataStatusFailed      = 3
)

// gpuMetadataWalk is what the device made of one symbol's metadata strip: the
// colour mode, the embedded palette and everything the classifier derives from
// it, and how much of the grid the walk consumed.
//
// Defaulted and Unsupported are not failures. The first is how a symbol
// carrying no explicit metadata presents, which the host answers with default
// metadata; the second is a colour mode the device classifier does not cover,
// which the host answers from its own walk. The syndrome is reported rather
// than enforced, exactly as the host reports it, because metadata has fallback
// ladders of its own.
type gpuMetadataWalk struct {
	NC          int
	Colors      int
	Defaulted   bool
	Unsupported bool
	SyndromeOK  bool
	ModuleCount int

	Palette    []byte
	Normalized []float64
	Thresholds []float64
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
	resident.metadataPaletteKernel, err = resident.kernels.metadataPalette()
	if err != nil {
		return err
	}
	resident.metadataPaletteBindings, err = resident.metadataPaletteKernel.NewBindings(
		vulki.BindBuffer(0, resident.metadataParams),
		vulki.BindBuffer(1, resident.sampleResult),
		vulki.BindBuffer(2, resident.ldpcNet),
		vulki.BindBuffer(3, resident.metadataRecord),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU metadata palette: %w", err)
	}
	return nil
}

// gpuMetadataParams builds the metadata walk's parameter block. The palette
// placement tables go in resolved for the wire variant, because which entry a
// copy carries at a slot is the format's business and not the kernel's, and the
// colour mode that selects between them is only known on the device.
func gpuMetadataParams(side image.Point, variant wire.Variant) [gpuMetadataParamWords * 4]byte {
	var params [gpuMetadataParamWords * 4]byte
	put := func(index int, value uint32) {
		binary.LittleEndian.PutUint32(params[index*4:], value)
	}
	put(gpuMetadataParamSideX, uint32(side.X))
	put(gpuMetadataParamSideY, uint32(side.Y))
	for copy := range spec.ColorPaletteNumber {
		for slot := range 4 {
			put(gpuMetadataParamPlacement4+copy*4+slot, uint32(
				tables.PrimaryPalettePlacementIndexVariant(copy, slot, 4, variant)%4))
		}
		for slot := range 8 {
			put(gpuMetadataParamPlacement8+copy*8+slot, uint32(
				tables.PrimaryPalettePlacementIndexVariant(copy, slot, 8, variant)%8))
		}
	}
	return params
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

// WalkMetadata reads the primary metadata strip off the resident module grid
// and interprets it where it lies: Part I and its correction, then the embedded
// palette and everything the classifier derives from it.
//
// The whole walk is one submission with no host contact between its stages -
// the colour mode Part I resolves is what tells the palette stage how many
// modules to read, and it reads that out of the corrector's own output. The
// method declines with an error, rather than a failed read, for a grid it does
// not own.
func (resident *gpuResidentBinarizer) WalkMetadata(
	side image.Point,
	variant wire.Variant,
) (gpuMetadataWalk, error) {
	var result gpuMetadataWalk
	if resident == nil || resident.closed || resident.metadataPaletteBindings == nil {
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

	params := gpuMetadataParams(side, variant)
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
	if err := recorder.Dispatch(
		resident.metadataPaletteKernel,
		resident.metadataPaletteBindings,
		vulki.Workgroups{X: 1, Y: 1, Z: 1},
	); err != nil {
		return result, fmt.Errorf("jabcode: dispatch GPU metadata palette: %w", err)
	}
	if err := recorder.Barrier(resident.metadataRecord); err != nil {
		return result, fmt.Errorf("jabcode: synchronize GPU metadata palette: %w", err)
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

	word := func(index int) uint32 {
		return binary.LittleEndian.Uint32(record[index*4:])
	}
	result.ModuleCount = int(word(gpuMetadataRecordModules))
	switch word(gpuMetadataRecordStatus) {
	case gpuMetadataStatusDefault:
		result.Defaulted = true
		return result, nil
	case gpuMetadataStatusUnsupported:
		result.Unsupported = true
		result.NC = int(word(gpuMetadataRecordNC))
		return result, nil
	case gpuMetadataStatusFailed:
		return result, fmt.Errorf("jabcode: GPU metadata walk left the symbol")
	}

	_, ok, err := gpuLDPCResult(plan, net, plan.net)
	if err != nil {
		return result, err
	}
	result.SyndromeOK = ok
	result.NC = int(word(gpuMetadataRecordNC))
	result.Colors = int(word(gpuMetadataRecordColors))

	entries := result.Colors * spec.ColorPaletteNumber
	result.Palette = make([]byte, entries*3)
	for i := range result.Palette {
		result.Palette[i] = byte(word(gpuMetadataRecordPalette + i))
	}
	result.Normalized = make([]float64, entries*4)
	for i := range result.Normalized {
		result.Normalized[i] = float64(math.Float32frombits(word(gpuMetadataRecordNormalized + i)))
	}
	result.Thresholds = make([]float64, 3*spec.ColorPaletteNumber)
	for i := range result.Thresholds {
		result.Thresholds[i] = float64(math.Float32frombits(word(gpuMetadataRecordThresholds + i)))
	}
	return result, nil
}
