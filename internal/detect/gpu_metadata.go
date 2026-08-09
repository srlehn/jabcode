//go:build !js

package detect

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"image"
	"math"
	"os"
	"sync/atomic"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/ecc"
	"github.com/srlehn/jabcode/internal/phaseprobe"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/tables"
	"github.com/srlehn/jabcode/internal/wire"
)

//go:embed shaders/metadata_part1.wgsl
var metadataPart1WGSL string

//go:embed shaders/metadata_palette.wgsl
var metadataPaletteWGSL string

//go:embed shaders/metadata_part2.wgsl
var metadataPart2WGSL string

//go:embed shaders/metadata_finish.wgsl
var metadataFinishWGSL string

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
	gpuMetadataRecordVersionX   = 6
	gpuMetadataRecordVersionY   = 7
	gpuMetadataRecordECLX       = 8
	gpuMetadataRecordECLY       = 9
	gpuMetadataRecordMask       = 10
	gpuMetadataRecordSyndrome1  = 12
	gpuMetadataRecordSyndrome2  = 13
	gpuMetadataRecordPalette    = 16
	gpuMetadataRecordNormalized = 112
	gpuMetadataRecordThresholds = 240
	gpuMetadataRecordWords      = 256
)

// Metadata record statuses. Anything other than ok means the walk resolved to
// something the host has a ladder for rather than that the read failed.
const (
	gpuMetadataStatusOK           = 0
	gpuMetadataStatusDefault      = 1
	gpuMetadataStatusUnsupported  = 2
	gpuMetadataStatusFailed       = 3
	gpuMetadataStatusSizeMismatch = 4
	gpuMetadataStatusECCOrder     = 5
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
	SideVersion image.Point
	ECL         image.Point
	MaskType    int
	Defaulted   bool
	Unsupported bool
	Rejected    bool
	ModuleCount int

	PartISyndromeOK  bool
	PartIISyndromeOK bool

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
	resident.metadataPart2Kernel, err = resident.kernels.metadataPart2()
	if err != nil {
		return err
	}
	resident.metadataPart2Bindings, err = resident.metadataPart2Kernel.NewBindings(
		vulki.BindBuffer(0, resident.metadataParams),
		vulki.BindBuffer(1, resident.sampleResult),
		vulki.BindBuffer(2, resident.ldpcBits),
		vulki.BindBuffer(3, resident.metadataRecord),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU metadata Part II: %w", err)
	}
	resident.metadataFinishKernel, err = resident.kernels.metadataFinish()
	if err != nil {
		return err
	}
	resident.metadataFinishBindings, err = resident.metadataFinishKernel.NewBindings(
		vulki.BindBuffer(0, resident.metadataParams),
		vulki.BindBuffer(1, resident.ldpcNet),
		vulki.BindBuffer(2, resident.metadataRecord),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU metadata fields: %w", err)
	}
	return nil
}

// gpuMetadataPartIIPlan is the Part II correction plan. Its wc is the one
// DecodePrimaryMetadataPartII passes, which the split then rebinds.
func gpuMetadataPartIIPlan(variant wire.Variant) (gpuLDPCPlan, error) {
	wc := 3
	if spec.PrimaryMetadataPart2Length > 36 {
		wc = 4
	}
	return gpuMetadataLDPCPlan(spec.PrimaryMetadataPart2Length, wc, variant)
}

// recordMetadataCorrection stages one metadata part's correction: its parity
// rows, its parameters and the dispatch. The two parts share the corrector, so
// the second overwrites the first's inputs, which is why each stage that needs
// a result of the first copies it into the record before this runs again.
func (resident *gpuResidentBinarizer) recordMetadataCorrection(
	recorder *vulki.Recorder,
	plan gpuLDPCPlan,
) error {
	if err := gpuLDPCUploadRows(recorder, resident.ldpcRows, plan); err != nil {
		return err
	}
	params := gpuLDPCParams(plan)
	if err := recorder.Update(resident.ldpcParams, 0, params[:]); err != nil {
		return fmt.Errorf("jabcode: update GPU metadata correction parameters: %w", err)
	}
	if err := recorder.Barrier(resident.ldpcRows, resident.ldpcParams); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU metadata correction inputs: %w", err)
	}
	if err := recorder.Dispatch(
		resident.ldpcKernel,
		resident.ldpcBindings,
		vulki.Workgroups{X: uint32(plan.blocks), Y: 1, Z: 1},
	); err != nil {
		return fmt.Errorf("jabcode: dispatch GPU metadata correction: %w", err)
	}
	if err := recorder.Barrier(resident.ldpcNet, resident.ldpcBits); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU metadata correction: %w", err)
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
// and interprets it where it lies: Part I and its correction, the embedded
// palette and everything the classifier derives from it, then Part II and the
// symbol shape it declares.
//
// The whole walk is one submission with no host contact between its stages.
// Each stage depends on the one before through device memory alone - the colour
// mode Part I resolves is what tells the palette stage how many modules to
// read, and the palette is what Part II is classified against - so the host
// learns none of it until the record comes back at the end. The method declines
// with an error, rather than a failed read, for a grid it does not own.
func (resident *gpuResidentBinarizer) WalkMetadata(
	side image.Point,
	variant wire.Variant,
) (gpuMetadataWalk, error) {
	var result gpuMetadataWalk
	if resident == nil || resident.closed || resident.metadataFinishBindings == nil {
		return result, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	if !spec.ValidSideSize(side.X) || !spec.ValidSideSize(side.Y) ||
		side.X > gpuSampleMaxSide || side.Y > gpuSampleMaxSide {
		return result, fmt.Errorf("jabcode: GPU metadata side %dx%d is out of range", side.X, side.Y)
	}
	partI, err := gpuMetadataPartIPlan(variant)
	if err != nil {
		return result, err
	}
	partII, err := gpuMetadataPartIIPlan(variant)
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
	// A classifier writes only the bits it resolves, so each part's codeword has
	// to start clear rather than carry what the buffer last held.
	if err := recorder.Fill(resident.ldpcBits, 0, uint64(partI.length*4), 0); err != nil {
		return result, fmt.Errorf("jabcode: clear GPU metadata codeword: %w", err)
	}
	if err := recorder.Barrier(resident.metadataParams, resident.ldpcBits); err != nil {
		return result, fmt.Errorf("jabcode: synchronize GPU metadata inputs: %w", err)
	}

	stage := func(kernel *vulki.Kernel, bindings *vulki.BindingSet, name string) error {
		if err := recorder.Dispatch(kernel, bindings, vulki.Workgroups{X: 1, Y: 1, Z: 1}); err != nil {
			return fmt.Errorf("jabcode: dispatch GPU metadata %s: %w", name, err)
		}
		if err := recorder.Barrier(resident.ldpcBits, resident.metadataRecord); err != nil {
			return fmt.Errorf("jabcode: synchronize GPU metadata %s: %w", name, err)
		}
		return nil
	}

	if err := stage(resident.metadataPart1Kernel, resident.metadataPart1Bindings, "Part I"); err != nil {
		return result, err
	}
	if err := resident.recordMetadataCorrection(recorder, partI); err != nil {
		return result, err
	}
	if err := stage(resident.metadataPaletteKernel, resident.metadataPaletteBindings, "palette"); err != nil {
		return result, err
	}
	if err := recorder.Fill(resident.ldpcBits, 0, uint64(partII.length*4), 0); err != nil {
		return result, fmt.Errorf("jabcode: clear GPU metadata Part II codeword: %w", err)
	}
	if err := recorder.Barrier(resident.ldpcBits); err != nil {
		return result, fmt.Errorf("jabcode: synchronize GPU metadata Part II codeword: %w", err)
	}
	if err := stage(resident.metadataPart2Kernel, resident.metadataPart2Bindings, "Part II"); err != nil {
		return result, err
	}
	if err := resident.recordMetadataCorrection(recorder, partII); err != nil {
		return result, err
	}
	if err := stage(resident.metadataFinishKernel, resident.metadataFinishBindings, "fields"); err != nil {
		return result, err
	}

	record := make([]byte, gpuMetadataRecordWords*4)
	phaseprobe.Count("download.metadata_record", len(record))
	if err := recorder.Download(resident.metadataRecord, 0, record); err != nil {
		return result, fmt.Errorf("jabcode: record GPU metadata download: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return result, fmt.Errorf("jabcode: run GPU metadata walk: %w", err)
	}
	return gpuMetadataResult(record)
}

// gpuMetadataWalker holds the resident walk to the route context that produced
// the grid, so a detector outliving its context declines instead of reading a
// buffer a later route has overwritten.
type gpuMetadataWalker struct {
	resident *gpuResidentBinarizer
	epoch    *atomic.Uint64
	lease    uint64
}

func (walker gpuMetadataWalker) WalkPrimaryMetadata(
	matrix *core.Bitmap,
	symbol *core.DecodedSymbol,
) (core.PrimaryMetadata, error) {
	if walker.epoch.Load() != walker.lease {
		return core.PrimaryMetadata{}, fmt.Errorf("jabcode: GPU route context was released before the metadata walk")
	}
	return walker.resident.WalkPrimaryMetadata(matrix, symbol)
}

// WalkPrimaryMetadata answers the host's metadata question off the resident
// grid. The colour modes the device stages do not implement decline here rather
// than resolving to something plausible, because the host walk covers them and
// a wrong colour mode is not a failed read, it is a wrong one.
func (resident *gpuResidentBinarizer) WalkPrimaryMetadata(
	matrix *core.Bitmap,
	symbol *core.DecodedSymbol,
) (core.PrimaryMetadata, error) {
	var meta core.PrimaryMetadata
	if resident == nil || symbol == nil || matrix == nil {
		return meta, fmt.Errorf("jabcode: GPU metadata request is incomplete")
	}
	resident.mu.Lock()
	owned := matrix == resident.sampledGrid
	resident.mu.Unlock()
	if !owned {
		fmt.Fprintf(os.Stderr, "GRIDTRACE metadata-refused %p %dx%d\n", matrix, matrix.Width, matrix.Height)
		return meta, fmt.Errorf("jabcode: GPU metadata walk was asked about another sample")
	}

	walk, err := resident.WalkMetadata(image.Pt(matrix.Width, matrix.Height), symbol.WireVariant)
	if err != nil {
		return meta, err
	}
	if walk.Unsupported {
		return meta, fmt.Errorf("jabcode: GPU metadata walk does not cover %d colours", walk.Colors)
	}
	return core.PrimaryMetadata{
		Defaulted:        walk.Defaulted,
		Rejected:         walk.Rejected,
		NC:               walk.NC,
		Colors:           walk.Colors,
		SideVersion:      walk.SideVersion,
		ECL:              walk.ECL,
		MaskType:         walk.MaskType,
		Palette:          walk.Palette,
		MetadataModules:  walk.ModuleCount,
		PartISyndromeOK:  walk.PartISyndromeOK,
		PartIISyndromeOK: walk.PartIISyndromeOK,
	}, nil
}

// gpuMetadataResult reads the device's record. Every field is already
// interpreted there; nothing here re-derives anything from module data.
func gpuMetadataResult(record []byte) (gpuMetadataWalk, error) {
	var result gpuMetadataWalk
	word := func(index int) uint32 {
		return binary.LittleEndian.Uint32(record[index*4:])
	}
	result.ModuleCount = int(word(gpuMetadataRecordModules))
	result.NC = int(word(gpuMetadataRecordNC))
	// Read before the status is acted on, so an unsupported walk still reports
	// which colour mode it declined. Leaving it unset made every such decline
	// read as "does not cover 0 colours", which names no mode at all and so
	// cannot say what extending the walk would have to cover.
	result.Colors = int(word(gpuMetadataRecordColors))
	switch word(gpuMetadataRecordStatus) {
	case gpuMetadataStatusDefault:
		result.Defaulted = true
		result.NC = 0
		return result, nil
	case gpuMetadataStatusUnsupported:
		result.Unsupported = true
		return result, nil
	case gpuMetadataStatusFailed:
		return result, fmt.Errorf("jabcode: GPU metadata walk left the symbol")
	case gpuMetadataStatusSizeMismatch, gpuMetadataStatusECCOrder:
		// The symbol declared a shape its own sample contradicts. The host has
		// ladders for both, so this is a rejected interpretation rather than an
		// error, and it still carries what it read.
		result.Rejected = true
	}

	result.SideVersion = image.Pt(int(word(gpuMetadataRecordVersionX)), int(word(gpuMetadataRecordVersionY)))
	result.ECL = image.Pt(int(word(gpuMetadataRecordECLX)), int(word(gpuMetadataRecordECLY)))
	result.MaskType = int(word(gpuMetadataRecordMask))
	result.PartISyndromeOK = word(gpuMetadataRecordSyndrome1) == 0
	result.PartIISyndromeOK = word(gpuMetadataRecordSyndrome2) == 0

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
