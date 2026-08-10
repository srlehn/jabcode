//go:build !js

package detect

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"image"
	"math"
	"math/bits"
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

//go:embed shaders/metadata_payload.wgsl
var metadataPayloadWGSL string

// gpuMetadataLDPCRowWords holds the larger fixed metadata code. Part II has
// nineteen rows with at most nineteen unique columns per row; rounding the 361
// word ceiling up leaves space for either generator without retaining a full
// payload-sized row set in every route context.
const gpuMetadataLDPCRowWords = 512

// Parameter word indices, matching the metadata shaders.
const (
	gpuMetadataParamSideX = 0
	gpuMetadataParamSideY = 1

	// The colour mode a symbol carrying no explicit metadata is read in, so
	// the default ladder is a format constant the host states rather than a
	// number the kernel knows.
	gpuMetadataParamDefaultNC   = 2
	gpuMetadataParamGenerator   = 3
	gpuMetadataParamDefaultWC   = 4
	gpuMetadataParamDefaultWR   = 5
	gpuMetadataParamDefaultMask = 6

	// One description per colour mode, indexed by NC, so the kernel reads a
	// mode's shape rather than naming colour counts. A mode the host did not
	// describe has zero copies, and the walk declines it: which modes the
	// device covers is then a host table, not a kernel branch.
	gpuMetadataParamMode = 8

	// Where each mode's palette placement starts inside the placement region.
	gpuMetadataParamPlacement = gpuMetadataParamMode +
		gpuMetadataModeCount*gpuMetadataModeWords
	gpuMetadataParamPayload       = gpuMetadataParamPlacement + gpuMetadataPlacementWords
	gpuMetadataParamPayloadAPNumX = gpuMetadataParamPayload
	gpuMetadataParamPayloadAPNumY = gpuMetadataParamPayload + 1
	gpuMetadataParamPayloadAPPosX = gpuMetadataParamPayload + 2
	gpuMetadataParamPayloadAPPosY = gpuMetadataParamPayload + 11
	gpuMetadataParamWords         = gpuMetadataParamPayload + 20
)

// Fields of one mode description, matching metadata_palette.wgsl.
const (
	gpuMetadataModeCopies    = 0
	gpuMetadataModeSlots     = 1
	gpuMetadataModeFinder    = 2
	gpuMetadataModeBase      = 3
	gpuMetadataModeThreshold = 4
	gpuMetadataModeWords     = 5

	// NC spans zero to seven, so the table covers every encodable mode
	// including the ones no host table describes.
	gpuMetadataModeCount = 8
)

// gpuMetadataPlacementWords sizes the placement region for every colour mode a
// JAB symbol can declare, so admitting a mode later is a host table entry rather
// than a parameter-layout change. A mode contributes its copies times the number
// of palette slots it reads from the strip, which tops out at 64 because the
// higher modes interpolate the rest.
const gpuMetadataPlacementWords = 4*4 + 4*8 + 2*16 + 2*32 + 2*64 + 2*64 + 2*64

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
	gpuMetadataRecordNormalized = gpuMetadataRecordPalette + gpuMetadataMaxPaletteEntries*3
	gpuMetadataRecordThresholds = gpuMetadataRecordNormalized + gpuMetadataNormalizedEntries*4
	gpuMetadataRecordWords      = gpuMetadataRecordThresholds + 3*spec.ColorPaletteNumber
)

// gpuMetadataMaxPaletteEntries is the largest palette the walk holds, in
// entries: the colour count times the copy count, maximized over the modes the
// palette kernel accepts. The higher modes embed fewer copies, so this is not
// simply the largest colour count times the largest copy count.
//
// It is the reconstructed size rather than the embedded one. A 256-colour symbol
// carries 64 representatives, but the kernel interpolates the rest in place
// before Part II is classified against them.
const gpuMetadataMaxPaletteEntries = 256 * 2

// gpuMetadataNormalizedEntries sizes the normalized region, which only the
// modes at or below eight colours have. Above eight the classifier ranks
// absolute distance against the palette bytes themselves and never reads a
// normalized entry, so sizing this for the whole colour range would reserve
// thousands of words nothing writes.
const gpuMetadataNormalizedEntries = 8 * spec.ColorPaletteNumber

// gpuMetadataRecordCrossWords is how much of the record crosses to the host on
// an ordinary walk: the header and a palette of at most eight colours.
//
// The rest never crosses. The normalized entries and the black thresholds are
// derived from the palette, and the host rederives both in its own arithmetic
// rather than adopting the device's, so shipping them would be paying for bytes
// that are then discarded.
//
// A higher colour mode has a palette longer than this, and fetches the rest in
// a second pass keyed on the mode the first one reported. Sizing the ordinary
// download for the largest mode instead would make every four- and
// eight-colour read pay for a palette no symbol in the corpus has.
const gpuMetadataRecordCrossWords = gpuMetadataRecordPalette +
	gpuMetadataNormalizedEntries*3

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
	resident.metadataRows, err = resident.device.NewBuffer(gpuMetadataLDPCRowWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU metadata parity rows: %w", err)
	}
	resident.metadataLDPCBindings, err = resident.ldpcKernel.NewBindings(
		vulki.BindBuffer(0, resident.metadataRows),
		vulki.BindBuffer(1, resident.ldpcBits),
		vulki.BindBuffer(2, resident.ldpcParams),
		vulki.BindBuffer(3, resident.ldpcNet),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU metadata correction: %w", err)
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
	resident.metadataPayloadKernel, err = resident.kernels.metadataPayload()
	if err != nil {
		return err
	}
	resident.metadataPayloadBindings, err = resident.metadataPayloadKernel.NewBindings(
		vulki.BindBuffer(0, resident.metadataParams),
		vulki.BindBuffer(1, resident.metadataRecord),
		vulki.BindBuffer(2, resident.payloadParams),
		vulki.BindBuffer(3, resident.ldpcParams),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU metadata payload control: %w", err)
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
	if err := gpuLDPCUploadRows(recorder, resident.metadataRows, plan); err != nil {
		return err
	}
	params := gpuLDPCParams(plan)
	if err := recordGPUUpdate(
		recorder, "upload.ldpc_params", resident.ldpcParams, 0, params[:],
	); err != nil {
		return fmt.Errorf("jabcode: update GPU metadata correction parameters: %w", err)
	}
	if err := recorder.Barrier(resident.metadataRows, resident.ldpcParams); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU metadata correction inputs: %w", err)
	}
	if err := recorder.Dispatch(
		resident.ldpcKernel,
		resident.metadataLDPCBindings,
		vulki.Workgroups{X: uint32(plan.blocks), Y: 1, Z: 1},
	); err != nil {
		return fmt.Errorf("jabcode: dispatch GPU metadata correction: %w", err)
	}
	if err := recorder.Barrier(resident.ldpcNet, resident.ldpcBits); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU metadata correction: %w", err)
	}
	return nil
}

// gpuMetadataDeviceColorModes are the colour modes the device chain covers for
// a wire variant. Everything a mode's walk needs - how many palette copies it
// embeds, how many slots it reads from the strip, how many colours its finder
// carries and which black-threshold rule applies - is a property of the format,
// so the host states it here and the kernel reads it. A mode absent from the
// list is declined, which is what makes admitting one a table entry.
//
// ISO admits four and eight colours and nothing else, and the host observation
// rejects any other mode outright on that variant. A device answering where the
// host rejects would decode symbols the untagged reader is not allowed to read,
// so the conformance rule is applied here rather than left to the kernel.
//
// 128 and 256 embed 64 representatives and the kernel reconstructs the rest, so
// they cost the same walk with a wider placement table.
func gpuMetadataDeviceColorModes(variant wire.Variant) []int {
	if variant == wire.ISO23634 {
		return []int{4, 8}
	}
	return []int{4, 8, 16, 32, 64, 128, 256}
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
	put(gpuMetadataParamDefaultNC, uint32(spec.DefaultModuleColorMode))
	defaultECL := spec.ECCWeights[spec.DefaultECCLevel]
	put(gpuMetadataParamDefaultWC, uint32(defaultECL[0]))
	put(gpuMetadataParamDefaultWR, uint32(defaultECL[1]))
	put(gpuMetadataParamDefaultMask, uint32(spec.DefaultMaskingReference))
	if !variant.UsesISO23634Base() {
		put(gpuMetadataParamGenerator, gpuPayloadGeneratorLCG)
	}
	vx, vy := spec.SizeToVersion(side.X)-1, spec.SizeToVersion(side.Y)-1
	put(gpuMetadataParamPayloadAPNumX, uint32(tables.APNum[vx]))
	put(gpuMetadataParamPayloadAPNumY, uint32(tables.APNum[vy]))
	for i, position := range tables.APPos[vx] {
		put(gpuMetadataParamPayloadAPPosX+i, uint32(position))
	}
	for i, position := range tables.APPos[vy] {
		put(gpuMetadataParamPayloadAPPosY+i, uint32(position))
	}
	base := 0
	for _, colors := range gpuMetadataDeviceColorModes(variant) {
		nc := bits.TrailingZeros(uint(colors)) - 1
		copies := spec.PaletteCopies(colors)
		slots := min(colors, spec.PalettePlacementSlots)
		mode := gpuMetadataParamMode + nc*gpuMetadataModeWords
		put(mode+gpuMetadataModeCopies, uint32(copies))
		put(mode+gpuMetadataModeSlots, uint32(slots))
		put(mode+gpuMetadataModeFinder, uint32(spec.PaletteFinderColors(colors)))
		put(mode+gpuMetadataModeBase, uint32(base))
		// Only four and eight colours have a black-threshold rule at all; the
		// higher modes leave the thresholds zero on the host too, so the rule
		// is named rather than derived from the colour count.
		if colors == 4 || colors == 8 {
			put(mode+gpuMetadataModeThreshold, uint32(colors))
		}
		for copy := range copies {
			for slot := range slots {
				put(gpuMetadataParamPlacement+base+copy*slots+slot, uint32(
					tables.PrimaryPalettePlacementIndexVariant(copy, slot, colors, variant)%colors))
			}
		}
		base += copies * slots
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
	resident.payloadControlReady = false

	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return result, fmt.Errorf("jabcode: create GPU metadata recorder: %w", err)
	}
	defer recorder.Abort()

	if err := resident.recordMetadataWalk(recorder, side, variant, partI, partII); err != nil {
		return result, err
	}

	record := make([]byte, resident.metadataRecordFetchWords()*4)
	phaseprobe.Count("download.metadata_record", len(record))
	if err := recorder.Download(resident.metadataRecord, 0, record); err != nil {
		return result, fmt.Errorf("jabcode: record GPU metadata download: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return result, fmt.Errorf("jabcode: run GPU metadata walk: %w", err)
	}
	record, err = resident.fetchMetadataPaletteTail(record)
	if err != nil {
		return result, err
	}
	result, err = gpuMetadataResult(record)
	if err == nil && !result.Unsupported && !result.Rejected {
		resident.payloadControlReady = true
		resident.payloadControlVariant = variant
	}
	return result, err
}

func (resident *gpuResidentBinarizer) recordMetadataWalk(
	recorder *vulki.Recorder,
	side image.Point,
	variant wire.Variant,
	partI, partII gpuLDPCPlan,
) error {
	params := gpuMetadataParams(side, variant)
	if err := recordGPUUpdate(
		recorder, "upload.metadata_params", resident.metadataParams, 0, params[:],
	); err != nil {
		return fmt.Errorf("jabcode: update GPU metadata parameters: %w", err)
	}
	// A classifier writes only the bits it resolves, so each part starts clear
	// rather than carrying what the previous metadata walk left in the buffer.
	if err := recorder.Fill(resident.ldpcBits, 0, uint64(partI.length*4), 0); err != nil {
		return fmt.Errorf("jabcode: clear GPU metadata codeword: %w", err)
	}
	if err := recorder.Barrier(resident.metadataParams, resident.ldpcBits); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU metadata inputs: %w", err)
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
		return err
	}
	if err := resident.recordMetadataCorrection(recorder, partI); err != nil {
		return err
	}
	if err := stage(resident.metadataPaletteKernel, resident.metadataPaletteBindings, "palette"); err != nil {
		return err
	}
	if err := recorder.Fill(resident.ldpcBits, 0, uint64(partII.length*4), 0); err != nil {
		return fmt.Errorf("jabcode: clear GPU metadata Part II codeword: %w", err)
	}
	if err := recorder.Barrier(resident.ldpcBits); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU metadata Part II codeword: %w", err)
	}
	if err := stage(resident.metadataPart2Kernel, resident.metadataPart2Bindings, "Part II"); err != nil {
		return err
	}
	if err := resident.recordMetadataCorrection(recorder, partII); err != nil {
		return err
	}
	if err := stage(resident.metadataFinishKernel, resident.metadataFinishBindings, "fields"); err != nil {
		return err
	}
	if err := recorder.Dispatch(
		resident.metadataPayloadKernel,
		resident.metadataPayloadBindings,
		vulki.Workgroups{X: 1, Y: 1, Z: 1},
	); err != nil {
		return fmt.Errorf("jabcode: dispatch GPU metadata payload control: %w", err)
	}
	if err := recorder.Barrier(resident.payloadParams, resident.ldpcParams); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU metadata payload control: %w", err)
	}
	return nil
}

// fetchMetadataPaletteTail extends a downloaded record when the mode it reports
// embeds a palette longer than the ordinary download carries.
//
// It is a second round trip, and it is deliberately conditional rather than
// folded into the first: the colour mode is only known once the record is read,
// and downloading for the largest mode unconditionally would charge every four-
// and eight-colour symbol for a palette it does not have. A high-colour symbol
// pays one small extra download here instead of the whole module grid later.
func (resident *gpuResidentBinarizer) fetchMetadataPaletteTail(record []byte) ([]byte, error) {
	if len(record) < (gpuMetadataRecordColors+1)*4 {
		return record, nil
	}
	if binary.LittleEndian.Uint32(record[gpuMetadataRecordStatus*4:]) != gpuMetadataStatusOK &&
		binary.LittleEndian.Uint32(record[gpuMetadataRecordStatus*4:]) != gpuMetadataStatusSizeMismatch &&
		binary.LittleEndian.Uint32(record[gpuMetadataRecordStatus*4:]) != gpuMetadataStatusECCOrder {
		return record, nil
	}
	colors := int(binary.LittleEndian.Uint32(record[gpuMetadataRecordColors*4:]))
	if colors <= 0 || colors > 1<<gpuMetadataModeCount {
		return record, nil
	}
	want := gpuMetadataRecordPalette + colors*spec.PaletteCopies(colors)*3
	if want*4 <= len(record) {
		return record, nil
	}
	tail := make([]byte, want*4-len(record))
	phaseprobe.Count("download.metadata_palette_tail", len(tail))
	if err := resident.metadataRecord.DownloadAt(uint64(len(record)), tail); err != nil {
		return nil, fmt.Errorf("jabcode: download GPU metadata palette tail: %w", err)
	}
	return append(record, tail...), nil
}

// metadataRecordFetchWords is how much of the record a walk brings back. A
// decode wants the crossing region alone; the cross-check that holds the
// device's derived palette to the host's asks for the whole buffer, which is
// the only thing that ever reads past it.
func (resident *gpuResidentBinarizer) metadataRecordFetchWords() int {
	if resident != nil && resident.metadataFetchDerived {
		return gpuMetadataRecordWords
	}
	return gpuMetadataRecordCrossWords
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
		return meta, fmt.Errorf("jabcode: GPU metadata walk was asked about another sample")
	}

	walk, err := resident.WalkMetadata(image.Pt(matrix.Width, matrix.Height), symbol.WireVariant)
	if err != nil {
		return meta, err
	}
	if walk.Unsupported {
		return meta, fmt.Errorf("jabcode: GPU metadata walk does not cover %d colours", walk.Colors)
	}
	return gpuPrimaryMetadata(walk), nil
}

func gpuPrimaryMetadata(walk gpuMetadataWalk) core.PrimaryMetadata {
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
	}
}

// gpuMetadataResult reads the device's record. Every field is already
// interpreted there; nothing here re-derives anything from module data.
func gpuMetadataResult(record []byte) (gpuMetadataWalk, error) {
	var result gpuMetadataWalk
	// A short record reads as zeros rather than panicking. How much of the
	// record was fetched depends on the colour mode the device reported, so a
	// mode this side sizes differently from that one is a wrong answer to
	// correct, never a crash in a decode of arbitrary input.
	word := func(index int) uint32 {
		if index < 0 || (index+1)*4 > len(record) {
			return 0
		}
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
		// The walk continues through the palette in the default colour mode and
		// stops before Part II, exactly as the host ladder does, so this carries
		// a palette and a mode rather than being a bare decline.
		result.Defaulted = true
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

	// A defaulted walk stops before Part II, which is what writes the shape, so
	// the record's shape words are whatever the previous walk left there. The
	// caller takes the format's defaults instead, and reporting nothing is what
	// keeps a stale version from ever being mistaken for a read one.
	if !result.Defaulted {
		result.SideVersion = image.Pt(int(word(gpuMetadataRecordVersionX)), int(word(gpuMetadataRecordVersionY)))
		result.ECL = image.Pt(int(word(gpuMetadataRecordECLX)), int(word(gpuMetadataRecordECLY)))
		result.MaskType = int(word(gpuMetadataRecordMask))
		result.PartISyndromeOK = word(gpuMetadataRecordSyndrome1) == 0
		result.PartIISyndromeOK = word(gpuMetadataRecordSyndrome2) == 0
	}

	// Copies, not the corner count: the modes above eight colours embed two
	// palettes rather than four, and reading four of them walks off the record.
	entries := result.Colors * spec.PaletteCopies(result.Colors)
	result.Palette = make([]byte, entries*3)
	for i := range result.Palette {
		result.Palette[i] = byte(word(gpuMetadataRecordPalette + i))
	}
	// Whatever the caller fetched beyond the palette. A decode never asks for
	// the derived region, so this fills only when something deliberately did.
	if len(record) >= gpuMetadataRecordWords*4 {
		// The thresholds are one triple per spatial corner whatever the mode,
		// but only the modes classified by direction have normalized entries at
		// all. Sixteen colours in two copies fills the region exactly, so the
		// bound has to be the classifier's and not the region's.
		if result.Colors <= 8 {
			result.Normalized = make([]float64, entries*4)
			for i := range result.Normalized {
				result.Normalized[i] = float64(math.Float32frombits(word(gpuMetadataRecordNormalized + i)))
			}
		}
		result.Thresholds = make([]float64, 3*spec.ColorPaletteNumber)
		for i := range result.Thresholds {
			result.Thresholds[i] = float64(math.Float32frombits(word(gpuMetadataRecordThresholds + i)))
		}
	}
	return result, nil
}
