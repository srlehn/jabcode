//go:build !js

package detect

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"math"
	"sync/atomic"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/ecc"
	"github.com/srlehn/jabcode/internal/phaseprobe"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/tables"
)

//go:embed shaders/payload_map.wgsl
var payloadMapWGSL string

//go:embed shaders/payload_permute.wgsl
var payloadPermuteWGSL string

//go:embed shaders/payload_bits.wgsl
var payloadBitsWGSL string

//go:embed shaders/payload_reliability.wgsl
var payloadReliabilityWGSL string

//go:embed shaders/admission_fixed.wgsl
var admissionFixedWGSL string

// Parameter word indices, shared by the three payload kernels.
const (
	gpuPayloadParamSideX      = 0
	gpuPayloadParamSideY      = 1
	gpuPayloadParamMeta       = 2
	gpuPayloadParamColors     = 3
	gpuPayloadParamBits       = 4
	gpuPayloadParamMask       = 5
	gpuPayloadParamAPNumX     = 6
	gpuPayloadParamAPNumY     = 7
	gpuPayloadParamGrossBits  = 8
	gpuPayloadParamGenerator  = 9
	gpuPayloadParamSymbolType = 11
	gpuPayloadParamAPPosX     = 12
	gpuPayloadParamAPPosY     = 21
	gpuPayloadParamCopies     = 10
	gpuPayloadParamThresholds = 30
	gpuPayloadParamExtremes   = 42
	gpuPayloadParamPalette    = 50
	// The palette bytes, read only by the modes above eight colours, which
	// classify by absolute distance against them rather than by direction
	// against the normalized entries.
	gpuPayloadParamPaletteBytes = 178
	gpuPayloadParamAdmission    = gpuPayloadParamPaletteBytes +
		gpuPayloadMaxColors*3*2
	// The data-map fold writes these from the exact number of unreserved
	// modules it just numbered. Keeping the result beside the classifier inputs
	// lets later resident stages select their correction shape without asking
	// the host to count the map or interpret metadata first.
	gpuPayloadParamDataModules = gpuPayloadParamAdmission + 1
	gpuPayloadParamNetBits     = gpuPayloadParamAdmission + 2
	gpuPayloadParamWC          = gpuPayloadParamAdmission + 3
	gpuPayloadParamWR          = gpuPayloadParamAdmission + 4
	gpuPayloadParamWords       = gpuPayloadParamAdmission + 5
)

// gpuPayloadMaxColors bounds the colour modes the device chain classifies, which
// is now every mode a JAB symbol can declare. Above eight colours the classifier
// ranks absolute palette distance instead of normalized direction, and the
// palette it ranks against arrives already reconstructed.
const gpuPayloadMaxColors = 256

// gpuPayloadGeneratorLCG selects the C-family 64-bit generator for the
// deinterleaving permutation; zero is the ISO one.
const gpuPayloadGeneratorLCG = 1

// gpuPayloadMapWords is one entry per module of the largest legal symbol.
const gpuPayloadMapWords = gpuSampleMaxSide * gpuSampleMaxSide

// gpuPayloadMaxBits bounds the codeword the chain can build: every module of the
// largest symbol carrying the most bits the device classifier admits.
const gpuPayloadMaxBits = gpuPayloadMapWords * 8

// gpuPayloadRetainedBytes is what the payload chain holds on the device: the
// data map, the deinterleaving permutation, and the parameter block.
const gpuPayloadRetainedBytes = (gpuPayloadMapWords + gpuPayloadMaxBits + gpuPayloadParamWords) * 4

// gpuMetadataRetainedBytes is what the metadata walk holds on the device: its
// parameter block and the record it interprets the symbol into. Both are fixed
// and tiny; the walk reads the module grid and writes the corrector's codeword,
// neither of which it owns.
const gpuMetadataRetainedBytes = (gpuMetadataParamWords + gpuMetadataRecordWords) * 4

// initializePayload allocates the payload chain's buffers and compiles its
// kernels with the rest of the resident stage set, so the compiles land in
// warm-up rather than on the decode call.
func (resident *gpuResidentBinarizer) initializePayload() error {
	var err error
	resident.payloadParams, err = resident.device.NewBuffer(gpuPayloadParamWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU payload parameters: %w", err)
	}
	resident.payloadMap, err = resident.device.NewBuffer(gpuPayloadMapWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU payload data map: %w", err)
	}
	resident.payloadPermutation, err = resident.device.NewBuffer(gpuPayloadMaxBits * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU deinterleaving permutation: %w", err)
	}
	if resident.payloadMapKernel, err = resident.kernels.payloadMap(); err != nil {
		return err
	}
	if resident.payloadPermuteKernel, err = resident.kernels.payloadPermute(); err != nil {
		return err
	}
	if resident.payloadBitsKernel, err = resident.kernels.payloadBits(); err != nil {
		return err
	}
	if resident.payloadReliabilityKernel, err = resident.kernels.payloadReliability(); err != nil {
		return err
	}
	if resident.admissionFixedKernel, err = resident.kernels.admissionFixed(); err != nil {
		return err
	}
	resident.payloadMapBindings, err = resident.payloadMapKernel.NewBindings(
		vulki.BindBuffer(0, resident.payloadParams),
		vulki.BindBuffer(1, resident.payloadMap),
		vulki.BindBuffer(2, resident.ldpcParams),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU payload data map: %w", err)
	}
	resident.payloadPermuteBindings, err = resident.payloadPermuteKernel.NewBindings(
		vulki.BindBuffer(0, resident.payloadParams),
		vulki.BindBuffer(1, resident.payloadPermutation),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU deinterleaving permutation: %w", err)
	}
	resident.payloadBitsBindings, err = resident.payloadBitsKernel.NewBindings(
		vulki.BindBuffer(0, resident.payloadParams),
		vulki.BindBuffer(1, resident.sampleResult),
		vulki.BindBuffer(2, resident.payloadMap),
		vulki.BindBuffer(3, resident.payloadPermutation),
		vulki.BindBuffer(4, resident.ldpcBits),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU payload classification: %w", err)
	}
	resident.payloadReliabilityBindings, err = resident.payloadReliabilityKernel.NewBindings(
		vulki.BindBuffer(0, resident.payloadParams),
		vulki.BindBuffer(1, resident.sampleResult),
		vulki.BindBuffer(2, resident.payloadMap),
		vulki.BindBuffer(3, resident.payloadPermutation),
		vulki.BindBuffer(4, resident.ldpcReliability),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU payload reliability: %w", err)
	}
	resident.admissionFixedBindings, err = resident.admissionFixedKernel.NewBindings(
		vulki.BindBuffer(0, resident.payloadParams),
		vulki.BindBuffer(1, resident.sampleResult),
		vulki.BindBuffer(2, resident.ldpcParams),
		vulki.BindBuffer(3, resident.ldpcNet),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU fixed-pattern admission: %w", err)
	}
	resident.ldpcSoftPrepareBindings, err = resident.ldpcSoftPrepareKernel.NewBindings(
		vulki.BindBuffer(0, resident.payloadParams),
		vulki.BindBuffer(1, resident.ldpcParams),
		vulki.BindBuffer(2, resident.ldpcNet),
		vulki.BindBuffer(3, resident.ldpcSoftIndirect),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU soft LDPC preparation: %w", err)
	}
	return nil
}

// gpuPayloadCorrector is the route's handle on the resident payload chain. It
// carries the lease its detector was built under, so a correction attempted
// after the route context was released declines instead of reading whatever the
// next route left in the grid buffer.
type gpuPayloadCorrector struct {
	resident *gpuResidentBinarizer
	epoch    *atomic.Uint64
	lease    uint64
}

func (gpuPayloadCorrector) SupportsFixedPatternAdmission() bool { return true }

func (*gpuResidentBinarizer) SupportsFixedPatternAdmission() bool { return true }

func (corrector gpuPayloadCorrector) CorrectSymbolPayload(
	request core.PayloadRequest,
) (dec []byte, ok bool, err error) {
	if corrector.epoch.Load() != corrector.lease {
		return nil, false, fmt.Errorf("jabcode: GPU route context was released before payload correction")
	}
	return corrector.resident.CorrectSymbolPayload(request)
}

// gpuPayloadShape is everything the payload chain derives from a request before
// it records anything, so an unsupported symbol declines without touching the
// device.
type gpuPayloadShape struct {
	params    [gpuPayloadParamWords * 4]byte
	ldpc      gpuLDPCPlan
	net       int
	gross     int
	generator uint32
}

// CorrectSymbolPayload runs the whole chain between the sampled module grid and
// the corrected message on the device: the data map, palette classification,
// unmasking, bit expansion, deinterleaving, hard LDPC correction, and a
// soft-decision retry for only the failed sub-blocks, in one submission whose
// only result is the message.
//
// It declines - with an error rather than a failed read - for any symbol the
// device chain does not cover, so the host chain answers those exactly as
// before. The grid identity check is what makes the chain safe to run at all:
// the resident sampler holds one grid, and a correction asked about any other
// sample would silently classify the wrong symbol.
func (resident *gpuResidentBinarizer) CorrectSymbolPayload(
	request core.PayloadRequest,
) (dec []byte, ok bool, err error) {
	if resident == nil || resident.closed || resident.payloadBitsBindings == nil {
		return nil, false, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	shape, err := gpuPayloadShapeOf(request)
	if err != nil {
		return nil, false, err
	}

	resident.mu.Lock()
	defer resident.mu.Unlock()
	if request.Matrix == nil || request.Matrix != resident.sampledGrid {
		return nil, false, fmt.Errorf("jabcode: GPU payload correction was asked about another sample")
	}

	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return nil, false, fmt.Errorf("jabcode: create GPU payload recorder: %w", err)
	}
	defer recorder.Abort()

	if err := recorder.Update(resident.payloadParams, 0, shape.params[:]); err != nil {
		return nil, false, fmt.Errorf("jabcode: update GPU payload parameters: %w", err)
	}
	if err := gpuLDPCUploadRows(recorder, resident.ldpcRows, shape.ldpc); err != nil {
		return nil, false, err
	}
	params := gpuLDPCParams(shape.ldpc)
	binary.LittleEndian.PutUint32(
		params[gpuLDPCParamAdmission*4:],
		binary.LittleEndian.Uint32(shape.params[gpuPayloadParamAdmission*4:]),
	)
	if err := recorder.Update(resident.ldpcParams, 0, params[:]); err != nil {
		return nil, false, fmt.Errorf("jabcode: update GPU payload correction parameters: %w", err)
	}
	if err := recorder.Barrier(resident.payloadParams, resident.ldpcRows, resident.ldpcParams); err != nil {
		return nil, false, fmt.Errorf("jabcode: synchronize GPU payload inputs: %w", err)
	}
	if request.RequireFixedPatternAgreement {
		if err := recorder.Dispatch(
			resident.admissionFixedKernel,
			resident.admissionFixedBindings,
			vulki.Workgroups{X: 1, Y: 1, Z: 1},
		); err != nil {
			return nil, false, fmt.Errorf("jabcode: dispatch GPU fixed-pattern admission: %w", err)
		}
		if err := recorder.Barrier(resident.payloadParams, resident.ldpcParams, resident.ldpcNet); err != nil {
			return nil, false, fmt.Errorf("jabcode: synchronize GPU fixed-pattern admission: %w", err)
		}
	}

	if err := recorder.Dispatch(
		resident.payloadMapKernel,
		resident.payloadMapBindings,
		vulki.Workgroups{X: 1, Y: 1, Z: 1},
	); err != nil {
		return nil, false, fmt.Errorf("jabcode: dispatch GPU payload data map: %w", err)
	}
	// The permutation depends only on the codeword length and the generator, so
	// it is rebuilt only when one of those changes and every other correction
	// reuses it. Recording the rebuild is not the same as running it, so the
	// cache is surrendered here and only re-established once the submission has
	// actually happened: every return between the two leaves the buffer
	// half-built, and a reuse of that would deinterleave into the wrong order
	// with nothing downstream to notice.
	rebuildPermutation := resident.permutationLength != shape.gross ||
		resident.permutationGenerator != shape.generator
	if rebuildPermutation {
		resident.permutationLength = 0
		resident.permutationGenerator = 0
	}
	if err := recorder.Barrier(resident.payloadMap, resident.payloadParams, resident.ldpcParams); err != nil {
		return nil, false, fmt.Errorf("jabcode: synchronize GPU payload data map: %w", err)
	}
	if rebuildPermutation {
		if err := recorder.Dispatch(
			resident.payloadPermuteKernel,
			resident.payloadPermuteBindings,
			vulki.Workgroups{X: 1, Y: 1, Z: 1},
		); err != nil {
			return nil, false, fmt.Errorf("jabcode: dispatch GPU deinterleaving permutation: %w", err)
		}
	}
	if err := recorder.Barrier(resident.payloadPermutation); err != nil {
		return nil, false, fmt.Errorf("jabcode: synchronize GPU payload permutation: %w", err)
	}

	modules := request.Symbol.SideSize.X * request.Symbol.SideSize.Y
	if err := recorder.Dispatch(
		resident.payloadBitsKernel,
		resident.payloadBitsBindings,
		vulki.Workgroups{X: uint32((modules + 63) / 64), Y: 1, Z: 1},
	); err != nil {
		return nil, false, fmt.Errorf("jabcode: dispatch GPU payload classification: %w", err)
	}
	if err := recorder.Barrier(resident.ldpcBits); err != nil {
		return nil, false, fmt.Errorf("jabcode: synchronize GPU payload codeword: %w", err)
	}
	if err := recorder.Dispatch(
		resident.ldpcKernel,
		resident.ldpcBindings,
		vulki.Workgroups{X: uint32(shape.ldpc.blocks), Y: 1, Z: 1},
	); err != nil {
		return nil, false, fmt.Errorf("jabcode: dispatch GPU payload correction: %w", err)
	}
	if err := recorder.Barrier(resident.ldpcNet); err != nil {
		return nil, false, fmt.Errorf("jabcode: synchronize GPU hard payload correction: %w", err)
	}
	if err := recorder.Dispatch(
		resident.ldpcSoftPrepareKernel,
		resident.ldpcSoftPrepareBindings,
		vulki.Workgroups{X: 1, Y: 1, Z: 1},
	); err != nil {
		return nil, false, fmt.Errorf("jabcode: prepare GPU soft payload dispatches: %w", err)
	}
	if err := recorder.Barrier(resident.ldpcSoftIndirect); err != nil {
		return nil, false, fmt.Errorf("jabcode: synchronize GPU soft payload dispatches: %w", err)
	}
	if err := recorder.DispatchIndirect(
		resident.ldpcSoftGraphKernel,
		resident.ldpcSoftGraphBindings,
		resident.ldpcSoftIndirect,
		gpuLDPCSoftGraphIndirectOffset,
	); err != nil {
		return nil, false, fmt.Errorf("jabcode: build GPU soft payload graph: %w", err)
	}
	if err := recorder.DispatchIndirect(
		resident.payloadReliabilityKernel,
		resident.payloadReliabilityBindings,
		resident.ldpcSoftIndirect,
		gpuLDPCSoftReliabilityIndirectOffset,
	); err != nil {
		return nil, false, fmt.Errorf("jabcode: dispatch GPU payload reliability: %w", err)
	}
	if err := recorder.Barrier(resident.ldpcSoftGraph, resident.ldpcReliability); err != nil {
		return nil, false, fmt.Errorf("jabcode: synchronize GPU soft payload inputs: %w", err)
	}
	if err := recorder.DispatchIndirect(
		resident.ldpcSoftKernel,
		resident.ldpcSoftBindings,
		resident.ldpcSoftIndirect,
		gpuLDPCSoftCorrectionIndirectOffset,
	); err != nil {
		return nil, false, fmt.Errorf("jabcode: dispatch GPU soft payload correction: %w", err)
	}
	if err := recorder.Barrier(resident.ldpcNet); err != nil {
		return nil, false, fmt.Errorf("jabcode: synchronize GPU soft payload correction: %w", err)
	}

	out := make([]byte, (shape.ldpc.blocks+shape.net)*4)
	phaseprobe.Count("download.ldpc_net", len(out))
	if err := recorder.Download(resident.ldpcNet, 0, out); err != nil {
		return nil, false, fmt.Errorf("jabcode: record GPU payload download: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		// A failed submission leaves the permutation buffer in an unknown
		// state, so the next correction rebuilds it rather than trusting it.
		resident.permutationLength = 0
		resident.permutationGenerator = 0
		return nil, false, fmt.Errorf("jabcode: run GPU payload correction: %w", err)
	}
	// Only the net message bits come back, which is the whole point of the
	// chain: the codeword, the classifications and the grid all stay resident.
	dec, ok, err = gpuLDPCResult(shape.ldpc, out, shape.net)
	if rebuildPermutation && err == nil && (!request.RequireFixedPatternAgreement || ok) {
		resident.permutationLength = shape.gross
		resident.permutationGenerator = shape.generator
	}
	return dec, ok, err
}

// gpuPayloadShapeOf resolves a request into the parameter block and correction
// plan, or reports why the device chain cannot answer it.
func gpuPayloadShapeOf(request core.PayloadRequest) (gpuPayloadShape, error) {
	var shape gpuPayloadShape
	symbol := request.Symbol
	if symbol == nil || request.Matrix == nil {
		return shape, fmt.Errorf("jabcode: GPU payload request is incomplete")
	}
	side := symbol.SideSize
	if side.X != request.Matrix.Width || side.Y != request.Matrix.Height ||
		!spec.ValidSideSize(side.X) || !spec.ValidSideSize(side.Y) ||
		side.X > gpuSampleMaxSide || side.Y > gpuSampleMaxSide {
		return shape, fmt.Errorf("jabcode: GPU payload side %dx%d is out of range", side.X, side.Y)
	}
	colors := 1 << (symbol.Meta.NC + 1)
	if symbol.Meta.NC < 0 || colors > gpuPayloadMaxColors {
		return shape, fmt.Errorf("jabcode: GPU payload correction covers up to %d colours", gpuPayloadMaxColors)
	}
	copies := spec.PaletteCopies(colors)
	if len(request.NormalizedPalette) < colors*4*copies ||
		len(request.PaletteThresholds) < 3*spec.ColorPaletteNumber ||
		len(symbol.Palette) < colors*3*copies {
		return shape, fmt.Errorf("jabcode: GPU payload palette is incomplete")
	}
	if request.DataModules <= 0 || request.MetadataModules < 0 {
		return shape, fmt.Errorf("jabcode: GPU payload module counts are out of range")
	}

	bitsPerModule := symbol.Meta.NC + 1
	wc, wr := symbol.Meta.ECL.X, symbol.Meta.ECL.Y
	if wc < 3 || wc >= wr || wr > 11 {
		return shape, fmt.Errorf("jabcode: GPU payload ECC parameters are out of range")
	}
	layout := ecc.HardBlockSplit(request.DataModules*bitsPerModule, wc, wr)
	if layout.Blocks <= 0 || layout.Blocks > gpuLDPCMaxBlocks ||
		layout.GrossSub <= 0 || layout.GrossSub > gpuLDPCMaxSub {
		return shape, fmt.Errorf("jabcode: GPU payload block layout is unsupported")
	}
	if layout.Pg > gpuPayloadMaxBits {
		return shape, fmt.Errorf("jabcode: GPU payload codeword is too long")
	}
	rows, found := ecc.ParityRows(wc, wr, layout.GrossSub, symbol.WireVariant)
	if !found {
		return shape, fmt.Errorf("jabcode: no parity rows for wc=%d wr=%d gross=%d", wc, wr, layout.GrossSub)
	}

	shape.gross = layout.Pg
	shape.net = layout.Pn
	shape.ldpc = gpuLDPCPlan{
		rows:      rows.Rows,
		rowDegree: rows.Degree,
		length:    layout.GrossSub,
		height:    rows.Height,
		rank:      rows.Rank,
		net:       layout.NetSub,
		blocks:    layout.Blocks,
	}
	if !layout.Uniform {
		tailGross := layout.TrailingGrossSub()
		tailRows, tailFound := ecc.ParityRows(wc, wr, tailGross, symbol.WireVariant)
		if !tailFound || tailGross > gpuLDPCMaxSub {
			return shape, fmt.Errorf("jabcode: no trailing parity rows for gross=%d", tailGross)
		}
		shape.ldpc.tailRows = tailRows.Rows
		shape.ldpc.tailRowDegree = tailRows.Degree
		shape.ldpc.tailLength = tailGross
		shape.ldpc.tailHeight = tailRows.Height
		shape.ldpc.tailRank = tailRows.Rank
		shape.ldpc.tailNet = tailGross * (wr - wc) / wr
	}
	if !shape.ldpc.valid() {
		return shape, fmt.Errorf("jabcode: GPU payload correction plan is out of range")
	}
	if _, err := gpuLDPCSoftPlanOf(shape.ldpc); err != nil {
		return shape, err
	}

	put := func(index int, value uint32) {
		binary.LittleEndian.PutUint32(shape.params[index*4:], value)
	}
	putFloat := func(index int, value float64) {
		put(index, math.Float32bits(float32(value)))
	}
	put(gpuPayloadParamSideX, uint32(side.X))
	put(gpuPayloadParamSideY, uint32(side.Y))
	put(gpuPayloadParamMeta, uint32(request.MetadataModules))
	put(gpuPayloadParamColors, uint32(colors))
	put(gpuPayloadParamBits, uint32(bitsPerModule))
	put(gpuPayloadParamMask, uint32(symbol.Meta.MaskType))
	put(gpuPayloadParamGrossBits, uint32(layout.Pg))
	put(gpuPayloadParamDataModules, uint32(request.DataModules))
	put(gpuPayloadParamNetBits, uint32(layout.Pn))
	put(gpuPayloadParamWC, uint32(wc))
	put(gpuPayloadParamWR, uint32(wr))
	put(gpuPayloadParamSymbolType, 0)
	if request.RequireFixedPatternAgreement {
		put(gpuPayloadParamAdmission, 1)
	}
	if !symbol.WireVariant.UsesISO23634Base() {
		shape.generator = gpuPayloadGeneratorLCG
		put(gpuPayloadParamGenerator, shape.generator)
	}

	vx, vy := spec.SizeToVersion(side.X)-1, spec.SizeToVersion(side.Y)-1
	if vx < 0 || vx >= len(tables.APNum) || vy < 0 || vy >= len(tables.APNum) {
		return shape, fmt.Errorf("jabcode: GPU payload side version is out of range")
	}
	put(gpuPayloadParamAPNumX, uint32(tables.APNum[vx]))
	put(gpuPayloadParamAPNumY, uint32(tables.APNum[vy]))
	for i, position := range tables.APPos[vx] {
		put(gpuPayloadParamAPPosX+i, uint32(position))
	}
	for i, position := range tables.APPos[vy] {
		put(gpuPayloadParamAPPosY+i, uint32(position))
	}
	for i := range 3 * spec.ColorPaletteNumber {
		putFloat(gpuPayloadParamThresholds+i, request.PaletteThresholds[i])
	}
	// The two classifiers read different things and never both apply: at or
	// below eight colours the ranking is by normalized direction, above it by
	// absolute distance against the palette bytes.
	if colors <= 8 {
		for i := range colors * 4 * copies {
			putFloat(gpuPayloadParamPalette+i, request.NormalizedPalette[i])
		}
	} else {
		for i := range colors * 3 * copies {
			put(gpuPayloadParamPaletteBytes+i, uint32(symbol.Palette[i]))
		}
	}
	put(gpuPayloadParamCopies, uint32(copies))
	if colors == 8 {
		// The eight-colour classifier separates black from white by total
		// intensity, because the two normalize to the same direction.
		for copy := range copies {
			base := colors * 3 * copy
			put(gpuPayloadParamExtremes+copy*2, paletteEntrySum(symbol.Palette, base))
			put(gpuPayloadParamExtremes+copy*2+1, paletteEntrySum(symbol.Palette, base+7*3))
		}
	}
	return shape, nil
}

func paletteEntrySum(palette []byte, at int) uint32 {
	return uint32(palette[at]) + uint32(palette[at+1]) + uint32(palette[at+2])
}
