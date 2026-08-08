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
	gpuPayloadParamThresholds = 30
	gpuPayloadParamExtremes   = 42
	gpuPayloadParamPalette    = 50
	gpuPayloadParamWords      = 192
)

// gpuPayloadMaxColors bounds the colour modes the device chain classifies. ISO
// admits four and eight colours only, and above eight the host classifier
// switches to absolute palette distance over interpolated representatives,
// which is a different classifier rather than a wider loop.
const gpuPayloadMaxColors = 8

// gpuPayloadGeneratorLCG selects the C-family 64-bit generator for the
// deinterleaving permutation; zero is the ISO one.
const gpuPayloadGeneratorLCG = 1

// gpuPayloadMapWords is one entry per module of the largest legal symbol.
const gpuPayloadMapWords = gpuSampleMaxSide * gpuSampleMaxSide

// gpuPayloadMaxBits bounds the codeword the chain can build: every module of the
// largest symbol carrying the most bits the device classifier admits.
const gpuPayloadMaxBits = gpuPayloadMapWords * 3

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
	resident.payloadMapBindings, err = resident.payloadMapKernel.NewBindings(
		vulki.BindBuffer(0, resident.payloadParams),
		vulki.BindBuffer(1, resident.payloadMap),
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
// unmasking, bit expansion, deinterleaving and hard LDPC correction, in one
// submission whose only result is the message.
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
	if err := recorder.Update(resident.ldpcParams, 0, params[:]); err != nil {
		return nil, false, fmt.Errorf("jabcode: update GPU payload correction parameters: %w", err)
	}
	if err := recorder.Barrier(resident.payloadParams, resident.ldpcRows, resident.ldpcParams); err != nil {
		return nil, false, fmt.Errorf("jabcode: synchronize GPU payload inputs: %w", err)
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
		if err := recorder.Dispatch(
			resident.payloadPermuteKernel,
			resident.payloadPermuteBindings,
			vulki.Workgroups{X: 1, Y: 1, Z: 1},
		); err != nil {
			return nil, false, fmt.Errorf("jabcode: dispatch GPU deinterleaving permutation: %w", err)
		}
	}
	if err := recorder.Barrier(resident.payloadMap, resident.payloadPermutation); err != nil {
		return nil, false, fmt.Errorf("jabcode: synchronize GPU payload data map: %w", err)
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
		return nil, false, fmt.Errorf("jabcode: synchronize GPU payload correction: %w", err)
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
	if rebuildPermutation {
		resident.permutationLength = shape.gross
		resident.permutationGenerator = shape.generator
	}
	// Only the net message bits come back, which is the whole point of the
	// chain: the codeword, the classifications and the grid all stay resident.
	return gpuLDPCResult(shape.ldpc, out, shape.net)
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
	put(gpuPayloadParamSymbolType, 0)
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
	for i := range colors * 4 * copies {
		putFloat(gpuPayloadParamPalette+i, request.NormalizedPalette[i])
	}
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
