//go:build !js

package detect

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"image"
	"math"
	"sync/atomic"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/phaseprobe"
	"github.com/srlehn/jabcode/internal/spec"
)

//go:embed shaders/primary_result.wgsl
var primaryResultWGSL string

const (
	gpuPrimaryResultMagic   = 0x4a414252
	gpuPrimaryResultVersion = 2

	gpuPrimaryResultHeaderWords  = 32
	gpuPrimaryResultPaletteWords = (gpuMetadataMaxPaletteEntries*3 + 3) / 4
	gpuPrimaryResultPayload      = gpuPrimaryResultHeaderWords + gpuPrimaryResultPaletteWords
	gpuPrimaryResultWords        = gpuPrimaryResultPayload + (gpuPayloadMaxBits+31)/32

	gpuPrimaryResultMetaStatus            = 2
	gpuPrimaryResultPayloadStatus         = 3
	gpuPrimaryResultMetaModules           = 4
	gpuPrimaryResultNC                    = 5
	gpuPrimaryResultColors                = 6
	gpuPrimaryResultVersionX              = 7
	gpuPrimaryResultVersionY              = 8
	gpuPrimaryResultECLX                  = 9
	gpuPrimaryResultECLY                  = 10
	gpuPrimaryResultMask                  = 11
	gpuPrimaryResultSyndrome1             = 12
	gpuPrimaryResultSyndrome2             = 13
	gpuPrimaryResultPaletteLen            = 14
	gpuPrimaryResultNetBits               = 15
	gpuPrimaryResultMetaPart1Initial      = 16
	gpuPrimaryResultMetaPart1Corrections  = 17
	gpuPrimaryResultMetaPart2Initial      = 18
	gpuPrimaryResultMetaPart2Corrections  = 19
	gpuPrimaryResultPaletteSeparation     = 20
	gpuPrimaryResultPaletteDisagreement   = 21
	gpuPrimaryResultFixedAgreements       = 22
	gpuPrimaryResultFixedChecks           = 23
	gpuPrimaryResultEvidenceFlags         = 24
	gpuPrimaryResultPayloadInitial        = 25
	gpuPrimaryResultPayloadHardResidual   = 26
	gpuPrimaryResultPayloadCorrections    = 27
	gpuPrimaryResultPayloadSoftUsed       = 28
	gpuPrimaryResultPayloadSoftResidual   = 29
	gpuPrimaryResultPayloadSoftIterations = 30
)

const gpuPrimaryResultRetainedBytes = gpuPrimaryResultWords * 4

const (
	gpuPrimaryPayloadOK = iota
	gpuPrimaryPayloadFailed
	gpuPrimaryPayloadRejected
	gpuPrimaryPayloadInvalid
)

// gpuPrimaryResultBytes bounds a compact result by the sampled geometry. The
// metadata mode is not host-visible yet, so its prefix reserves the largest
// legal palette; corrected bits are packed and need only the sample's maximum
// possible eight bits per module.
func gpuPrimaryResultBytes(side image.Point) int {
	bits := side.X * side.Y * 8
	if bits < 0 || bits > gpuPayloadMaxBits {
		bits = gpuPayloadMaxBits
	}
	return (gpuPrimaryResultPayload + (bits+31)/32) * 4
}

func gpuPrimaryPayloadResult(out []byte, wantBits int) (dec []byte, ok bool, err error) {
	if !gpuPrimaryResultHeaderOK(out) {
		return nil, false, fmt.Errorf("jabcode: GPU primary result header is invalid")
	}
	status, haveStatus := gpuPrimaryResultWord(out, gpuPrimaryResultPayloadStatus)
	bits, haveBits := gpuPrimaryResultWord(out, gpuPrimaryResultNetBits)
	if !haveStatus || !haveBits || int(bits) != wantBits ||
		wantBits < 0 || wantBits > gpuPayloadMaxBits {
		return nil, false, fmt.Errorf("jabcode: GPU primary payload result shape is invalid")
	}
	last := gpuPrimaryResultPayload + (wantBits+31)/32
	if last*4 > len(out) {
		return nil, false, fmt.Errorf("jabcode: GPU primary payload result is truncated")
	}
	dec = make([]byte, wantBits)
	for at := range dec {
		packed, _ := gpuPrimaryResultWord(out, gpuPrimaryResultPayload+at/32)
		dec[at] = byte((packed >> (at % 32)) & 1)
	}
	return dec, status == gpuPrimaryPayloadOK, nil
}

func gpuPrimaryMetadataResult(out []byte) (gpuMetadataWalk, error) {
	var result gpuMetadataWalk
	if !gpuPrimaryResultHeaderOK(out) {
		return result, fmt.Errorf("jabcode: GPU primary result header is invalid")
	}
	fields := [...]struct {
		result int
		record int
	}{
		{gpuPrimaryResultMetaStatus, gpuMetadataRecordStatus},
		{gpuPrimaryResultMetaModules, gpuMetadataRecordModules},
		{gpuPrimaryResultNC, gpuMetadataRecordNC},
		{gpuPrimaryResultColors, gpuMetadataRecordColors},
		{gpuPrimaryResultVersionX, gpuMetadataRecordVersionX},
		{gpuPrimaryResultVersionY, gpuMetadataRecordVersionY},
		{gpuPrimaryResultECLX, gpuMetadataRecordECLX},
		{gpuPrimaryResultECLY, gpuMetadataRecordECLY},
		{gpuPrimaryResultMask, gpuMetadataRecordMask},
		{gpuPrimaryResultSyndrome1, gpuMetadataRecordSyndrome1},
		{gpuPrimaryResultSyndrome2, gpuMetadataRecordSyndrome2},
	}
	paletteLen, ok := gpuPrimaryResultWord(out, gpuPrimaryResultPaletteLen)
	if !ok || paletteLen > gpuMetadataMaxPaletteEntries*3 {
		return result, fmt.Errorf("jabcode: GPU primary metadata palette is invalid")
	}
	status, _ := gpuPrimaryResultWord(out, gpuPrimaryResultMetaStatus)
	colors, _ := gpuPrimaryResultWord(out, gpuPrimaryResultColors)
	nc, _ := gpuPrimaryResultWord(out, gpuPrimaryResultNC)
	if status != gpuMetadataStatusUnsupported {
		if nc >= gpuMetadataModeCount || colors != 1<<(nc+1) ||
			colors < 4 || colors > gpuPayloadMaxColors ||
			int(paletteLen) != int(colors)*spec.PaletteCopies(int(colors))*3 {
			return result, fmt.Errorf("jabcode: GPU primary metadata palette shape is invalid")
		}
	}
	record := make([]byte, (gpuMetadataRecordPalette+int(paletteLen))*4)
	for _, field := range fields {
		value, have := gpuPrimaryResultWord(out, field.result)
		if !have {
			return result, fmt.Errorf("jabcode: GPU primary metadata result is truncated")
		}
		binary.LittleEndian.PutUint32(record[field.record*4:], value)
	}
	for at := range int(paletteLen) {
		packed, have := gpuPrimaryResultWord(out, gpuPrimaryResultHeaderWords+at/4)
		if !have {
			return result, fmt.Errorf("jabcode: GPU primary metadata palette is truncated")
		}
		value := (packed >> ((at % 4) * 8)) & 0xff
		binary.LittleEndian.PutUint32(record[(gpuMetadataRecordPalette+at)*4:], value)
	}
	return gpuMetadataResult(record)
}

func gpuPrimaryEvidenceResult(out []byte, metadata gpuMetadataWalk) (core.PrimaryEvidence, error) {
	word := func(index int) (uint32, error) {
		value, ok := gpuPrimaryResultWord(out, index)
		if !ok {
			return 0, fmt.Errorf("jabcode: GPU primary evidence is truncated")
		}
		return value, nil
	}
	indices := [...]int{
		gpuPrimaryResultMetaPart1Initial,
		gpuPrimaryResultSyndrome1,
		gpuPrimaryResultMetaPart1Corrections,
		gpuPrimaryResultMetaPart2Initial,
		gpuPrimaryResultSyndrome2,
		gpuPrimaryResultMetaPart2Corrections,
		gpuPrimaryResultPaletteSeparation,
		gpuPrimaryResultPaletteDisagreement,
		gpuPrimaryResultFixedAgreements,
		gpuPrimaryResultFixedChecks,
		gpuPrimaryResultEvidenceFlags,
		gpuPrimaryResultPayloadInitial,
		gpuPrimaryResultPayloadHardResidual,
		gpuPrimaryResultPayloadCorrections,
		gpuPrimaryResultPayloadSoftUsed,
		gpuPrimaryResultPayloadSoftResidual,
		gpuPrimaryResultPayloadSoftIterations,
	}
	var values [len(indices)]uint32
	for at, index := range indices {
		value, err := word(index)
		if err != nil {
			return core.PrimaryEvidence{}, err
		}
		values[at] = value
	}
	return core.PrimaryEvidence{
		Available:                 true,
		MetadataExplicit:          !metadata.Defaulted,
		FixedPatternUsed:          values[10]&1 != 0,
		SoftFallbackUsed:          values[14] != 0,
		MetadataPartIInitial:      values[0],
		MetadataPartIResidual:     values[1],
		MetadataPartICorrections:  values[2],
		MetadataPartIIInitial:     values[3],
		MetadataPartIIResidual:    values[4],
		MetadataPartIICorrections: values[5],
		PaletteSeparation:         math.Float32frombits(values[6]),
		PaletteDisagreement:       math.Float32frombits(values[7]),
		FixedAgreements:           values[8],
		FixedChecks:               values[9],
		PayloadInitial:            values[11],
		PayloadHardResidual:       values[12],
		PayloadCorrections:        values[13],
		PayloadSoftResidual:       values[15],
		PayloadSoftIterations:     values[16],
	}, nil
}

func gpuPrimaryResultHeaderOK(out []byte) bool {
	magic, haveMagic := gpuPrimaryResultWord(out, 0)
	version, haveVersion := gpuPrimaryResultWord(out, 1)
	return haveMagic && haveVersion && magic == gpuPrimaryResultMagic &&
		version == gpuPrimaryResultVersion
}

func gpuPrimaryResultWord(out []byte, index int) (uint32, bool) {
	if index < 0 || (index+1)*4 > len(out) {
		return 0, false
	}
	return binary.LittleEndian.Uint32(out[index*4:]), true
}

func (resident *gpuResidentBinarizer) initializePrimaryResult() error {
	var err error
	resident.primaryResult, err = resident.device.NewBuffer(gpuPrimaryResultWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU primary result: %w", err)
	}
	resident.primaryResultKernel, err = resident.kernels.primaryResult()
	if err != nil {
		return err
	}
	resident.primaryResultBindings, err = resident.primaryResultKernel.NewBindings(
		vulki.BindBuffer(0, resident.metadataRecord),
		vulki.BindBuffer(1, resident.payloadParams),
		vulki.BindBuffer(2, resident.ldpcParams),
		vulki.BindBuffer(3, resident.ldpcNet),
		vulki.BindBuffer(4, resident.primaryResult),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU primary result: %w", err)
	}
	return nil
}

// gpuPrimaryDecoder holds the fused decode to the route lease that produced the
// sampled grid. A stale detector must decline instead of consuming the next
// route's resident buffers.
type gpuPrimaryDecoder struct {
	resident *gpuResidentBinarizer
	epoch    *atomic.Uint64
	lease    uint64
}

func (decoder gpuPrimaryDecoder) DecodePrimary(
	matrix *core.Bitmap,
	symbol *core.DecodedSymbol,
) (core.PrimaryDeviceResult, error) {
	if decoder.epoch.Load() != decoder.lease {
		return core.PrimaryDeviceResult{}, fmt.Errorf(
			"jabcode: GPU route context was released before primary decode")
	}
	return decoder.resident.DecodePrimary(matrix, symbol)
}

// DecodePrimary records metadata interpretation through payload correction and
// result packing as one submission. The sampled module grid is its only input;
// the packed primary result is its only transfer back.
func (resident *gpuResidentBinarizer) DecodePrimary(
	matrix *core.Bitmap,
	symbol *core.DecodedSymbol,
) (core.PrimaryDeviceResult, error) {
	var result core.PrimaryDeviceResult
	if resident == nil || resident.closed || resident.primaryResultBindings == nil ||
		matrix == nil || symbol == nil {
		return result, fmt.Errorf("jabcode: resident GPU primary decoder is unavailable")
	}
	side := image.Pt(matrix.Width, matrix.Height)
	if !spec.ValidSideSize(side.X) || !spec.ValidSideSize(side.Y) ||
		side.X > gpuSampleMaxSide || side.Y > gpuSampleMaxSide {
		return result, fmt.Errorf("jabcode: GPU primary side %dx%d is out of range", side.X, side.Y)
	}
	resident.mu.Lock()
	defer resident.mu.Unlock()
	if matrix != resident.sampledGrid {
		return result, fmt.Errorf("jabcode: GPU primary decode was asked about another sample")
	}
	resident.payloadControlReady = false

	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return result, fmt.Errorf("jabcode: create GPU primary recorder: %w", err)
	}
	defer recorder.Abort()
	if err := resident.recordMetadataWalk(recorder, symbol.WireVariant); err != nil {
		return result, err
	}
	if err := resident.recordPayloadCorrection(recorder, nil); err != nil {
		return result, err
	}

	out := make([]byte, gpuPrimaryResultBytes(side))
	phaseprobe.Count("download.primary_result", len(out))
	if err := recorder.Download(resident.primaryResult, 0, out); err != nil {
		return result, fmt.Errorf("jabcode: record GPU primary result download: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		resident.permutationCacheDirty = true
		resident.ldpcMatrixCacheDirty = true
		return result, fmt.Errorf("jabcode: run fused GPU primary decode: %w", err)
	}
	resident.permutationCacheDirty = false
	resident.ldpcMatrixCacheDirty = false
	resident.metadataRowsReady = true

	walk, err := gpuPrimaryMetadataResult(out)
	if err != nil {
		return result, err
	}
	if walk.Unsupported {
		return result, fmt.Errorf("jabcode: GPU primary decode does not cover %d colours", walk.Colors)
	}
	bits, haveBits := gpuPrimaryResultWord(out, gpuPrimaryResultNetBits)
	if !haveBits || bits > gpuPayloadMaxBits {
		return result, fmt.Errorf("jabcode: GPU primary result payload length is invalid")
	}
	payload, payloadOK, err := gpuPrimaryPayloadResult(out, int(bits))
	if err != nil {
		return result, err
	}
	result.Metadata = gpuPrimaryMetadata(walk)
	result.Payload = payload
	result.PayloadOK = payloadOK
	result.Evidence, err = gpuPrimaryEvidenceResult(out, walk)
	if err != nil {
		return core.PrimaryDeviceResult{}, err
	}
	return result, nil
}
