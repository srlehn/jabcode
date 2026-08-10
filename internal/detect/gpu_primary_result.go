//go:build !js

package detect

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"image"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/spec"
)

//go:embed shaders/primary_result.wgsl
var primaryResultWGSL string

const (
	gpuPrimaryResultMagic   = 0x4a414252
	gpuPrimaryResultVersion = 1

	gpuPrimaryResultHeaderWords  = 16
	gpuPrimaryResultPaletteWords = (gpuMetadataMaxPaletteEntries*3 + 3) / 4
	gpuPrimaryResultPayload      = gpuPrimaryResultHeaderWords + gpuPrimaryResultPaletteWords
	gpuPrimaryResultWords        = gpuPrimaryResultPayload + (gpuPayloadMaxBits+31)/32

	gpuPrimaryResultMetaStatus    = 2
	gpuPrimaryResultPayloadStatus = 3
	gpuPrimaryResultMetaModules   = 4
	gpuPrimaryResultNC            = 5
	gpuPrimaryResultColors        = 6
	gpuPrimaryResultVersionX      = 7
	gpuPrimaryResultVersionY      = 8
	gpuPrimaryResultECLX          = 9
	gpuPrimaryResultECLY          = 10
	gpuPrimaryResultMask          = 11
	gpuPrimaryResultSyndrome1     = 12
	gpuPrimaryResultSyndrome2     = 13
	gpuPrimaryResultPaletteLen    = 14
	gpuPrimaryResultNetBits       = 15
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
