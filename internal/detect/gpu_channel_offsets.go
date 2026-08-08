//go:build !js

package detect

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"image"
	"math"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/phaseprobe"
)

//go:embed shaders/channel_offsets.wgsl
var channelOffsetsWGSL string

// Parameter word indices, matching channel_offsets.wgsl.
const (
	gpuChannelOffsetParamWidth      = 0
	gpuChannelOffsetParamHeight     = 1
	gpuChannelOffsetParamSideX      = 2
	gpuChannelOffsetParamSideY      = 3
	gpuChannelOffsetParamCandidates = 4
	gpuChannelOffsetParamModW       = 5
	gpuChannelOffsetParamModH       = 6
	gpuChannelOffsetParamMinRange   = 7
	gpuChannelOffsetParamTransform  = 8
	gpuChannelOffsetParamGrid       = 17
	gpuChannelOffsetParamWords      = gpuChannelOffsetParamGrid + 16

	// Three channels and two interleaved module halves per candidate offset.
	gpuChannelOffsetSlotsPerCandidate = 6
)

// gpuChannelOffsetSteps is the candidate grid's length per axis, as a constant
// so the route context's byte accounting can name this stage's buffers. The
// host grid stays its definition; this stops compiling if the two diverge.
const gpuChannelOffsetSteps = 9

var _ [len(channelOffsetGrid) - gpuChannelOffsetSteps]struct{}

// gpuChannelOffsetSlots is the score table's length: one score per candidate
// offset per channel per parity.
const gpuChannelOffsetSlots = gpuChannelOffsetSteps * gpuChannelOffsetSteps *
	gpuChannelOffsetSlotsPerCandidate

// initializeChannelOffsets allocates the score table and compiles the search,
// with the rest of the resident stages so the compile lands in warm-up. Only a
// print-level detection ever dispatches it.
func (resident *gpuResidentBinarizer) initializeChannelOffsets() error {
	var err error
	resident.offsetScores, err = resident.device.NewBuffer(uint64(gpuChannelOffsetSlots) * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU channel-offset scores: %w", err)
	}
	resident.offsetParams, err = resident.device.NewBuffer(gpuChannelOffsetParamWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate resident GPU channel-offset parameters: %w", err)
	}
	resident.offsetKernel, err = resident.kernels.channelOffsets()
	if err != nil {
		return err
	}
	resident.offsetBindings, err = resident.offsetKernel.NewBindings(
		vulki.BindBuffer(0, resident.balanced),
		vulki.BindBuffer(1, resident.offsetScores),
		vulki.BindBuffer(2, resident.offsetParams),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU channel-offset search: %w", err)
	}
	return nil
}

// SearchChannelOffsets scores every candidate plane displacement where the
// balanced pixels already are, and returns the per-channel offset.
//
// This is the stage that used to download the whole frame: the host search
// reads every module's footprint on all three channels for a few hundred
// candidates, so it needed the image, and on a print capture that is the
// largest single transfer a read makes. The candidates are independent, so the
// device scores them in one dispatch and only the score table comes back.
func (resident *gpuResidentBinarizer) SearchChannelOffsets(
	width, height int,
	pt core.Perspective,
	side image.Point,
) ([3]core.PointF, error) {
	var delta [3]core.PointF
	if resident == nil || resident.closed || resident.offsetBindings == nil {
		return delta, fmt.Errorf("jabcode: resident GPU binarizer is closed")
	}
	if side.X <= 0 || side.Y <= 0 {
		return delta, fmt.Errorf("jabcode: GPU channel-offset side %dx%d is out of range", side.X, side.Y)
	}
	if width <= 0 || height <= 0 ||
		width > resident.binarizer.maxWidth || height > resident.binarizer.maxHeight {
		return delta, fmt.Errorf("jabcode: GPU channel-offset dimensions are unavailable")
	}
	modW, modH := moduleExtent(pt, side)
	// The two gates the host applies before scoring anything: below the
	// footprint regime the grid steps quantize to a pixel or two, and a
	// population this small cannot carry deciles.
	if min(modW, modH) < legacySampleBelowPx {
		return delta, nil
	}
	if channelOffsetSpots(side) < 16 {
		return delta, nil
	}

	params := gpuChannelOffsetParams(width, height, pt, side, modW, modH)
	recorder, err := resident.device.NewRecorder()
	if err != nil {
		return delta, fmt.Errorf("jabcode: create GPU channel-offset recorder: %w", err)
	}
	defer recorder.Abort()
	if err := recorder.Update(resident.offsetParams, 0, params[:]); err != nil {
		return delta, fmt.Errorf("jabcode: update GPU channel-offset parameters: %w", err)
	}
	if err := recorder.Barrier(resident.offsetParams); err != nil {
		return delta, fmt.Errorf("jabcode: synchronize GPU channel-offset parameters: %w", err)
	}
	if err := recorder.Dispatch(
		resident.offsetKernel,
		resident.offsetBindings,
		vulki.Workgroups{X: uint32(gpuChannelOffsetSlots), Y: 1, Z: 1},
	); err != nil {
		return delta, fmt.Errorf("jabcode: dispatch GPU channel-offset search: %w", err)
	}
	if err := recorder.Barrier(resident.offsetScores); err != nil {
		return delta, fmt.Errorf("jabcode: synchronize GPU channel-offset scores: %w", err)
	}
	scores := make([]byte, gpuChannelOffsetSlots*4)
	phaseprobe.Count("download.channel_offsets", len(scores))
	if err := recorder.Download(resident.offsetScores, 0, scores); err != nil {
		return delta, fmt.Errorf("jabcode: download GPU channel-offset scores: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return delta, fmt.Errorf("jabcode: run GPU channel-offset search: %w", err)
	}

	table := make([]float64, gpuChannelOffsetSlots)
	for slot := range table {
		table[slot] = float64(math.Float32frombits(
			binary.LittleEndian.Uint32(scores[slot*4:]),
		))
	}
	resident.offsetTable = table
	return pickChannelOffsets(func(c, candidate, parity int) float64 {
		return table[channelOffsetSlot(c, candidate, parity)]
	}, modW, modH), nil
}

// channelOffsetSpots counts the module centres the search scores, which is the
// host's stride-two subset of the grid.
func channelOffsetSpots(side image.Point) int {
	return ((side.X + 1) / 2) * ((side.Y + 1) / 2)
}

func gpuChannelOffsetParams(
	width, height int,
	pt core.Perspective,
	side image.Point,
	modW, modH float64,
) [gpuChannelOffsetParamWords * 4]byte {
	var params [gpuChannelOffsetParamWords * 4]byte
	put := func(index int, value uint32) {
		binary.LittleEndian.PutUint32(params[index*4:], value)
	}
	putFloat := func(index int, value float64) {
		put(index, math.Float32bits(float32(value)))
	}
	put(gpuChannelOffsetParamWidth, uint32(width))
	put(gpuChannelOffsetParamHeight, uint32(height))
	put(gpuChannelOffsetParamSideX, uint32(side.X))
	put(gpuChannelOffsetParamSideY, uint32(side.Y))
	put(gpuChannelOffsetParamCandidates, uint32(len(channelOffsetGrid)*len(channelOffsetGrid)))
	putFloat(gpuChannelOffsetParamModW, modW)
	putFloat(gpuChannelOffsetParamModH, modH)
	putFloat(gpuChannelOffsetParamMinRange, minBimodalRange)
	for i, coefficient := range pt.Coefficients() {
		putFloat(gpuChannelOffsetParamTransform+i, coefficient)
	}
	// The grid is uploaded rather than derived so the host stays its single
	// definition; the kernel indexes it by candidate row and column.
	for i, fraction := range channelOffsetGrid {
		putFloat(gpuChannelOffsetParamGrid+i, fraction)
	}
	return params
}
