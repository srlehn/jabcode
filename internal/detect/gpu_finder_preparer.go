//go:build !js

package detect

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"math"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/phaseprobe"
)

//go:embed shaders/finder_average.wgsl
var finderAverageWGSL string

//go:embed shaders/finder_average_reduce.wgsl
var finderAverageReduceWGSL string

//go:embed shaders/finder_retry_control.wgsl
var finderRetryControlWGSL string

//go:embed shaders/pitch_samples.wgsl
var pitchSamplesWGSL string

//go:embed shaders/pitch_line_sums.wgsl
var pitchLineSumsWGSL string

//go:embed shaders/pitch_center.wgsl
var pitchCenterWGSL string

//go:embed shaders/pitch_acf.wgsl
var pitchACFWGSL string

//go:embed shaders/pitch_schedule.wgsl
var pitchScheduleWGSL string

//go:embed shaders/descreen_horizontal.wgsl
var descreenHorizontalWGSL string

//go:embed shaders/descreen_vertical.wgsl
var descreenVerticalWGSL string

const (
	gpuFinderAverageParamsSize  = 18 * 4
	gpuFinderRetryControlWords  = 56
	gpuFinderAveragePartialSize = 4 * 64 * 4 * 4
	// Three channel averages, padded to a four-word block.
	gpuFinderAverageResultSize = 4 * 4
	gpuDescreenParamsSize      = 4 * 4
	// gpuPitchLagLineBytes holds one f32 sum per sampled line per axis.
	gpuPitchLagLineBytes = 2 * pitchSampleLines * 4

	gpuPitchScheduleControlWords = 9
	gpuPitchRetryRecordWords     = 4
	gpuPitchScheduleRecords      = 4
	gpuPitchScheduleIndirects    = 5
	gpuPitchScheduleWords        = gpuPitchScheduleRecords*gpuPitchRetryRecordWords +
		gpuPitchScheduleIndirects*3
	gpuPitchScheduleStages = 6
)

const (
	gpuFinderRetryStageCaptureAverageRow = iota
	gpuFinderRetryStageCaptureSurvivors
	gpuFinderRetryStageSelectAverage
	gpuFinderRetryStageSelectPitch
	gpuFinderRetryStageCaptureAverageDecision
	gpuFinderRetryStageCapturePassResult
)

const (
	gpuFinderRetryControlStage        = 18 * 4
	gpuFinderRetryControlMaxSurvivors = 19 * 4
	gpuFinderRetryIndirectAverage     = 20 * 4
	gpuFinderRetryIndirectReduce      = 23 * 4
	gpuFinderRetryIndirectCanvas      = 26 * 4
	gpuFinderRetryIndirectBlocks      = 29 * 4
	gpuFinderRetryIndirectPack        = 32 * 4
	gpuFinderRetryIndirectScan        = 35 * 4
	gpuFinderRetryIndirectChain       = 38 * 4
	gpuFinderRetryIndirectPitchSample = 41 * 4
	gpuFinderRetryIndirectPitchOne    = 44 * 4
	gpuFinderRetryIndirectPitchCenter = 47 * 4
	gpuFinderRetryIndirectPitchACF    = 50 * 4
	gpuFinderRetryIndirectPitchSelect = 53 * 4
)

const (
	gpuPitchRetryIndirectCanvas = (gpuPitchScheduleRecords * gpuPitchRetryRecordWords) * 4
	gpuPitchRetryIndirectBlocks = gpuPitchRetryIndirectCanvas + 3*4
	gpuPitchRetryIndirectPack   = gpuPitchRetryIndirectBlocks + 3*4
	gpuPitchRetryIndirectScan   = gpuPitchRetryIndirectPack + 3*4
	gpuPitchRetryIndirectChain  = gpuPitchRetryIndirectScan + 3*4
)

var gpuPitchRetryDispatchOffsets = gpuBinarizerIndirectOffsets{
	canvas: gpuPitchRetryIndirectCanvas,
	blocks: gpuPitchRetryIndirectBlocks,
	pack:   gpuPitchRetryIndirectPack,
	scan:   gpuPitchRetryIndirectScan,
	chain:  gpuPitchRetryIndirectChain,
}

var gpuFinderRetryDispatchOffsets = gpuBinarizerIndirectOffsets{
	canvas: gpuFinderRetryIndirectCanvas,
	blocks: gpuFinderRetryIndirectBlocks,
	pack:   gpuFinderRetryIndirectPack,
	scan:   gpuFinderRetryIndirectScan,
	chain:  gpuFinderRetryIndirectChain,
}

const (
	gpuPitchControlRowValley = iota
	gpuPitchControlColumnValley
	gpuPitchControlRowPeak
	gpuPitchControlColumnPeak
	gpuPitchControlRowLag
	gpuPitchControlColumnLag
	gpuPitchControlMedianBucket
	gpuPitchControlSelector
	gpuPitchControlStage
)

const (
	gpuPitchStageValley = iota
	gpuPitchStagePeak
	gpuPitchStageLag
	gpuPitchStageSchedule
	gpuPitchStagePrint
	gpuPitchStageSelect
)

type gpuFinderPassPreparer struct {
	device   *vulki.Device
	kernels  *gpuDecodeKernels
	resident *gpuResidentBinarizer
	width    int
	height   int
	trace    bool

	averageParams   *vulki.Buffer
	averagePartials *vulki.Buffer
	averageKernel   *vulki.Kernel
	averageBindings *vulki.BindingSet

	// The partials are folded on the device into the next pass's parameters.
	// averageResult exists only so diagnostics and parity tests can request the
	// same three values without ever exposing the 4 KB of per-lane sums.
	averageResult         *vulki.Buffer
	averageReduceKernel   *vulki.Kernel
	averageReduceBindings *vulki.BindingSet
	averageBytes          [gpuFinderAverageResultSize]byte
	retryControlKernel    *vulki.Kernel
	retryControlBindings  *vulki.BindingSet

	pitchSamples  *vulki.Buffer
	pitchKernel   *vulki.Kernel
	pitchBindings *vulki.BindingSet
	// pitchBytes is the host destination of the fallback sample download,
	// allocated on the first estimate that runs without the lag kernels.
	pitchBytes []byte

	pitchLagSums           *vulki.Buffer
	pitchLagCentered       *vulki.Buffer
	pitchLagACF            *vulki.Buffer
	pitchLagSumsKernel     *vulki.Kernel
	pitchLagCenterKernel   *vulki.Kernel
	pitchLagACFKernel      *vulki.Kernel
	pitchLagSumsBindings   *vulki.BindingSet
	pitchLagCenterBindings *vulki.BindingSet
	pitchLagACFBindings    *vulki.BindingSet
	pitchLagACFBytes       []byte

	pitchScheduleControl *vulki.Buffer
	pitchSchedule        *vulki.Buffer
	pitchScheduleKernel  *vulki.Kernel
	pitchScheduleBinding *vulki.BindingSet
	pitchScheduleReady   bool

	descreenParams     *vulki.Buffer
	descreenLinear     *vulki.Buffer
	descreenFiltered   *vulki.Buffer
	descreenHorizontal *vulki.Kernel
	descreenVertical   *vulki.Kernel
	descreenHBindings  *vulki.BindingSet
	descreenVBindings  *vulki.BindingSet
}

func (preparer *gpuFinderPassPreparer) setInput(width, height int, trace bool) {
	preparer.width = width
	preparer.height = height
	preparer.trace = trace
	preparer.pitchScheduleReady = false
}

func newGPUFinderPassPreparer(
	device *vulki.Device,
	kernels *gpuDecodeKernels,
	resident *gpuResidentBinarizer,
) (*gpuFinderPassPreparer, error) {
	if device == nil || device.Closed() || resident == nil {
		return nil, fmt.Errorf("jabcode: GPU finder preparer needs an open resident device")
	}
	preparer := &gpuFinderPassPreparer{device: device, kernels: kernels, resident: resident}
	var err error
	preparer.averageParams, err = device.NewBuffer(gpuFinderRetryControlWords * 4)
	if err != nil {
		return nil, fmt.Errorf("jabcode: allocate GPU finder-average parameters: %w", err)
	}
	preparer.averagePartials, err = device.NewBuffer(gpuFinderAveragePartialSize)
	if err != nil {
		_ = preparer.Close()
		return nil, fmt.Errorf("jabcode: allocate GPU finder-average partials: %w", err)
	}
	preparer.averageKernel, err = kernels.finderAverage()
	if err != nil {
		_ = preparer.Close()
		return nil, err
	}
	preparer.averageBindings, err = preparer.averageKernel.NewBindings(
		vulki.BindBuffer(0, resident.balanced),
		vulki.BindBuffer(1, preparer.averagePartials),
		vulki.BindBuffer(2, preparer.averageParams),
	)
	if err != nil {
		_ = preparer.Close()
		return nil, fmt.Errorf("jabcode: bind GPU finder-average kernel: %w", err)
	}
	preparer.averageResult, err = device.NewBuffer(gpuFinderAverageResultSize)
	if err != nil {
		_ = preparer.Close()
		return nil, fmt.Errorf("jabcode: allocate GPU finder-average result: %w", err)
	}
	preparer.averageReduceKernel, err = kernels.finderAverageReduce()
	if err != nil {
		_ = preparer.Close()
		return nil, err
	}
	preparer.averageReduceBindings, err = preparer.averageReduceKernel.NewBindings(
		vulki.BindBuffer(0, preparer.averagePartials),
		vulki.BindBuffer(1, preparer.averageResult),
		vulki.BindBuffer(2, resident.binarizer.params),
	)
	if err != nil {
		_ = preparer.Close()
		return nil, fmt.Errorf("jabcode: bind GPU finder-average reduction: %w", err)
	}
	maxSamples := gpuPitchSampleCount(
		resident.binarizer.maxWidth,
		resident.binarizer.maxHeight,
	)
	preparer.pitchSamples, err = device.NewBuffer(uint64(maxSamples) * 4)
	if err != nil {
		_ = preparer.Close()
		return nil, fmt.Errorf("jabcode: allocate GPU pitch samples: %w", err)
	}
	preparer.pitchKernel, err = kernels.pitchSamples()
	if err != nil {
		_ = preparer.Close()
		return nil, err
	}
	preparer.pitchBindings, err = preparer.pitchKernel.NewBindings(
		vulki.BindBuffer(0, resident.balanced),
		vulki.BindBuffer(1, preparer.pitchSamples),
		vulki.BindBuffer(2, resident.binarizer.params),
	)
	if err != nil {
		_ = preparer.Close()
		return nil, fmt.Errorf("jabcode: bind GPU pitch-sample kernel: %w", err)
	}
	return preparer, nil
}

// ensureDescreen allocates the descreen chain on first use. The linear
// intermediate costs 16 bytes per pixel, by far the largest buffer a route
// context can hold, and only the print-level retry passes ever need it - a
// context that never descreens never pays for it.
func (preparer *gpuFinderPassPreparer) ensureDescreen() error {
	if preparer.descreenVBindings != nil {
		return nil
	}
	resident := preparer.resident
	area := uint64(resident.binarizer.maxWidth) * uint64(resident.binarizer.maxHeight)
	var err error
	preparer.descreenParams, err = preparer.device.NewBuffer(gpuDescreenParamsSize)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU descreen parameters: %w", err)
	}
	preparer.descreenLinear, err = preparer.device.NewBuffer(area * 16)
	if err != nil {
		_ = preparer.closeDescreen()
		return fmt.Errorf("jabcode: allocate GPU descreen linear image: %w", err)
	}
	preparer.descreenFiltered, err = preparer.device.NewBuffer(area * 4)
	if err != nil {
		_ = preparer.closeDescreen()
		return fmt.Errorf("jabcode: allocate GPU descreen output: %w", err)
	}
	preparer.descreenHorizontal, err = preparer.kernels.descreenHorizontal()
	if err != nil {
		_ = preparer.closeDescreen()
		return err
	}
	preparer.descreenVertical, err = preparer.kernels.descreenVertical()
	if err != nil {
		_ = preparer.closeDescreen()
		return err
	}
	preparer.descreenHBindings, err = preparer.descreenHorizontal.NewBindings(
		vulki.BindBuffer(0, resident.balanced),
		vulki.BindBuffer(1, preparer.descreenLinear),
		vulki.BindBuffer(2, preparer.descreenParams),
	)
	if err != nil {
		_ = preparer.closeDescreen()
		return fmt.Errorf("jabcode: bind GPU horizontal descreen kernel: %w", err)
	}
	preparer.descreenVBindings, err = preparer.descreenVertical.NewBindings(
		vulki.BindBuffer(0, preparer.descreenLinear),
		vulki.BindBuffer(1, preparer.descreenFiltered),
		vulki.BindBuffer(2, resident.balanced),
		vulki.BindBuffer(3, preparer.descreenParams),
	)
	if err != nil {
		_ = preparer.closeDescreen()
		return fmt.Errorf("jabcode: bind GPU vertical descreen kernel: %w", err)
	}
	return nil
}

// ensurePitchLag allocates the resident autocorrelation chain on first use.
// Only the descreen retry tier estimates pitch, so contexts that never
// reach it hold none of these buffers; the centered samples are the largest
// at four bytes per sampled pixel.
func (preparer *gpuFinderPassPreparer) ensurePitchLag() error {
	if preparer.pitchLagACFBindings != nil {
		return nil
	}
	resident := preparer.resident
	maxWidth := resident.binarizer.maxWidth
	maxHeight := resident.binarizer.maxHeight
	maxSamples := gpuPitchSampleCount(maxWidth, maxHeight)
	maxLags := max(2, min(maxWidth, maxHeight)/8) + 1
	var err error
	preparer.pitchLagSums, err = preparer.device.NewBuffer(gpuPitchLagLineBytes)
	if err != nil {
		_ = preparer.closePitchLag()
		return fmt.Errorf("jabcode: allocate GPU pitch line sums: %w", err)
	}
	preparer.pitchLagCentered, err = preparer.device.NewBuffer(uint64(maxSamples) * 4)
	if err != nil {
		_ = preparer.closePitchLag()
		return fmt.Errorf("jabcode: allocate GPU centered pitch samples: %w", err)
	}
	preparer.pitchLagACF, err = preparer.device.NewBuffer(uint64(2*maxLags) * 4)
	if err != nil {
		_ = preparer.closePitchLag()
		return fmt.Errorf("jabcode: allocate GPU pitch autocorrelation: %w", err)
	}
	preparer.pitchLagSumsKernel, err = preparer.kernels.pitchLineSums()
	if err != nil {
		_ = preparer.closePitchLag()
		return err
	}
	preparer.pitchLagCenterKernel, err = preparer.kernels.pitchCenter()
	if err != nil {
		_ = preparer.closePitchLag()
		return err
	}
	preparer.pitchLagACFKernel, err = preparer.kernels.pitchACF()
	if err != nil {
		_ = preparer.closePitchLag()
		return err
	}
	preparer.pitchLagSumsBindings, err = preparer.pitchLagSumsKernel.NewBindings(
		vulki.BindBuffer(0, preparer.pitchSamples),
		vulki.BindBuffer(1, preparer.pitchLagSums),
		vulki.BindBuffer(2, resident.binarizer.params),
	)
	if err != nil {
		_ = preparer.closePitchLag()
		return fmt.Errorf("jabcode: bind GPU pitch-sum kernel: %w", err)
	}
	preparer.pitchLagCenterBindings, err = preparer.pitchLagCenterKernel.NewBindings(
		vulki.BindBuffer(0, preparer.pitchSamples),
		vulki.BindBuffer(1, preparer.pitchLagSums),
		vulki.BindBuffer(2, preparer.pitchLagCentered),
		vulki.BindBuffer(3, resident.binarizer.params),
	)
	if err != nil {
		_ = preparer.closePitchLag()
		return fmt.Errorf("jabcode: bind GPU pitch-center kernel: %w", err)
	}
	preparer.pitchLagACFBindings, err = preparer.pitchLagACFKernel.NewBindings(
		vulki.BindBuffer(0, preparer.pitchLagCentered),
		vulki.BindBuffer(1, preparer.pitchLagACF),
		vulki.BindBuffer(2, resident.binarizer.params),
	)
	if err != nil {
		_ = preparer.closePitchLag()
		return fmt.Errorf("jabcode: bind GPU pitch-lag kernel: %w", err)
	}
	preparer.pitchLagACFBytes = make([]byte, 2*maxLags*4)
	return nil
}

// ensurePitchSchedule attaches the reduction to the buffers that already own
// its evidence and to the retry controls it produces. It is lazy for the same
// reason as the pitch and descreen chains: a route that settles on its first
// pass should retain none of this state.
func (preparer *gpuFinderPassPreparer) ensurePitchSchedule() error {
	if preparer.pitchScheduleBinding != nil {
		return nil
	}
	if err := preparer.ensurePitchLag(); err != nil {
		return err
	}
	if err := preparer.ensureDescreen(); err != nil {
		return err
	}
	var err error
	preparer.pitchScheduleControl, err = preparer.device.NewBuffer(gpuPitchScheduleControlWords * 4)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU pitch schedule control: %w", err)
	}
	preparer.pitchSchedule, err = preparer.device.NewBuffer(gpuPitchScheduleWords * 4)
	if err != nil {
		_ = preparer.closePitchSchedule()
		return fmt.Errorf("jabcode: allocate GPU pitch schedule: %w", err)
	}
	preparer.pitchScheduleKernel, err = preparer.kernels.pitchSchedule()
	if err != nil {
		_ = preparer.closePitchSchedule()
		return err
	}
	preparer.pitchScheduleBinding, err = preparer.pitchScheduleKernel.NewBindings(
		vulki.BindBuffer(0, preparer.pitchLagACF),
		vulki.BindBuffer(1, preparer.resident.binarizer.seedHistogram),
		vulki.BindBuffer(2, preparer.resident.binarizer.params),
		vulki.BindBuffer(3, preparer.pitchScheduleControl),
		vulki.BindBuffer(4, preparer.pitchSchedule),
		vulki.BindBuffer(5, preparer.descreenParams),
		vulki.BindBuffer(6, preparer.resident.binarizer.scanParams),
		vulki.BindBuffer(7, preparer.resident.binarizer.chainParams),
		vulki.BindBuffer(8, preparer.averageParams),
		vulki.BindBuffer(9, preparer.resident.finderDecisionIndirect),
	)
	if err != nil {
		_ = preparer.closePitchSchedule()
		return fmt.Errorf("jabcode: bind GPU pitch schedule: %w", err)
	}
	return nil
}

func (preparer *gpuFinderPassPreparer) closePitchSchedule() error {
	var closeErrors []error
	if preparer.pitchScheduleBinding != nil {
		closeErrors = append(closeErrors, preparer.pitchScheduleBinding.Close())
	}
	preparer.pitchScheduleBinding = nil
	preparer.pitchScheduleKernel = nil
	for _, buffer := range []*vulki.Buffer{
		preparer.pitchSchedule,
		preparer.pitchScheduleControl,
	} {
		if buffer != nil {
			closeErrors = append(closeErrors, buffer.Close())
		}
	}
	preparer.pitchSchedule = nil
	preparer.pitchScheduleControl = nil
	preparer.pitchScheduleReady = false
	return errors.Join(closeErrors...)
}

// closePitchLag releases the lazily-created autocorrelation chain. The
// kernels stay in the shared per-device set; only references are dropped.
func (preparer *gpuFinderPassPreparer) closePitchLag() error {
	var closeErrors []error
	for _, bindings := range []*vulki.BindingSet{
		preparer.pitchLagACFBindings,
		preparer.pitchLagCenterBindings,
		preparer.pitchLagSumsBindings,
	} {
		if bindings != nil {
			closeErrors = append(closeErrors, bindings.Close())
		}
	}
	preparer.pitchLagACFBindings = nil
	preparer.pitchLagCenterBindings = nil
	preparer.pitchLagSumsBindings = nil
	preparer.pitchLagACFKernel = nil
	preparer.pitchLagCenterKernel = nil
	preparer.pitchLagSumsKernel = nil
	for _, buffer := range []*vulki.Buffer{
		preparer.pitchLagACF,
		preparer.pitchLagCentered,
		preparer.pitchLagSums,
	} {
		if buffer != nil {
			closeErrors = append(closeErrors, buffer.Close())
		}
	}
	preparer.pitchLagACF = nil
	preparer.pitchLagCentered = nil
	preparer.pitchLagSums = nil
	return errors.Join(closeErrors...)
}

func (preparer *gpuFinderPassPreparer) averagePixelValue(
	fps []FinderPattern,
) ([3]float32, error) {
	var empty [3]float32
	if err := preparer.validateAverage(); err != nil {
		return empty, err
	}
	recorder, err := preparer.device.NewRecorder()
	if err != nil {
		return empty, fmt.Errorf("jabcode: create GPU finder-average recorder: %w", err)
	}
	defer recorder.Abort()
	if err := preparer.recordAverage(recorder, fps); err != nil {
		return empty, err
	}
	phaseprobe.Count("download.finder_average", len(preparer.averageBytes))
	if err := recorder.Download(preparer.averageResult, 0, preparer.averageBytes[:]); err != nil {
		return empty, fmt.Errorf("jabcode: download GPU finder average: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return empty, fmt.Errorf("jabcode: run GPU finder-average kernel: %w", err)
	}
	var average [3]float32
	for channel := range average {
		average[channel] = math.Float32frombits(
			binary.LittleEndian.Uint32(preparer.averageBytes[channel*4:]))
	}
	return average, nil
}

func (preparer *gpuFinderPassPreparer) validateAverage() error {
	if preparer == nil || preparer.resident == nil || preparer.averageBindings == nil {
		return fmt.Errorf("jabcode: GPU finder preparer is closed")
	}
	width := preparer.width
	height := preparer.height
	if width <= 0 || height <= 0 || width > preparer.resident.binarizer.maxWidth ||
		height > preparer.resident.binarizer.maxHeight {
		return fmt.Errorf("jabcode: GPU finder-average dimensions are unavailable")
	}
	if uint64(width)*uint64(height) > 1_000_000_000 {
		return fmt.Errorf("jabcode: GPU finder-average image exceeds exact partial-sum limit")
	}
	return nil
}

func (preparer *gpuFinderPassPreparer) ensureRetryControl() error {
	if preparer.retryControlBindings != nil {
		return nil
	}
	resident := preparer.resident
	if resident == nil || resident.foldSelection == nil || resident.foldRecord == nil ||
		resident.finderDecision == nil || resident.binarizer == nil {
		return fmt.Errorf("jabcode: resident GPU retry control has no finder decision")
	}
	kernel, err := preparer.kernels.finderRetryControl()
	if err != nil {
		return err
	}
	bindings, err := kernel.NewBindings(
		vulki.BindBuffer(0, resident.foldSelection),
		vulki.BindBuffer(1, resident.foldRecord),
		vulki.BindBuffer(2, resident.finderDecision),
		vulki.BindBuffer(3, resident.binarizer.params),
		vulki.BindBuffer(4, resident.binarizer.scanParams),
		vulki.BindBuffer(5, resident.binarizer.chainParams),
		vulki.BindBuffer(6, preparer.averageParams),
		vulki.BindBuffer(7, resident.finderDecisionIndirect),
	)
	if err != nil {
		return fmt.Errorf("jabcode: bind resident GPU retry control: %w", err)
	}
	preparer.retryControlKernel = kernel
	preparer.retryControlBindings = bindings
	return nil
}

func (preparer *gpuFinderPassPreparer) recordRetryControl(
	recorder *vulki.Recorder,
	stage uint32,
) error {
	if err := preparer.ensureRetryControl(); err != nil {
		return err
	}
	if err := recorder.Fill(
		preparer.averageParams,
		gpuFinderRetryControlStage,
		4,
		stage,
	); err != nil {
		return fmt.Errorf("jabcode: select resident GPU retry-control stage: %w", err)
	}
	if err := recorder.Barrier(preparer.averageParams); err != nil {
		return fmt.Errorf("jabcode: synchronize resident GPU retry-control stage: %w", err)
	}
	if err := recorder.Dispatch(
		preparer.retryControlKernel,
		preparer.retryControlBindings,
		vulki.Workgroups{X: 1, Y: 1, Z: 1},
	); err != nil {
		return fmt.Errorf("jabcode: dispatch resident GPU retry control: %w", err)
	}
	if err := recorder.Barrier(
		preparer.averageParams,
		preparer.resident.binarizer.params,
		preparer.resident.binarizer.scanParams,
		preparer.resident.binarizer.chainParams,
		preparer.resident.finderDecision,
		preparer.resident.finderDecisionIndirect,
	); err != nil {
		return fmt.Errorf("jabcode: synchronize resident GPU retry control: %w", err)
	}
	return nil
}

func (preparer *gpuFinderPassPreparer) recordAverageBatchRetry(
	recorder *vulki.Recorder,
) error {
	if err := preparer.validateAverage(); err != nil {
		return err
	}
	resident := preparer.resident
	bindings, err := resident.preparedBindingsFor(resident.balanced)
	if err != nil {
		return err
	}
	if err := recorder.DispatchIndirect(
		preparer.averageKernel,
		preparer.averageBindings,
		preparer.averageParams,
		gpuFinderRetryIndirectAverage,
	); err != nil {
		return fmt.Errorf("jabcode: dispatch resident GPU finder average: %w", err)
	}
	if err := recorder.Barrier(preparer.averagePartials); err != nil {
		return fmt.Errorf("jabcode: synchronize resident GPU finder-average partials: %w", err)
	}
	if err := recorder.DispatchIndirect(
		preparer.averageReduceKernel,
		preparer.averageReduceBindings,
		preparer.averageParams,
		gpuFinderRetryIndirectReduce,
	); err != nil {
		return fmt.Errorf("jabcode: dispatch resident GPU finder-average reduction: %w", err)
	}
	if err := recorder.Barrier(
		preparer.averageResult,
		resident.binarizer.params,
	); err != nil {
		return fmt.Errorf("jabcode: synchronize resident GPU finder-average reduction: %w", err)
	}
	_, err = resident.recordPreparedBinarizationIndirectLocked(
		recorder,
		bindings,
		1<<currentFamilySeekChannel,
		false,
		preparer.averageParams,
		gpuFinderRetryDispatchOffsets,
		false,
	)
	return err
}

func (preparer *gpuFinderPassPreparer) recordAverage(
	recorder *vulki.Recorder,
	fps []FinderPattern,
) error {
	params := gpuFinderAverageParams(preparer.width, preparer.height, fps)
	if err := recordGPUUpdate(
		recorder, "upload.finder_average_params", preparer.averageParams, 0, params[:],
	); err != nil {
		return fmt.Errorf("jabcode: update GPU finder-average parameters: %w", err)
	}
	if err := recorder.Dispatch(
		preparer.averageKernel,
		preparer.averageBindings,
		vulki.Workgroups{X: 4, Y: 1, Z: 1},
	); err != nil {
		return fmt.Errorf("jabcode: dispatch GPU finder-average kernel: %w", err)
	}
	if err := recorder.Barrier(preparer.averagePartials); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU finder-average partials: %w", err)
	}
	if err := recorder.Dispatch(
		preparer.averageReduceKernel,
		preparer.averageReduceBindings,
		vulki.Workgroups{X: 1, Y: 1, Z: 1},
	); err != nil {
		return fmt.Errorf("jabcode: dispatch GPU finder-average reduction: %w", err)
	}
	if err := recorder.Barrier(preparer.averageResult, preparer.resident.binarizer.params); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU finder-average result: %w", err)
	}
	return nil
}

func (preparer *gpuFinderPassPreparer) prepareAverage(
	fps []FinderPattern,
	scanChannels uint32,
) (preparedFinderPass, error) {
	var pass preparedFinderPass
	if err := preparer.validateAverage(); err != nil {
		return pass, err
	}
	resident := preparer.resident
	fixedThresholds := [3]float32{}

	resident.mu.Lock()
	err := func() error {
		defer resident.mu.Unlock()
		if resident.closed || resident.device == nil || resident.device.Closed() ||
			resident.binarizer == nil {
			return fmt.Errorf("jabcode: resident GPU binarizer is closed")
		}
		pixelCount, err := resident.validateBinarizationLocked(
			preparer.width, preparer.height, fixedThresholds[:])
		if err != nil {
			return err
		}
		input := resident.balanced
		if input == nil || input.Size() < uint64(pixelCount)*4 {
			return fmt.Errorf("jabcode: resident GPU prepared input buffer is too small")
		}
		params, blocksX, blocksY := gpuResidentBinarizerParams(
			preparer.width, preparer.height, fixedThresholds[:], false,
			resident.rowStride,
		)
		bindings, err := resident.preparedBindingsFor(input)
		if err != nil {
			return err
		}
		recorder, err := resident.device.NewRecorder()
		if err != nil {
			return fmt.Errorf("jabcode: create resident GPU average retry recorder: %w", err)
		}
		defer recorder.Abort()
		// The average reduction writes the three threshold words. Updating only
		// the prefix preserves them until that dispatch replaces them in this
		// same submission.
		if err := recordGPUUpdate(
			recorder, "upload.binarizer_params", resident.binarizer.params, 0,
			params[:gpuBinarizerFixedThresholdOffset],
		); err != nil {
			return fmt.Errorf("jabcode: update resident GPU average retry parameters: %w", err)
		}
		if err := preparer.recordAverage(recorder, fps); err != nil {
			return err
		}
		chainChannels, err := resident.recordPreparedBinarizationLocked(
			recorder, bindings, preparer.width, preparer.height,
			fixedThresholds[:], blocksX, blocksY, scanChannels, false,
			false,
		)
		if err != nil {
			return err
		}
		if preparer.trace {
			phaseprobe.Count("download.finder_average", len(preparer.averageBytes))
			if err := recorder.Download(preparer.averageResult, 0, preparer.averageBytes[:]); err != nil {
				return fmt.Errorf("jabcode: record GPU finder-average diagnostic download: %w", err)
			}
		}
		if err := recorder.SubmitAndWait(); err != nil {
			return fmt.Errorf("jabcode: run resident GPU average retry: %w", err)
		}
		pass.channels, pass.materialize = resident.lazyChannelsLocked(preparer.width, preparer.height)
		pass.rowHits = resident.finishScanHitsLocked(
			preparer.width, preparer.height, scanChannels, chainChannels, false)
		return nil
	}()
	if err != nil {
		return preparedFinderPass{}, err
	}
	if preparer.trace {
		for channel := range pass.average {
			pass.average[channel] = math.Float32frombits(
				binary.LittleEndian.Uint32(preparer.averageBytes[channel*4:]))
		}
		pass.input, err = resident.DownloadPrepared(
			resident.balanced, preparer.width, preparer.height)
		if err != nil {
			return preparedFinderPass{}, err
		}
	}
	return pass, nil
}

func gpuFinderAverageParams(width, height int, fps []FinderPattern) [gpuFinderAverageParamsSize]byte {
	var params [gpuFinderAverageParamsSize]byte
	binary.LittleEndian.PutUint32(params[0:], uint32(width))
	binary.LittleEndian.PutUint32(params[4:], uint32(height))
	for index := range 4 {
		if index >= len(fps) || fps[index].FoundCount <= 0 {
			continue
		}
		radius := fps[index].ModuleSize * 4
		startX := max(int(fps[index].Center.X-radius), 0)
		startY := max(int(fps[index].Center.Y-radius), 0)
		endX := min(int(fps[index].Center.X+radius), width-1)
		endY := min(int(fps[index].Center.Y+radius), height-1)
		offset := (2 + index*4) * 4
		binary.LittleEndian.PutUint32(params[offset+0:], uint32(startX))
		binary.LittleEndian.PutUint32(params[offset+4:], uint32(startY))
		binary.LittleEndian.PutUint32(params[offset+8:], uint32(endX))
		binary.LittleEndian.PutUint32(params[offset+12:], uint32(endY))
	}
	return params
}

func (preparer *gpuFinderPassPreparer) estimatePitch() (int, int, error) {
	if preparer == nil || preparer.pitchBindings == nil || preparer.width <= 0 || preparer.height <= 0 {
		return 0, 0, fmt.Errorf("jabcode: GPU pitch sampler is closed")
	}
	minDim := min(preparer.width, preparer.height)
	if minDim < 4 {
		return 0, 0, nil
	}
	// The resident fold follows the per-hit chains: bit-identical either way,
	// so it rides the same policy rather than being decided separately, and
	// falls back to the CPU twin whenever the pitch-lag kernels are not
	// compiled yet.
	if !preparer.resident.binarizer.scanOnly && preparer.kernels.pitchLagKernelsReady() {
		if px, py, err := preparer.estimatePitchResident(minDim); err == nil {
			return px, py, nil
		}
		// Any resident-lag failure degrades to the sample download below.
	}
	return preparer.estimatePitchDownloaded(minDim)
}

// estimatePitchResident folds the autocorrelation on the device and
// downloads only the summed lags; the result is bit-identical to
// estimatePitchDownloaded.
func (preparer *gpuFinderPassPreparer) estimatePitchResident(minDim int) (int, int, error) {
	rows, columns, maxLag, err := preparer.pitchResidentACF(minDim)
	if err != nil {
		return 0, 0, err
	}
	return dominantLagFromACF(rows, maxLag), dominantLagFromACF(columns, maxLag), nil
}

// pitchResidentACF samples the resident balanced canvas and reduces it to the
// two per-axis autocorrelation curves in one submission. Line normalization is
// part of the centering dispatch, so only the curves leave the device.
func (preparer *gpuFinderPassPreparer) pitchResidentACF(minDim int) (rows, columns []float64, maxLag int, err error) {
	if err := preparer.ensurePitchLag(); err != nil {
		return nil, nil, 0, err
	}
	maxLag = max(2, minDim/8)

	recorder, err := preparer.device.NewRecorder()
	if err != nil {
		return nil, nil, 0, fmt.Errorf("jabcode: create GPU pitch-sum recorder: %w", err)
	}
	defer recorder.Abort()
	if err := preparer.recordPitchACF(recorder, maxLag); err != nil {
		return nil, nil, 0, err
	}
	acfBytes := preparer.pitchLagACFBytes[:2*(maxLag+1)*4]
	phaseprobe.Count("download.pitch_autocorrelation", len(acfBytes))
	if err := recorder.Download(preparer.pitchLagACF, 0, acfBytes); err != nil {
		return nil, nil, 0, fmt.Errorf("jabcode: download GPU pitch autocorrelation: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return nil, nil, 0, fmt.Errorf("jabcode: run GPU pitch-lag submission: %w", err)
	}
	rows = make([]float64, maxLag+1)
	columns = make([]float64, maxLag+1)
	for lag := range rows {
		rows[lag] = getGPUScalar(acfBytes[lag*4:])
		columns[lag] = getGPUScalar(acfBytes[(maxLag+1+lag)*4:])
	}
	return rows, columns, maxLag, nil
}

func (preparer *gpuFinderPassPreparer) recordPitchACF(
	recorder *vulki.Recorder,
	maxLag int,
) error {
	sampleCount := gpuPitchSampleCount(preparer.width, preparer.height)
	sampleGroups := vulki.Workgroups{X: uint32((sampleCount + 63) / 64), Y: 1, Z: 1}
	if err := recorder.Dispatch(preparer.pitchKernel, preparer.pitchBindings, sampleGroups); err != nil {
		return fmt.Errorf("jabcode: dispatch GPU pitch-sample kernel: %w", err)
	}
	if err := recorder.Barrier(preparer.pitchSamples); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU pitch samples: %w", err)
	}
	if err := recorder.Dispatch(
		preparer.pitchLagSumsKernel,
		preparer.pitchLagSumsBindings,
		vulki.Workgroups{X: 1, Y: 1, Z: 1},
	); err != nil {
		return fmt.Errorf("jabcode: dispatch GPU pitch-sum kernel: %w", err)
	}
	if err := recorder.Barrier(preparer.pitchLagSums); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU pitch line sums: %w", err)
	}
	if err := recorder.Dispatch(
		preparer.pitchLagCenterKernel,
		preparer.pitchLagCenterBindings,
		sampleGroups,
	); err != nil {
		return fmt.Errorf("jabcode: dispatch GPU pitch-center kernel: %w", err)
	}
	if err := recorder.Barrier(preparer.pitchLagCentered); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU centered pitch samples: %w", err)
	}
	lagGroups := vulki.Workgroups{X: uint32((2*(maxLag+1) + 63) / 64), Y: 1, Z: 1}
	if err := recorder.Dispatch(preparer.pitchLagACFKernel, preparer.pitchLagACFBindings, lagGroups); err != nil {
		return fmt.Errorf("jabcode: dispatch GPU pitch-lag kernel: %w", err)
	}
	if err := recorder.Barrier(preparer.pitchLagACF); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU pitch autocorrelation: %w", err)
	}
	return nil
}

func (preparer *gpuFinderPassPreparer) recordPitchACFIndirect(
	recorder *vulki.Recorder,
) error {
	if err := recorder.DispatchIndirect(
		preparer.pitchKernel,
		preparer.pitchBindings,
		preparer.averageParams,
		gpuFinderRetryIndirectPitchSample,
	); err != nil {
		return fmt.Errorf("jabcode: dispatch resident GPU pitch samples: %w", err)
	}
	if err := recorder.Barrier(preparer.pitchSamples); err != nil {
		return fmt.Errorf("jabcode: synchronize resident GPU pitch samples: %w", err)
	}
	if err := recorder.DispatchIndirect(
		preparer.pitchLagSumsKernel,
		preparer.pitchLagSumsBindings,
		preparer.averageParams,
		gpuFinderRetryIndirectPitchOne,
	); err != nil {
		return fmt.Errorf("jabcode: dispatch resident GPU pitch sums: %w", err)
	}
	if err := recorder.Barrier(preparer.pitchLagSums); err != nil {
		return fmt.Errorf("jabcode: synchronize resident GPU pitch sums: %w", err)
	}
	if err := recorder.DispatchIndirect(
		preparer.pitchLagCenterKernel,
		preparer.pitchLagCenterBindings,
		preparer.averageParams,
		gpuFinderRetryIndirectPitchCenter,
	); err != nil {
		return fmt.Errorf("jabcode: dispatch resident GPU pitch centering: %w", err)
	}
	if err := recorder.Barrier(preparer.pitchLagCentered); err != nil {
		return fmt.Errorf("jabcode: synchronize resident GPU centered pitch samples: %w", err)
	}
	if err := recorder.DispatchIndirect(
		preparer.pitchLagACFKernel,
		preparer.pitchLagACFBindings,
		preparer.averageParams,
		gpuFinderRetryIndirectPitchACF,
	); err != nil {
		return fmt.Errorf("jabcode: dispatch resident GPU pitch autocorrelation: %w", err)
	}
	if err := recorder.Barrier(preparer.pitchLagACF); err != nil {
		return fmt.Errorf("jabcode: synchronize resident GPU pitch autocorrelation: %w", err)
	}
	return nil
}

// estimatePitchDownloaded is the fallback estimate: download every luma
// sample and fold the autocorrelation on the host.
func (preparer *gpuFinderPassPreparer) estimatePitchDownloaded(minDim int) (int, int, error) {
	sampleCount := gpuPitchSampleCount(preparer.width, preparer.height)
	if preparer.pitchBytes == nil {
		maxSamples := gpuPitchSampleCount(
			preparer.resident.binarizer.maxWidth,
			preparer.resident.binarizer.maxHeight,
		)
		preparer.pitchBytes = make([]byte, maxSamples*4)
	}
	samples := preparer.pitchBytes[:sampleCount*4]
	recorder, err := preparer.device.NewRecorder()
	if err != nil {
		return 0, 0, fmt.Errorf("jabcode: create GPU pitch-sample recorder: %w", err)
	}
	defer recorder.Abort()
	groups := uint32((sampleCount + 63) / 64)
	if err := recorder.Dispatch(
		preparer.pitchKernel,
		preparer.pitchBindings,
		vulki.Workgroups{X: groups, Y: 1, Z: 1},
	); err != nil {
		return 0, 0, fmt.Errorf("jabcode: dispatch GPU pitch-sample kernel: %w", err)
	}
	if err := recorder.Barrier(preparer.pitchSamples); err != nil {
		return 0, 0, fmt.Errorf("jabcode: synchronize GPU pitch samples: %w", err)
	}
	phaseprobe.Count("download.pitch_samples", len(samples))
	if err := recorder.Download(preparer.pitchSamples, 0, samples); err != nil {
		return 0, 0, fmt.Errorf("jabcode: download GPU pitch samples: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return 0, 0, fmt.Errorf("jabcode: run GPU pitch-sample kernel: %w", err)
	}
	rows, columns := decodeGPUPitchSamples(samples, preparer.width, preparer.height)
	maxLag := max(2, minDim/8)
	return dominantLag(rows, maxLag), dominantLag(columns, maxLag), nil
}

// getGPUScalar reads one kernel-side f32.
func getGPUScalar(b []byte) float64 {
	return float64(math.Float32frombits(binary.LittleEndian.Uint32(b)))
}

func gpuPitchSampleCount(width, height int) int {
	return min(pitchSampleLines, height)*width + min(pitchSampleLines, width)*height
}

func decodeGPUPitchSamples(samples []byte, width, height int) (rows, columns [][]float64) {
	rowCount := min(pitchSampleLines, height)
	columnCount := min(pitchSampleLines, width)
	rows = make([][]float64, rowCount)
	offset := 0
	for row := range rows {
		rows[row] = make([]float64, width)
		for x := range width {
			rows[row][x] = float64(binary.LittleEndian.Uint32(samples[offset:])) / 3
			offset += 4
		}
	}
	columns = make([][]float64, columnCount)
	for column := range columns {
		columns[column] = make([]float64, height)
		for y := range height {
			columns[column][y] = float64(binary.LittleEndian.Uint32(samples[offset:])) / 3
			offset += 4
		}
	}
	return rows, columns
}

// residentRetriesReady reports whether the whole recorded retry graph exists,
// not just the reduction that drives it. The indirect scan has no host-download
// fallback, so it needs the row chain as much as the pitch-lag kernels, and the
// caller commits to the resident schedule before its first retry: an unmet
// requirement discovered while recording fails the locate outright instead of
// degrading. Background warm compiles the chains first, so this costs nothing
// on a healthy device and keeps a failed chain compile on the host schedule.
func (preparer *gpuFinderPassPreparer) residentRetriesReady() bool {
	return preparer != nil && !preparer.trace && preparer.width >= 4 && preparer.height >= 4 &&
		preparer.resident != nil && preparer.resident.binarizer != nil &&
		!preparer.resident.binarizer.scanOnly && preparer.kernels.pitchLagKernelsReady() &&
		preparer.kernels.finderChainsReady()
}

// prepareResidentRetry lets one device reduction drive every descreen and
// print retry without exposing its histogram, pitch curves, peak choices or
// pass order. Inactive records produce empty masks, so the host can preserve
// the bounded traversal order without learning whether a record was admitted.
func (preparer *gpuFinderPassPreparer) prepareResidentRetry(
	retry int,
	scanChannels uint32,
) (preparedFinderPass, error) {
	var pass preparedFinderPass
	if !preparer.residentRetriesReady() || retry < finderRetryDescreenFirst || retry > finderRetryPrintSecond ||
		scanChannels != 1<<currentFamilySeekChannel {
		return pass, fmt.Errorf("jabcode: resident GPU retry schedule is unavailable")
	}
	if err := preparer.ensurePitchSchedule(); err != nil {
		return pass, err
	}
	resident := preparer.resident
	resident.mu.Lock()
	err := func() error {
		defer resident.mu.Unlock()
		if resident.closed || resident.device == nil || resident.device.Closed() ||
			resident.binarizer == nil {
			return fmt.Errorf("jabcode: resident GPU binarizer is closed")
		}
		pixelCount, err := resident.validateBinarizationLocked(preparer.width, preparer.height, nil)
		if err != nil {
			return err
		}
		if preparer.descreenFiltered == nil || preparer.descreenFiltered.Size() < uint64(pixelCount)*4 {
			return fmt.Errorf("jabcode: resident GPU retry input buffer is too small")
		}
		preparedBindings, err := resident.preparedBindingsFor(preparer.descreenFiltered)
		if err != nil {
			return err
		}
		recorder, err := resident.device.NewRecorder()
		if err != nil {
			return fmt.Errorf("jabcode: create resident GPU retry recorder: %w", err)
		}
		defer recorder.Abort()
		if err := recorder.Fill(
			resident.binarizer.params,
			gpuBinarizerScanCapacityOffset,
			4,
			uint32(resident.binarizer.scanCapacity),
		); err != nil {
			return fmt.Errorf("jabcode: set resident GPU retry scan capacity: %w", err)
		}
		if !preparer.pitchScheduleReady {
			if retry != finderRetryDescreenFirst {
				return fmt.Errorf("jabcode: resident GPU retry schedule was not initialized")
			}
			if err := preparer.recordPitchSchedule(recorder); err != nil {
				return err
			}
		}
		if retry == finderRetryPrintFirst {
			if err := preparer.recordPitchReduction(
				recorder, gpuPitchStagePrint, nil, 0, "print",
			); err != nil {
				return err
			}
		}
		// The host reaches each retry only after the preceding locate failed.
		// Re-arm that fact explicitly because selection writes its result back
		// into this word; an inactive schedule slot must not suppress a later,
		// independently admitted print retry.
		if err := recorder.Fill(
			resident.binarizer.params,
			gpuBinarizerRetryActiveOffset,
			4,
			1,
		); err != nil {
			return fmt.Errorf("jabcode: arm resident GPU retry: %w", err)
		}
		if err := recorder.Fill(
			preparer.pitchScheduleControl,
			uint64(gpuPitchControlSelector*4),
			4,
			uint32(retry),
		); err != nil {
			return fmt.Errorf("jabcode: select resident GPU retry: %w", err)
		}
		if err := recorder.Fill(
			preparer.pitchScheduleControl,
			uint64(gpuPitchControlStage*4),
			4,
			gpuPitchStageSelect,
		); err != nil {
			return fmt.Errorf("jabcode: select resident GPU retry stage: %w", err)
		}
		if err := recorder.Barrier(
			preparer.pitchScheduleControl,
			resident.binarizer.params,
		); err != nil {
			return fmt.Errorf("jabcode: synchronize resident GPU retry selection: %w", err)
		}
		if err := recorder.Dispatch(
			preparer.pitchScheduleKernel,
			preparer.pitchScheduleBinding,
			vulki.Workgroups{X: 1, Y: 1, Z: 1},
		); err != nil {
			return fmt.Errorf("jabcode: dispatch resident GPU retry selection: %w", err)
		}
		if err := recorder.Barrier(
			preparer.descreenParams,
			preparer.pitchSchedule,
			resident.binarizer.params,
			resident.binarizer.scanParams,
			resident.binarizer.chainParams,
		); err != nil {
			return fmt.Errorf("jabcode: synchronize resident GPU retry controls: %w", err)
		}
		if err := recorder.DispatchIndirect(
			preparer.descreenHorizontal,
			preparer.descreenHBindings,
			preparer.pitchSchedule,
			gpuPitchRetryIndirectCanvas,
		); err != nil {
			return fmt.Errorf("jabcode: dispatch resident GPU horizontal retry filter: %w", err)
		}
		if err := recorder.Barrier(preparer.descreenLinear); err != nil {
			return fmt.Errorf("jabcode: synchronize resident GPU horizontal retry filter: %w", err)
		}
		if err := recorder.DispatchIndirect(
			preparer.descreenVertical,
			preparer.descreenVBindings,
			preparer.pitchSchedule,
			gpuPitchRetryIndirectCanvas,
		); err != nil {
			return fmt.Errorf("jabcode: dispatch resident GPU vertical retry filter: %w", err)
		}
		if err := recorder.Barrier(preparer.descreenFiltered); err != nil {
			return fmt.Errorf("jabcode: synchronize resident GPU retry filter: %w", err)
		}
		printLevels := retry >= finderRetryPrintFirst
		chainChannels, err := resident.recordPreparedBinarizationIndirectLocked(
			recorder,
			preparedBindings,
			scanChannels,
			printLevels,
			preparer.pitchSchedule,
			gpuPitchRetryDispatchOffsets,
			true,
		)
		if err != nil {
			return err
		}
		if err := recorder.SubmitAndWait(); err != nil {
			return fmt.Errorf("jabcode: run resident GPU retry: %w", err)
		}
		preparer.pitchScheduleReady = true
		pass.channels, pass.materialize = resident.lazyChannelsLocked(preparer.width, preparer.height)
		pass.rowHits = resident.finishScanHitsLocked(
			preparer.width,
			preparer.height,
			scanChannels,
			chainChannels,
			printLevels,
		)
		return nil
	}()
	if err != nil {
		return preparedFinderPass{}, err
	}
	return pass, nil
}

func (preparer *gpuFinderPassPreparer) recordPitchSchedule(recorder *vulki.Recorder) error {
	maxLag := max(2, min(preparer.width, preparer.height)/8)
	if err := recorder.Fill(
		preparer.pitchScheduleControl,
		0,
		gpuPitchScheduleControlWords*4,
		math.MaxUint32,
	); err != nil {
		return fmt.Errorf("jabcode: reset GPU pitch schedule control: %w", err)
	}
	if err := recorder.Fill(
		preparer.pitchScheduleControl,
		uint64(gpuPitchControlRowPeak*4),
		2*4,
		0,
	); err != nil {
		return fmt.Errorf("jabcode: reset GPU pitch peaks: %w", err)
	}
	if err := recorder.Fill(preparer.pitchSchedule, 0, gpuPitchScheduleWords*4, 0); err != nil {
		return fmt.Errorf("jabcode: reset GPU retry schedule: %w", err)
	}
	if err := recorder.Barrier(
		preparer.resident.binarizer.params,
		preparer.pitchScheduleControl,
		preparer.pitchSchedule,
	); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU pitch schedule reset: %w", err)
	}
	if err := preparer.recordPitchACF(recorder, maxLag); err != nil {
		return err
	}
	groups := vulki.Workgroups{X: uint32((2*(maxLag+1) + 255) / 256), Y: 1, Z: 1}
	for stage := gpuPitchStageValley; stage <= gpuPitchStageLag; stage++ {
		if err := recorder.Fill(
			preparer.pitchScheduleControl,
			uint64(gpuPitchControlStage*4),
			4,
			uint32(stage),
		); err != nil {
			return fmt.Errorf("jabcode: select GPU pitch schedule stage %d: %w", stage, err)
		}
		if err := recorder.Barrier(preparer.pitchScheduleControl); err != nil {
			return fmt.Errorf("jabcode: synchronize GPU pitch schedule stage %d selection: %w", stage, err)
		}
		if err := recorder.Dispatch(
			preparer.pitchScheduleKernel,
			preparer.pitchScheduleBinding,
			groups,
		); err != nil {
			return fmt.Errorf("jabcode: dispatch GPU pitch schedule stage %d: %w", stage, err)
		}
		if err := recorder.Barrier(preparer.pitchScheduleControl); err != nil {
			return fmt.Errorf("jabcode: synchronize GPU pitch schedule stage %d: %w", stage, err)
		}
	}
	return preparer.recordPitchReduction(
		recorder, gpuPitchStageSchedule, nil, 0, "descreen",
	)
}

func (preparer *gpuFinderPassPreparer) recordPitchScheduleBatch(
	recorder *vulki.Recorder,
) error {
	if err := preparer.ensurePitchSchedule(); err != nil {
		return err
	}
	if err := recorder.Fill(
		preparer.pitchScheduleControl,
		0,
		gpuPitchScheduleControlWords*4,
		math.MaxUint32,
	); err != nil {
		return fmt.Errorf("jabcode: reset resident GPU pitch schedule control: %w", err)
	}
	if err := recorder.Fill(
		preparer.pitchScheduleControl,
		uint64(gpuPitchControlRowPeak*4),
		2*4,
		0,
	); err != nil {
		return fmt.Errorf("jabcode: reset resident GPU pitch peaks: %w", err)
	}
	if err := recorder.Fill(preparer.pitchSchedule, 0, gpuPitchScheduleWords*4, 0); err != nil {
		return fmt.Errorf("jabcode: reset resident GPU retry schedule: %w", err)
	}
	if err := recorder.Barrier(
		preparer.pitchScheduleControl,
		preparer.pitchSchedule,
	); err != nil {
		return fmt.Errorf("jabcode: synchronize resident GPU retry schedule reset: %w", err)
	}
	if err := preparer.recordPitchACFIndirect(recorder); err != nil {
		return err
	}
	for stage := gpuPitchStageValley; stage <= gpuPitchStageLag; stage++ {
		if err := recorder.Fill(
			preparer.pitchScheduleControl,
			uint64(gpuPitchControlStage*4),
			4,
			uint32(stage),
		); err != nil {
			return fmt.Errorf("jabcode: select resident GPU pitch stage %d: %w", stage, err)
		}
		if err := recorder.Barrier(preparer.pitchScheduleControl); err != nil {
			return fmt.Errorf("jabcode: synchronize resident GPU pitch stage %d: %w", stage, err)
		}
		if err := recorder.DispatchIndirect(
			preparer.pitchScheduleKernel,
			preparer.pitchScheduleBinding,
			preparer.averageParams,
			gpuFinderRetryIndirectPitchSelect,
		); err != nil {
			return fmt.Errorf("jabcode: dispatch resident GPU pitch stage %d: %w", stage, err)
		}
		if err := recorder.Barrier(preparer.pitchScheduleControl); err != nil {
			return fmt.Errorf("jabcode: finish resident GPU pitch stage %d: %w", stage, err)
		}
	}
	return preparer.recordPitchReduction(
		recorder, gpuPitchStageSchedule, preparer.averageParams,
		gpuFinderRetryIndirectPitchOne, "resident descreen",
	)
}

func (preparer *gpuFinderPassPreparer) recordPitchReduction(
	recorder *vulki.Recorder,
	stage uint32,
	indirect *vulki.Buffer,
	offset uint64,
	label string,
) error {
	if err := recorder.Fill(
		preparer.pitchScheduleControl,
		uint64(gpuPitchControlStage*4),
		4,
		stage,
	); err != nil {
		return fmt.Errorf("jabcode: select %s GPU retry reduction: %w", label, err)
	}
	if err := recorder.Barrier(preparer.pitchScheduleControl); err != nil {
		return fmt.Errorf("jabcode: synchronize %s GPU retry reduction: %w", label, err)
	}
	var err error
	if indirect == nil {
		err = recorder.Dispatch(
			preparer.pitchScheduleKernel,
			preparer.pitchScheduleBinding,
			vulki.Workgroups{X: 1, Y: 1, Z: 1},
		)
	} else {
		err = recorder.DispatchIndirect(
			preparer.pitchScheduleKernel,
			preparer.pitchScheduleBinding,
			indirect,
			offset,
		)
	}
	if err != nil {
		return fmt.Errorf("jabcode: dispatch %s GPU retry reduction: %w", label, err)
	}
	if err := recorder.Barrier(
		preparer.pitchScheduleControl,
		preparer.pitchSchedule,
		preparer.resident.binarizer.seedHistogram,
	); err != nil {
		return fmt.Errorf("jabcode: finish %s GPU retry reduction: %w", label, err)
	}
	return nil
}

func (preparer *gpuFinderPassPreparer) recordScheduledBatchRetry(
	recorder *vulki.Recorder,
	retry int,
) error {
	if retry < finderRetryDescreenFirst || retry > finderRetryPrintSecond {
		return fmt.Errorf("jabcode: invalid resident GPU retry slot %d", retry)
	}
	resident := preparer.resident
	if err := recorder.Fill(
		resident.binarizer.params,
		gpuBinarizerScanCapacityOffset,
		4,
		uint32(resident.binarizer.scanCapacity),
	); err != nil {
		return fmt.Errorf("jabcode: set resident GPU retry scan capacity: %w", err)
	}
	if err := recorder.Fill(
		preparer.pitchScheduleControl,
		uint64(gpuPitchControlSelector*4),
		4,
		uint32(retry),
	); err != nil {
		return fmt.Errorf("jabcode: select resident GPU retry slot: %w", err)
	}
	if err := recorder.Fill(
		preparer.pitchScheduleControl,
		uint64(gpuPitchControlStage*4),
		4,
		gpuPitchStageSelect,
	); err != nil {
		return fmt.Errorf("jabcode: select resident GPU retry stage: %w", err)
	}
	if err := recorder.Barrier(
		preparer.pitchScheduleControl,
		resident.binarizer.params,
	); err != nil {
		return fmt.Errorf("jabcode: synchronize resident GPU retry selection: %w", err)
	}
	if err := recorder.Dispatch(
		preparer.pitchScheduleKernel,
		preparer.pitchScheduleBinding,
		vulki.Workgroups{X: 1, Y: 1, Z: 1},
	); err != nil {
		return fmt.Errorf("jabcode: dispatch resident GPU retry selection: %w", err)
	}
	if err := recorder.Barrier(
		preparer.descreenParams,
		preparer.pitchSchedule,
		resident.binarizer.params,
		resident.binarizer.scanParams,
		resident.binarizer.chainParams,
		resident.finderDecisionIndirect,
	); err != nil {
		return fmt.Errorf("jabcode: synchronize resident GPU retry controls: %w", err)
	}
	if err := recorder.DispatchIndirect(
		preparer.descreenHorizontal,
		preparer.descreenHBindings,
		preparer.pitchSchedule,
		gpuPitchRetryIndirectCanvas,
	); err != nil {
		return fmt.Errorf("jabcode: dispatch resident GPU horizontal retry filter: %w", err)
	}
	if err := recorder.Barrier(preparer.descreenLinear); err != nil {
		return fmt.Errorf("jabcode: synchronize resident GPU horizontal retry filter: %w", err)
	}
	if err := recorder.DispatchIndirect(
		preparer.descreenVertical,
		preparer.descreenVBindings,
		preparer.pitchSchedule,
		gpuPitchRetryIndirectCanvas,
	); err != nil {
		return fmt.Errorf("jabcode: dispatch resident GPU vertical retry filter: %w", err)
	}
	if err := recorder.Barrier(preparer.descreenFiltered); err != nil {
		return fmt.Errorf("jabcode: synchronize resident GPU retry filter: %w", err)
	}
	bindings, err := resident.preparedBindingsFor(preparer.descreenFiltered)
	if err != nil {
		return err
	}
	_, err = resident.recordPreparedBinarizationIndirectLocked(
		recorder,
		bindings,
		1<<currentFamilySeekChannel,
		retry >= finderRetryPrintFirst,
		preparer.pitchSchedule,
		gpuPitchRetryDispatchOffsets,
		false,
	)
	return err
}

func (preparer *gpuFinderPassPreparer) prepare(
	rx, ry int,
	thresholds []float32,
	printLevels bool,
	scanChannels uint32,
) (*core.Bitmap, [3]*core.Bitmap, *finderPassRowHits, func() error, error) {
	input := preparer.resident.balanced
	if rx > 0 || ry > 0 {
		if err := preparer.descreen(rx, ry); err != nil {
			return nil, [3]*core.Bitmap{}, nil, nil, err
		}
		input = preparer.descreenFiltered
	}
	channels, hits, materialize, err := preparer.resident.BinarizePrepared(
		input,
		preparer.width,
		preparer.height,
		thresholds,
		printLevels,
		scanChannels,
	)
	if err != nil {
		return nil, [3]*core.Bitmap{}, nil, nil, err
	}
	if !preparer.trace {
		return nil, channels, hits, materialize, nil
	}
	inputBitmap, err := preparer.resident.DownloadPrepared(input, preparer.width, preparer.height)
	if err != nil {
		return nil, [3]*core.Bitmap{}, nil, nil, err
	}
	return inputBitmap, channels, hits, materialize, nil
}

// scanDirection runs the fused directional kernel over the masks the last
// prepare left resident, so the sweep costs no upload and no expansion.
func (preparer *gpuFinderPassPreparer) scanDirection(
	dir scanDirection,
	step, channel int,
) (finderDirSweep, error) {
	if preparer == nil || preparer.resident == nil {
		return finderDirSweep{}, nil
	}
	return preparer.resident.ScanDirection(preparer.width, preparer.height, dir, step, channel)
}

func (preparer *gpuFinderPassPreparer) scanDirectionBatch(
	dirs []scanDirection,
	step, channel int,
) ([]finderDirSweep, error) {
	if preparer == nil || preparer.resident == nil {
		return nil, nil
	}
	return preparer.resident.ScanDirectionBatch(preparer.width, preparer.height, dirs, step, channel)
}

// foldDirection assembles and selects a resident direction where the chain left
// its outcomes, so the candidates never cross in either direction.
func (preparer *gpuFinderPassPreparer) foldDirection(
	sweep finderDirSweep,
	printPass bool,
) (*finderDirQuad, error) {
	if preparer == nil || preparer.resident == nil || !sweep.resident {
		return nil, nil
	}
	return preparer.resident.FoldDirection(
		image.Pt(preparer.width, preparer.height), sweep, printPass, preparer.trace)
}

// foldRow assembles and selects the row pass where its chain left the compacted
// candidates, so they never cross for a pass the device can answer.
func (preparer *gpuFinderPassPreparer) foldRow(
	channel, count int,
	printPass bool,
) (*finderDirQuad, error) {
	if preparer == nil || preparer.resident == nil {
		return nil, nil
	}
	return preparer.resident.FoldRow(
		image.Pt(preparer.width, preparer.height), channel, count, printPass, preparer.trace)
}

// foldRowVertical sweeps the column direction and folds it with the row region,
// for the pass the row fold alone cannot answer.
func (preparer *gpuFinderPassPreparer) foldRowVertical(
	channel, count, step int,
	printPass bool,
) (*finderDirQuad, error) {
	if preparer == nil || preparer.resident == nil {
		return nil, nil
	}
	return preparer.resident.FoldRowVertical(
		image.Pt(preparer.width, preparer.height), channel, count, step, printPass,
		preparer.trace)
}

func (preparer *gpuFinderPassPreparer) descreen(rx, ry int) error {
	if preparer == nil || preparer.device == nil {
		return fmt.Errorf("jabcode: GPU descreen preparer is closed")
	}
	if err := preparer.ensureDescreen(); err != nil {
		return err
	}
	var params [16]byte
	binary.LittleEndian.PutUint32(params[0:], uint32(preparer.width))
	binary.LittleEndian.PutUint32(params[4:], uint32(preparer.height))
	binary.LittleEndian.PutUint32(params[8:], uint32(max(rx, 0)))
	binary.LittleEndian.PutUint32(params[12:], uint32(max(ry, 0)))
	recorder, err := preparer.device.NewRecorder()
	if err != nil {
		return fmt.Errorf("jabcode: create GPU descreen recorder: %w", err)
	}
	defer recorder.Abort()
	if err := recordGPUUpdate(
		recorder, "upload.descreen_params", preparer.descreenParams, 0, params[:],
	); err != nil {
		return fmt.Errorf("jabcode: update GPU descreen parameters: %w", err)
	}
	groups := gpuCanvasWorkgroups(preparer.width, preparer.height)
	if err := recorder.Dispatch(preparer.descreenHorizontal, preparer.descreenHBindings, groups); err != nil {
		return fmt.Errorf("jabcode: dispatch GPU horizontal descreen: %w", err)
	}
	if err := recorder.Barrier(preparer.descreenLinear); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU horizontal descreen: %w", err)
	}
	if err := recorder.Dispatch(preparer.descreenVertical, preparer.descreenVBindings, groups); err != nil {
		return fmt.Errorf("jabcode: dispatch GPU vertical descreen: %w", err)
	}
	if err := recorder.Barrier(preparer.descreenFiltered); err != nil {
		return fmt.Errorf("jabcode: synchronize GPU vertical descreen: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return fmt.Errorf("jabcode: run GPU descreen: %w", err)
	}
	return nil
}

// closeDescreen releases the lazily-created descreen chain. The descreen
// kernels stay in the shared per-device set; only references are dropped.
func (preparer *gpuFinderPassPreparer) closeDescreen() error {
	var closeErrors []error
	for _, bindings := range []*vulki.BindingSet{
		preparer.descreenVBindings,
		preparer.descreenHBindings,
	} {
		if bindings != nil {
			closeErrors = append(closeErrors, bindings.Close())
		}
	}
	preparer.descreenVBindings = nil
	preparer.descreenHBindings = nil
	preparer.descreenVertical = nil
	preparer.descreenHorizontal = nil
	for _, buffer := range []*vulki.Buffer{
		preparer.descreenFiltered,
		preparer.descreenLinear,
		preparer.descreenParams,
	} {
		if buffer != nil {
			closeErrors = append(closeErrors, buffer.Close())
		}
	}
	preparer.descreenFiltered = nil
	preparer.descreenLinear = nil
	preparer.descreenParams = nil
	return errors.Join(closeErrors...)
}

func (preparer *gpuFinderPassPreparer) Close() error {
	if preparer == nil {
		return nil
	}
	closeErrors := []error{
		preparer.closePitchSchedule(),
		preparer.closeDescreen(),
		preparer.closePitchLag(),
	}
	if preparer.retryControlBindings != nil {
		closeErrors = append(closeErrors, preparer.retryControlBindings.Close())
		preparer.retryControlBindings = nil
	}
	preparer.retryControlKernel = nil
	if preparer.pitchBindings != nil {
		closeErrors = append(closeErrors, preparer.pitchBindings.Close())
		preparer.pitchBindings = nil
	}
	preparer.pitchKernel = nil
	if preparer.pitchSamples != nil {
		closeErrors = append(closeErrors, preparer.pitchSamples.Close())
		preparer.pitchSamples = nil
	}
	if preparer.averageReduceBindings != nil {
		closeErrors = append(closeErrors, preparer.averageReduceBindings.Close())
		preparer.averageReduceBindings = nil
	}
	preparer.averageReduceKernel = nil
	if preparer.averageResult != nil {
		closeErrors = append(closeErrors, preparer.averageResult.Close())
		preparer.averageResult = nil
	}
	if preparer.averageBindings != nil {
		closeErrors = append(closeErrors, preparer.averageBindings.Close())
		preparer.averageBindings = nil
	}
	preparer.averageKernel = nil
	if preparer.averagePartials != nil {
		closeErrors = append(closeErrors, preparer.averagePartials.Close())
		preparer.averagePartials = nil
	}
	if preparer.averageParams != nil {
		closeErrors = append(closeErrors, preparer.averageParams.Close())
		preparer.averageParams = nil
	}
	preparer.device = nil
	return errors.Join(closeErrors...)
}
