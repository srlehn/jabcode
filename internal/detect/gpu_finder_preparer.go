package detect

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
)

//go:embed shaders/finder_average.wgsl
var finderAverageWGSL string

//go:embed shaders/pitch_samples.wgsl
var pitchSamplesWGSL string

//go:embed shaders/pitch_line_sums.wgsl
var pitchLineSumsWGSL string

//go:embed shaders/pitch_center.wgsl
var pitchCenterWGSL string

//go:embed shaders/pitch_acf.wgsl
var pitchACFWGSL string

//go:embed shaders/descreen_horizontal.wgsl
var descreenHorizontalWGSL string

//go:embed shaders/descreen_vertical.wgsl
var descreenVerticalWGSL string

const (
	gpuFinderAverageParamsSize  = 18 * 4
	gpuFinderAveragePartialSize = 4 * 64 * 4 * 4
	gpuPitchParamsSize          = 4 * 4
	gpuDescreenParamsSize       = 4 * 4
	gpuPitchLagParamsSize       = 12 * 4
	// gpuPitchLagLineBytes holds one float64 per sampled line per axis, for
	// the line-sum and line-mean buffers of the resident autocorrelation.
	gpuPitchLagLineBytes = 2 * pitchSampleLines * 8
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
	partialBytes    [gpuFinderAveragePartialSize]byte

	pitchParams   *vulki.Buffer
	pitchSamples  *vulki.Buffer
	pitchKernel   *vulki.Kernel
	pitchBindings *vulki.BindingSet
	// pitchBytes is the host destination of the fallback sample download,
	// allocated on the first estimate that runs without the lag kernels.
	pitchBytes []byte

	pitchLagParams         *vulki.Buffer
	pitchLagSums           *vulki.Buffer
	pitchLagMeans          *vulki.Buffer
	pitchLagCentered       *vulki.Buffer
	pitchLagACF            *vulki.Buffer
	pitchLagSumsKernel     *vulki.Kernel
	pitchLagCenterKernel   *vulki.Kernel
	pitchLagACFKernel      *vulki.Kernel
	pitchLagSumsBindings   *vulki.BindingSet
	pitchLagCenterBindings *vulki.BindingSet
	pitchLagACFBindings    *vulki.BindingSet
	pitchLagLineBytes      []byte
	pitchLagACFBytes       []byte

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
	preparer.averageParams, err = device.NewBuffer(gpuFinderAverageParamsSize)
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
	preparer.pitchParams, err = device.NewBuffer(gpuPitchParamsSize)
	if err != nil {
		_ = preparer.Close()
		return nil, fmt.Errorf("jabcode: allocate GPU pitch-sample parameters: %w", err)
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
		vulki.BindBuffer(2, preparer.pitchParams),
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
// at eight bytes per sampled pixel.
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
	preparer.pitchLagParams, err = preparer.device.NewBuffer(gpuPitchLagParamsSize)
	if err != nil {
		return fmt.Errorf("jabcode: allocate GPU pitch-lag parameters: %w", err)
	}
	preparer.pitchLagSums, err = preparer.device.NewBuffer(gpuPitchLagLineBytes)
	if err != nil {
		_ = preparer.closePitchLag()
		return fmt.Errorf("jabcode: allocate GPU pitch line sums: %w", err)
	}
	preparer.pitchLagMeans, err = preparer.device.NewBuffer(gpuPitchLagLineBytes)
	if err != nil {
		_ = preparer.closePitchLag()
		return fmt.Errorf("jabcode: allocate GPU pitch means: %w", err)
	}
	preparer.pitchLagCentered, err = preparer.device.NewBuffer(uint64(maxSamples) * 8)
	if err != nil {
		_ = preparer.closePitchLag()
		return fmt.Errorf("jabcode: allocate GPU centered pitch samples: %w", err)
	}
	preparer.pitchLagACF, err = preparer.device.NewBuffer(uint64(2*maxLags) * 8)
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
		vulki.BindBuffer(2, preparer.pitchLagParams),
	)
	if err != nil {
		_ = preparer.closePitchLag()
		return fmt.Errorf("jabcode: bind GPU pitch-sum kernel: %w", err)
	}
	preparer.pitchLagCenterBindings, err = preparer.pitchLagCenterKernel.NewBindings(
		vulki.BindBuffer(0, preparer.pitchSamples),
		vulki.BindBuffer(1, preparer.pitchLagMeans),
		vulki.BindBuffer(2, preparer.pitchLagCentered),
		vulki.BindBuffer(3, preparer.pitchLagParams),
	)
	if err != nil {
		_ = preparer.closePitchLag()
		return fmt.Errorf("jabcode: bind GPU pitch-center kernel: %w", err)
	}
	preparer.pitchLagACFBindings, err = preparer.pitchLagACFKernel.NewBindings(
		vulki.BindBuffer(0, preparer.pitchLagCentered),
		vulki.BindBuffer(1, preparer.pitchLagACF),
		vulki.BindBuffer(2, preparer.pitchLagParams),
	)
	if err != nil {
		_ = preparer.closePitchLag()
		return fmt.Errorf("jabcode: bind GPU pitch-lag kernel: %w", err)
	}
	preparer.pitchLagLineBytes = make([]byte, 2*pitchSampleLines*8)
	preparer.pitchLagACFBytes = make([]byte, 2*maxLags*8)
	return nil
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
		preparer.pitchLagMeans,
		preparer.pitchLagSums,
		preparer.pitchLagParams,
	} {
		if buffer != nil {
			closeErrors = append(closeErrors, buffer.Close())
		}
	}
	preparer.pitchLagACF = nil
	preparer.pitchLagCentered = nil
	preparer.pitchLagMeans = nil
	preparer.pitchLagSums = nil
	preparer.pitchLagParams = nil
	return errors.Join(closeErrors...)
}

func (preparer *gpuFinderPassPreparer) averagePixelValue(
	fps []FinderPattern,
) ([3]float32, error) {
	var empty [3]float32
	if preparer == nil || preparer.resident == nil || preparer.averageBindings == nil {
		return empty, fmt.Errorf("jabcode: GPU finder preparer is closed")
	}
	width := preparer.width
	height := preparer.height
	if width <= 0 || height <= 0 || width > preparer.resident.binarizer.maxWidth ||
		height > preparer.resident.binarizer.maxHeight {
		return empty, fmt.Errorf("jabcode: GPU finder-average dimensions are unavailable")
	}
	if uint64(width)*uint64(height) > 1_000_000_000 {
		return empty, fmt.Errorf("jabcode: GPU finder-average image exceeds exact partial-sum limit")
	}
	params := gpuFinderAverageParams(width, height, fps)
	recorder, err := preparer.device.NewRecorder()
	if err != nil {
		return empty, fmt.Errorf("jabcode: create GPU finder-average recorder: %w", err)
	}
	defer recorder.Abort()
	if err := recorder.Update(preparer.averageParams, 0, params[:]); err != nil {
		return empty, fmt.Errorf("jabcode: update GPU finder-average parameters: %w", err)
	}
	if err := recorder.Dispatch(
		preparer.averageKernel,
		preparer.averageBindings,
		vulki.Workgroups{X: 4, Y: 1, Z: 1},
	); err != nil {
		return empty, fmt.Errorf("jabcode: dispatch GPU finder-average kernel: %w", err)
	}
	if err := recorder.Barrier(preparer.averagePartials); err != nil {
		return empty, fmt.Errorf("jabcode: synchronize GPU finder-average partials: %w", err)
	}
	if err := recorder.Download(preparer.averagePartials, 0, preparer.partialBytes[:]); err != nil {
		return empty, fmt.Errorf("jabcode: download GPU finder-average partials: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return empty, fmt.Errorf("jabcode: run GPU finder-average kernel: %w", err)
	}
	return decodeGPUFinderAverage(preparer.partialBytes[:]), nil
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

func decodeGPUFinderAverage(partials []byte) [3]float32 {
	var perFinder [4][3]float64
	for finder := range 4 {
		var sum [3]uint64
		var count uint64
		for lane := range 64 {
			offset := (finder*64 + lane) * 16
			for channel := range 3 {
				sum[channel] += uint64(binary.LittleEndian.Uint32(partials[offset+channel*4:]))
			}
			count += uint64(binary.LittleEndian.Uint32(partials[offset+12:]))
		}
		if count > 0 {
			for channel := range 3 {
				perFinder[finder][channel] = float64(sum[channel]) / float64(count)
			}
		}
	}
	var sum [3]float64
	var count [3]int
	for finder := range 4 {
		for channel := range 3 {
			if perFinder[finder][channel] > 0 {
				sum[channel] += perFinder[finder][channel]
				count[channel]++
			}
		}
	}
	var average [3]float32
	for channel := range 3 {
		if count[channel] > 0 {
			average[channel] = float32(sum[channel] / float64(count[channel]))
		}
	}
	return average
}

func (preparer *gpuFinderPassPreparer) estimatePitch() (int, int, error) {
	if preparer == nil || preparer.pitchBindings == nil || preparer.width <= 0 || preparer.height <= 0 {
		return 0, 0, fmt.Errorf("jabcode: GPU pitch sampler is closed")
	}
	minDim := min(preparer.width, preparer.height)
	if minDim < 4 {
		return 0, 0, nil
	}
	// The resident fold joins the per-hit chains under the deviceReplay
	// policy: bit-identical on the device, but its two extra submissions sit
	// on the descreen retry tier's critical path, so pooled route contexts
	// keep the fold on idle CPU cores (measured about 0.6 seconds wall on
	// the adverse dev capture once the pipeline cache made these kernels
	// instantly available).
	if preparer.resident.binarizer.deviceReplay && preparer.kernels.pitchLagKernelsReady() {
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

// pitchResidentACF samples the resident balanced canvas and reduces it to
// the two per-axis autocorrelation curves. The exact per-line sums come
// back to the host for the mean divisions (mant_div_small cannot divide by
// an arbitrary line length correctly rounded), then a second submission
// centers the samples and folds the lag products in softfloat64.
func (preparer *gpuFinderPassPreparer) pitchResidentACF(minDim int) (rows, columns []float64, maxLag int, err error) {
	if err := preparer.ensurePitchLag(); err != nil {
		return nil, nil, 0, err
	}
	width, height := preparer.width, preparer.height
	rowCount := min(pitchSampleLines, height)
	columnCount := min(pitchSampleLines, width)
	lineCount := rowCount + columnCount
	sampleCount := gpuPitchSampleCount(width, height)
	maxLag = max(2, minDim/8)
	var sampleParams [16]byte
	binary.LittleEndian.PutUint32(sampleParams[0:], uint32(width))
	binary.LittleEndian.PutUint32(sampleParams[4:], uint32(height))
	binary.LittleEndian.PutUint32(sampleParams[8:], uint32(rowCount))
	binary.LittleEndian.PutUint32(sampleParams[12:], uint32(columnCount))
	var lagParams [gpuPitchLagParamsSize]byte
	binary.LittleEndian.PutUint32(lagParams[0:], uint32(width))
	binary.LittleEndian.PutUint32(lagParams[4:], uint32(height))
	binary.LittleEndian.PutUint32(lagParams[8:], uint32(rowCount))
	binary.LittleEndian.PutUint32(lagParams[12:], uint32(columnCount))
	binary.LittleEndian.PutUint32(lagParams[16:], uint32(maxLag))
	putGPUFloat64(lagParams[20:], 1/float64(width))
	putGPUFloat64(lagParams[28:], 1/float64(height))

	sums := preparer.pitchLagLineBytes[:lineCount*8]
	recorder, err := preparer.device.NewRecorder()
	if err != nil {
		return nil, nil, 0, fmt.Errorf("jabcode: create GPU pitch-sum recorder: %w", err)
	}
	defer recorder.Abort()
	if err := recorder.Update(preparer.pitchParams, 0, sampleParams[:]); err != nil {
		return nil, nil, 0, fmt.Errorf("jabcode: update GPU pitch-sample parameters: %w", err)
	}
	if err := recorder.Update(preparer.pitchLagParams, 0, lagParams[:]); err != nil {
		return nil, nil, 0, fmt.Errorf("jabcode: update GPU pitch-lag parameters: %w", err)
	}
	sampleGroups := vulki.Workgroups{X: uint32((sampleCount + 63) / 64), Y: 1, Z: 1}
	if err := recorder.Dispatch(preparer.pitchKernel, preparer.pitchBindings, sampleGroups); err != nil {
		return nil, nil, 0, fmt.Errorf("jabcode: dispatch GPU pitch-sample kernel: %w", err)
	}
	if err := recorder.Barrier(preparer.pitchSamples); err != nil {
		return nil, nil, 0, fmt.Errorf("jabcode: synchronize GPU pitch samples: %w", err)
	}
	if err := recorder.Dispatch(
		preparer.pitchLagSumsKernel,
		preparer.pitchLagSumsBindings,
		vulki.Workgroups{X: 1, Y: 1, Z: 1},
	); err != nil {
		return nil, nil, 0, fmt.Errorf("jabcode: dispatch GPU pitch-sum kernel: %w", err)
	}
	if err := recorder.Barrier(preparer.pitchLagSums); err != nil {
		return nil, nil, 0, fmt.Errorf("jabcode: synchronize GPU pitch line sums: %w", err)
	}
	if err := recorder.Download(preparer.pitchLagSums, 0, sums); err != nil {
		return nil, nil, 0, fmt.Errorf("jabcode: download GPU pitch line sums: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return nil, nil, 0, fmt.Errorf("jabcode: run GPU pitch-sum submission: %w", err)
	}

	// Divide each line's exact sum by its length natively; the same slice
	// goes back up as the means the center kernel subtracts.
	for line := range lineCount {
		length := width
		if line >= rowCount {
			length = height
		}
		putGPUFloat64(sums[line*8:], getGPUFloat64(sums[line*8:])/float64(length))
	}
	acfBytes := preparer.pitchLagACFBytes[:2*(maxLag+1)*8]
	second, err := preparer.device.NewRecorder()
	if err != nil {
		return nil, nil, 0, fmt.Errorf("jabcode: create GPU pitch-lag recorder: %w", err)
	}
	defer second.Abort()
	if err := second.Update(preparer.pitchLagMeans, 0, sums); err != nil {
		return nil, nil, 0, fmt.Errorf("jabcode: upload GPU pitch means: %w", err)
	}
	if err := second.Dispatch(
		preparer.pitchLagCenterKernel,
		preparer.pitchLagCenterBindings,
		sampleGroups,
	); err != nil {
		return nil, nil, 0, fmt.Errorf("jabcode: dispatch GPU pitch-center kernel: %w", err)
	}
	if err := second.Barrier(preparer.pitchLagCentered); err != nil {
		return nil, nil, 0, fmt.Errorf("jabcode: synchronize GPU centered pitch samples: %w", err)
	}
	lagGroups := vulki.Workgroups{X: uint32((2*(maxLag+1) + 63) / 64), Y: 1, Z: 1}
	if err := second.Dispatch(preparer.pitchLagACFKernel, preparer.pitchLagACFBindings, lagGroups); err != nil {
		return nil, nil, 0, fmt.Errorf("jabcode: dispatch GPU pitch-lag kernel: %w", err)
	}
	if err := second.Barrier(preparer.pitchLagACF); err != nil {
		return nil, nil, 0, fmt.Errorf("jabcode: synchronize GPU pitch autocorrelation: %w", err)
	}
	if err := second.Download(preparer.pitchLagACF, 0, acfBytes); err != nil {
		return nil, nil, 0, fmt.Errorf("jabcode: download GPU pitch autocorrelation: %w", err)
	}
	if err := second.SubmitAndWait(); err != nil {
		return nil, nil, 0, fmt.Errorf("jabcode: run GPU pitch-lag submission: %w", err)
	}
	rows = make([]float64, maxLag+1)
	columns = make([]float64, maxLag+1)
	for lag := range rows {
		rows[lag] = getGPUFloat64(acfBytes[lag*8:])
		columns[lag] = getGPUFloat64(acfBytes[(maxLag+1+lag)*8:])
	}
	return rows, columns, maxLag, nil
}

// estimatePitchDownloaded is the fallback estimate: download every luma
// sample and fold the autocorrelation on the host.
func (preparer *gpuFinderPassPreparer) estimatePitchDownloaded(minDim int) (int, int, error) {
	rowCount := min(pitchSampleLines, preparer.height)
	columnCount := min(pitchSampleLines, preparer.width)
	sampleCount := gpuPitchSampleCount(preparer.width, preparer.height)
	if preparer.pitchBytes == nil {
		maxSamples := gpuPitchSampleCount(
			preparer.resident.binarizer.maxWidth,
			preparer.resident.binarizer.maxHeight,
		)
		preparer.pitchBytes = make([]byte, maxSamples*4)
	}
	samples := preparer.pitchBytes[:sampleCount*4]
	var params [16]byte
	binary.LittleEndian.PutUint32(params[0:], uint32(preparer.width))
	binary.LittleEndian.PutUint32(params[4:], uint32(preparer.height))
	binary.LittleEndian.PutUint32(params[8:], uint32(rowCount))
	binary.LittleEndian.PutUint32(params[12:], uint32(columnCount))
	recorder, err := preparer.device.NewRecorder()
	if err != nil {
		return 0, 0, fmt.Errorf("jabcode: create GPU pitch-sample recorder: %w", err)
	}
	defer recorder.Abort()
	if err := recorder.Update(preparer.pitchParams, 0, params[:]); err != nil {
		return 0, 0, fmt.Errorf("jabcode: update GPU pitch-sample parameters: %w", err)
	}
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

// putGPUFloat64 stores a float64 in the split high-word-first layout of the
// kernels' F64 struct.
func putGPUFloat64(b []byte, v float64) {
	bits := math.Float64bits(v)
	binary.LittleEndian.PutUint32(b[0:], uint32(bits>>32))
	binary.LittleEndian.PutUint32(b[4:], uint32(bits))
}

// getGPUFloat64 reads a float64 from the split high-word-first layout of
// the kernels' F64 struct.
func getGPUFloat64(b []byte) float64 {
	hi := binary.LittleEndian.Uint32(b[0:])
	lo := binary.LittleEndian.Uint32(b[4:])
	return math.Float64frombits(uint64(hi)<<32 | uint64(lo))
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
	if err := recorder.Update(preparer.descreenParams, 0, params[:]); err != nil {
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
	closeErrors := []error{preparer.closeDescreen(), preparer.closePitchLag()}
	if preparer.pitchBindings != nil {
		closeErrors = append(closeErrors, preparer.pitchBindings.Close())
		preparer.pitchBindings = nil
	}
	preparer.pitchKernel = nil
	if preparer.pitchSamples != nil {
		closeErrors = append(closeErrors, preparer.pitchSamples.Close())
		preparer.pitchSamples = nil
	}
	if preparer.pitchParams != nil {
		closeErrors = append(closeErrors, preparer.pitchParams.Close())
		preparer.pitchParams = nil
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
