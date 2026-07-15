package detect

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
)

//go:embed shaders/finder_average.wgsl
var finderAverageWGSL string

//go:embed shaders/pitch_samples.wgsl
var pitchSamplesWGSL string

//go:embed shaders/descreen_horizontal.wgsl
var descreenHorizontalWGSL string

//go:embed shaders/descreen_vertical.wgsl
var descreenVerticalWGSL string

const (
	gpuFinderAverageParamsSize  = 18 * 4
	gpuFinderAveragePartialSize = 4 * 64 * 4 * 4
)

type gpuFinderPassPreparer struct {
	device   *vulki.Device
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
	pitchBytes    []byte

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
	resident *gpuResidentBinarizer,
) (*gpuFinderPassPreparer, error) {
	if device == nil || device.Closed() || resident == nil {
		return nil, fmt.Errorf("jabcode: GPU finder preparer needs an open resident device")
	}
	preparer := &gpuFinderPassPreparer{device: device, resident: resident}
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
	preparer.averageKernel, err = device.NewKernel(vulki.KernelOptions{
		WGSL: finderAverageWGSL,
		Bindings: []vulki.BindingLayout{
			{Binding: 0, Access: vulki.BufferReadOnly},
			{Binding: 1, Access: vulki.BufferReadWrite},
			{Binding: 2, Access: vulki.BufferReadOnly},
		},
	})
	if err != nil {
		_ = preparer.Close()
		return nil, fmt.Errorf("jabcode: create GPU finder-average kernel: %w", err)
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
	preparer.pitchParams, err = device.NewBuffer(4 * 4)
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
	preparer.pitchBytes = make([]byte, maxSamples*4)
	preparer.pitchKernel, err = device.NewKernel(vulki.KernelOptions{
		WGSL: pitchSamplesWGSL,
		Bindings: []vulki.BindingLayout{
			{Binding: 0, Access: vulki.BufferReadOnly},
			{Binding: 1, Access: vulki.BufferReadWrite},
			{Binding: 2, Access: vulki.BufferReadOnly},
		},
	})
	if err != nil {
		_ = preparer.Close()
		return nil, fmt.Errorf("jabcode: create GPU pitch-sample kernel: %w", err)
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
	area := uint64(resident.binarizer.maxWidth) * uint64(resident.binarizer.maxHeight)
	preparer.descreenParams, err = device.NewBuffer(4 * 4)
	if err != nil {
		_ = preparer.Close()
		return nil, fmt.Errorf("jabcode: allocate GPU descreen parameters: %w", err)
	}
	preparer.descreenLinear, err = device.NewBuffer(area * 16)
	if err != nil {
		_ = preparer.Close()
		return nil, fmt.Errorf("jabcode: allocate GPU descreen linear image: %w", err)
	}
	preparer.descreenFiltered, err = device.NewBuffer(area * 4)
	if err != nil {
		_ = preparer.Close()
		return nil, fmt.Errorf("jabcode: allocate GPU descreen output: %w", err)
	}
	preparer.descreenHorizontal, err = device.NewKernel(vulki.KernelOptions{
		WGSL: descreenHorizontalWGSL,
		Bindings: []vulki.BindingLayout{
			{Binding: 0, Access: vulki.BufferReadOnly},
			{Binding: 1, Access: vulki.BufferReadWrite},
			{Binding: 2, Access: vulki.BufferReadOnly},
		},
	})
	if err != nil {
		_ = preparer.Close()
		return nil, fmt.Errorf("jabcode: create GPU horizontal descreen kernel: %w", err)
	}
	preparer.descreenVertical, err = device.NewKernel(vulki.KernelOptions{
		WGSL: descreenVerticalWGSL,
		Bindings: []vulki.BindingLayout{
			{Binding: 0, Access: vulki.BufferReadOnly},
			{Binding: 1, Access: vulki.BufferReadWrite},
			{Binding: 2, Access: vulki.BufferReadOnly},
			{Binding: 3, Access: vulki.BufferReadOnly},
		},
	})
	if err != nil {
		_ = preparer.Close()
		return nil, fmt.Errorf("jabcode: create GPU vertical descreen kernel: %w", err)
	}
	preparer.descreenHBindings, err = preparer.descreenHorizontal.NewBindings(
		vulki.BindBuffer(0, resident.balanced),
		vulki.BindBuffer(1, preparer.descreenLinear),
		vulki.BindBuffer(2, preparer.descreenParams),
	)
	if err != nil {
		_ = preparer.Close()
		return nil, fmt.Errorf("jabcode: bind GPU horizontal descreen kernel: %w", err)
	}
	preparer.descreenVBindings, err = preparer.descreenVertical.NewBindings(
		vulki.BindBuffer(0, preparer.descreenLinear),
		vulki.BindBuffer(1, preparer.descreenFiltered),
		vulki.BindBuffer(2, resident.balanced),
		vulki.BindBuffer(3, preparer.descreenParams),
	)
	if err != nil {
		_ = preparer.Close()
		return nil, fmt.Errorf("jabcode: bind GPU vertical descreen kernel: %w", err)
	}
	return preparer, nil
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
	rowCount := min(pitchSampleLines, preparer.height)
	columnCount := min(pitchSampleLines, preparer.width)
	sampleCount := gpuPitchSampleCount(preparer.width, preparer.height)
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
) (*core.Bitmap, [3]*core.Bitmap, error) {
	input := preparer.resident.balanced
	if rx > 0 || ry > 0 {
		if err := preparer.descreen(rx, ry); err != nil {
			return nil, [3]*core.Bitmap{}, err
		}
		input = preparer.descreenFiltered
	}
	channels, err := preparer.resident.BinarizePrepared(
		input,
		preparer.width,
		preparer.height,
		thresholds,
		printLevels,
	)
	if err != nil {
		return nil, [3]*core.Bitmap{}, err
	}
	if !preparer.trace {
		return nil, channels, nil
	}
	inputBitmap, err := preparer.resident.DownloadPrepared(input, preparer.width, preparer.height)
	if err != nil {
		return nil, [3]*core.Bitmap{}, err
	}
	return inputBitmap, channels, nil
}

func (preparer *gpuFinderPassPreparer) descreen(rx, ry int) error {
	if preparer == nil || preparer.descreenHBindings == nil || preparer.descreenVBindings == nil {
		return fmt.Errorf("jabcode: GPU descreen preparer is closed")
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

func (preparer *gpuFinderPassPreparer) Close() error {
	if preparer == nil {
		return nil
	}
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
	for _, kernel := range []*vulki.Kernel{
		preparer.descreenVertical,
		preparer.descreenHorizontal,
	} {
		if kernel != nil {
			closeErrors = append(closeErrors, kernel.Close())
		}
	}
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
	if preparer.pitchBindings != nil {
		closeErrors = append(closeErrors, preparer.pitchBindings.Close())
		preparer.pitchBindings = nil
	}
	if preparer.pitchKernel != nil {
		closeErrors = append(closeErrors, preparer.pitchKernel.Close())
		preparer.pitchKernel = nil
	}
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
	if preparer.averageKernel != nil {
		closeErrors = append(closeErrors, preparer.averageKernel.Close())
		preparer.averageKernel = nil
	}
	if preparer.averagePartials != nil {
		closeErrors = append(closeErrors, preparer.averagePartials.Close())
		preparer.averagePartials = nil
	}
	if preparer.averageParams != nil {
		closeErrors = append(closeErrors, preparer.averageParams.Close())
		preparer.averageParams = nil
	}
	return errors.Join(closeErrors...)
}
