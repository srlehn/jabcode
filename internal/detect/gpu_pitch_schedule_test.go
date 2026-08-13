//go:build !js

package detect

import (
	"encoding/binary"
	"math"
	"reflect"
	"testing"

	"github.com/srlehn/vulki"
)

func TestGPUPitchScheduleLayoutMatchesHost(t *testing.T) {
	chainParams := gpuFinderChainParams(1, 1, gpuFinderScanCapacity, false)
	constants := map[string]uint32{
		"PITCH_STAGE_VALLEY":     gpuPitchStageValley,
		"PITCH_STAGE_PEAK":       gpuPitchStagePeak,
		"PITCH_STAGE_LAG":        gpuPitchStageLag,
		"PITCH_STAGE_SCHEDULE":   gpuPitchStageSchedule,
		"PITCH_STAGE_PRINT":      gpuPitchStagePrint,
		"PITCH_STAGE_SELECT":     gpuPitchStageSelect,
		"PITCH_SAMPLE_LINES":     pitchSampleLines,
		"SEED_BUCKETS":           moduleSeedsBuckets,
		"PRINT_MIN_SEEDS":        printRetryMinSeeds,
		"PRINT_BLUR_LEAD_RADIUS": printBlurLeadRadius,
		"BLOCK_DIVISOR":          binThresholdDivisor,
		"BLOCK_MIN":              binMinBlock,
		"BLOCK_MAX":              binMaxBlock,
		"SCAN_CHANNELS":          1 << currentFamilySeekChannel,
		"BIN_SCAN_CAPACITY":      gpuBinarizerScanCapacityOffset / 4,
		"CHAIN_CLASSIFY_CURRENT": binary.LittleEndian.Uint32(chainParams[16:]),
		"CHAIN_CLASSIFY_BSI":     binary.LittleEndian.Uint32(chainParams[20:]),
		"CHAIN_CROSS_COLOR_BITS": binary.LittleEndian.Uint32(chainParams[24:]),
		"CHAIN_COLOR_SOURCE":     gpuFinderChainFlagColorSource,
		"CHAIN_ROW_STRIDE_SHIFT": gpuChainFlagStrideShift,
		"CHAIN_COMPACT_CAPACITY": gpuRowCompactCapacity,
		"CONTROL_ROW_VALLEY":     gpuPitchControlRowValley,
		"CONTROL_COLUMN_VALLEY":  gpuPitchControlColumnValley,
		"CONTROL_ROW_PEAK":       gpuPitchControlRowPeak,
		"CONTROL_COLUMN_PEAK":    gpuPitchControlColumnPeak,
		"CONTROL_ROW_LAG":        gpuPitchControlRowLag,
		"CONTROL_COLUMN_LAG":     gpuPitchControlColumnLag,
		"CONTROL_MEDIAN_BUCKET":  gpuPitchControlMedianBucket,
		"CONTROL_SELECTOR":       gpuPitchControlSelector,
		"CONTROL_STAGE":          gpuPitchControlStage,
		"RETRY_DESCREEN_FIRST":   finderRetryDescreenFirst,
		"RETRY_DESCREEN_SECOND":  finderRetryDescreenSecond,
		"RETRY_PRINT_FIRST":      finderRetryPrintFirst,
		"RETRY_PRINT_SECOND":     finderRetryPrintSecond,
		"RETRY_RECORD_WORDS":     gpuPitchRetryRecordWords,
		"RETRY_INDIRECT_CANVAS":  gpuPitchRetryIndirectCanvas / 4,
		"RETRY_INDIRECT_BLOCKS":  gpuPitchRetryIndirectBlocks / 4,
		"RETRY_INDIRECT_PACK":    gpuPitchRetryIndirectPack / 4,
		"RETRY_INDIRECT_SCAN":    gpuPitchRetryIndirectScan / 4,
		"RETRY_INDIRECT_CHAIN":   gpuPitchRetryIndirectChain / 4,
	}
	for name, want := range constants {
		if got := wgslUintConstant(t, pitchScheduleWGSL, name); got != want {
			t.Errorf("%s = %d, want %d", name, got, want)
		}
	}
	for name, source := range map[string]string{
		"samples": pitchSamplesWGSL,
		"sums":    pitchLineSumsWGSL,
		"center":  pitchCenterWGSL,
		"acf":     pitchACFWGSL,
	} {
		if got := wgslUintConstant(t, source, "PITCH_SAMPLE_LINES"); got != pitchSampleLines {
			t.Errorf("%s pitch sample lines = %d, want %d", name, got, pitchSampleLines)
		}
	}
	if gpuPitchScheduleControlWords != gpuPitchControlStage+1 {
		t.Fatalf("pitch schedule control has %d words, want %d",
			gpuPitchScheduleControlWords, gpuPitchControlStage+1)
	}
	if gpuPitchScheduleStages != gpuPitchStageSelect+1 {
		t.Fatalf("pitch schedule has %d stages, want %d",
			gpuPitchScheduleStages, gpuPitchStageSelect+1)
	}
	if maxFinderDescreenPasses != finderRetryDescreenSecond-finderRetryDescreenFirst+1 ||
		maxFinderPrintPasses != finderRetryPrintSecond-finderRetryPrintFirst+1 {
		t.Fatalf("retry schedule shape = (%d,%d), finder bounds = (%d,%d)",
			finderRetryDescreenSecond-finderRetryDescreenFirst+1,
			finderRetryPrintSecond-finderRetryPrintFirst+1,
			maxFinderDescreenPasses,
			maxFinderPrintPasses)
	}
	wantWords := (finderRetryPrintSecond+1)*gpuPitchRetryRecordWords +
		gpuPitchScheduleIndirects*3
	if gpuPitchScheduleWords != wantWords {
		t.Fatalf("pitch schedule has %d words, want %d",
			gpuPitchScheduleWords, wantWords)
	}
}

func TestGPUPitchScheduleReductionMatchesHost(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	kernels := newGPUDecodeKernels(device)
	t.Cleanup(func() {
		if err := kernels.Close(); err != nil {
			t.Errorf("close GPU kernel set: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close GPU device: %v", err)
		}
	})

	const width, height = 64, 64
	maxLag := max(2, min(width, height)/8)
	type ownedBuffer struct {
		name   string
		buffer *vulki.Buffer
	}
	var owned []ownedBuffer
	newBuffer := func(name string, size int) *vulki.Buffer {
		t.Helper()
		buffer, err := device.NewBuffer(uint64(size))
		if err != nil {
			t.Fatalf("allocate %s: %v", name, err)
		}
		owned = append(owned, ownedBuffer{name: name, buffer: buffer})
		return buffer
	}
	acf := newBuffer("autocorrelation", 2*(maxLag+1)*4)
	histogram := newBuffer("seed histogram", moduleSeedsBuckets*4)
	binarizer := newBuffer("binarizer parameters", gpuBinarizerParamsSize)
	control := newBuffer("pitch control", gpuPitchScheduleControlWords*4)
	schedule := newBuffer("retry schedule", gpuPitchScheduleWords*4)
	descreen := newBuffer("descreen parameters", gpuDescreenParamsSize)
	scan := newBuffer("scan parameters", gpuFinderScanParamsSize)
	chain := newBuffer("chain parameters", gpuFinderChainParamsSize)
	retryControl := newBuffer("retry control", gpuFinderRetryControlWords*4)
	passIndirect := newBuffer("pass indirect", gpuFinderDecisionIndirectWords*4)
	t.Cleanup(func() {
		for i := len(owned) - 1; i >= 0; i-- {
			if err := owned[i].buffer.Close(); err != nil {
				t.Errorf("close %s: %v", owned[i].name, err)
			}
		}
	})

	kernel, err := kernels.pitchSchedule()
	if err != nil {
		t.Fatalf("compile pitch schedule: %v", err)
	}
	bindings, err := kernel.NewBindings(
		vulki.BindBuffer(0, acf),
		vulki.BindBuffer(1, histogram),
		vulki.BindBuffer(2, binarizer),
		vulki.BindBuffer(3, control),
		vulki.BindBuffer(4, schedule),
		vulki.BindBuffer(5, descreen),
		vulki.BindBuffer(6, scan),
		vulki.BindBuffer(7, chain),
		vulki.BindBuffer(8, retryControl),
		vulki.BindBuffer(9, passIndirect),
	)
	if err != nil {
		t.Fatalf("bind pitch schedule: %v", err)
	}
	t.Cleanup(func() {
		if err := bindings.Close(); err != nil {
			t.Errorf("close pitch schedule: %v", err)
		}
	})

	acfValues := make([]float32, 2*(maxLag+1))
	copy(acfValues, []float32{10, 8, 6, 7, 9, 5, 4, 3, 2})
	copy(acfValues[maxLag+1:], []float32{10, 9, 8, 7, 6, 5, 4, 3, 2})
	acfBytes := make([]byte, len(acfValues)*4)
	for index, value := range acfValues {
		binary.LittleEndian.PutUint32(acfBytes[index*4:], math.Float32bits(value))
	}
	histogramBytes := make([]byte, moduleSeedsBuckets*4)
	binary.LittleEndian.PutUint32(histogramBytes[27*4:], 101)
	params := make([]byte, gpuBinarizerParamsSize)
	binary.LittleEndian.PutUint32(params[0:], width)
	binary.LittleEndian.PutUint32(params[4:], height)
	binary.LittleEndian.PutUint32(params[9*4:], 1)
	binary.LittleEndian.PutUint32(params[gpuBinarizerScanCapacityOffset:], gpuFinderScanCapacity)
	if err := acf.Upload(acfBytes); err != nil {
		t.Fatalf("upload autocorrelation: %v", err)
	}
	if err := histogram.Upload(histogramBytes); err != nil {
		t.Fatalf("upload seed histogram: %v", err)
	}
	if err := binarizer.Upload(params); err != nil {
		t.Fatalf("upload binarizer parameters: %v", err)
	}

	out := make([]byte, gpuPitchScheduleWords*4)
	clearedHistogram := make([]byte, moduleSeedsBuckets*4)
	recorder, err := device.NewRecorder()
	if err != nil {
		t.Fatalf("create pitch schedule recorder: %v", err)
	}
	defer recorder.Abort()
	if err := recorder.Fill(control, 0, gpuPitchScheduleControlWords*4, math.MaxUint32); err != nil {
		t.Fatalf("reset pitch control: %v", err)
	}
	if err := recorder.Fill(control, uint64(gpuPitchControlRowPeak*4), 2*4, 0); err != nil {
		t.Fatalf("reset pitch peaks: %v", err)
	}
	if err := recorder.Fill(schedule, 0, gpuPitchScheduleWords*4, 0); err != nil {
		t.Fatalf("reset retry schedule: %v", err)
	}
	if err := recorder.Barrier(control, schedule); err != nil {
		t.Fatalf("synchronize schedule reset: %v", err)
	}
	groups := vulki.Workgroups{X: 1, Y: 1, Z: 1}
	for stage := gpuPitchStageValley; stage <= gpuPitchStageSchedule; stage++ {
		if err := recorder.Fill(control, uint64(gpuPitchControlStage*4), 4, uint32(stage)); err != nil {
			t.Fatalf("select pitch schedule stage %d: %v", stage, err)
		}
		if err := recorder.Barrier(control); err != nil {
			t.Fatalf("synchronize pitch schedule stage %d selection: %v", stage, err)
		}
		if err := recorder.Dispatch(kernel, bindings, groups); err != nil {
			t.Fatalf("dispatch pitch schedule stage %d: %v", stage, err)
		}
		if err := recorder.Barrier(control, schedule, histogram); err != nil {
			t.Fatalf("synchronize pitch schedule stage %d: %v", stage, err)
		}
	}
	if err := recorder.Fill(
		control, uint64(gpuPitchControlStage*4), 4, gpuPitchStagePrint,
	); err != nil {
		t.Fatalf("select print schedule stage: %v", err)
	}
	if err := recorder.Barrier(control); err != nil {
		t.Fatalf("synchronize print schedule stage selection: %v", err)
	}
	if err := recorder.Dispatch(kernel, bindings, groups); err != nil {
		t.Fatalf("dispatch print schedule stage: %v", err)
	}
	if err := recorder.Barrier(control, schedule, histogram); err != nil {
		t.Fatalf("synchronize print schedule stage: %v", err)
	}
	if err := recorder.Download(schedule, 0, out); err != nil {
		t.Fatalf("download retry schedule: %v", err)
	}
	if err := recorder.Download(histogram, 0, clearedHistogram); err != nil {
		t.Fatalf("download cleared seed histogram: %v", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		t.Fatalf("run pitch schedule reduction: %v", err)
	}

	words := make([]uint32, gpuPitchScheduleRecords*gpuPitchRetryRecordWords)
	for index := range words {
		words[index] = binary.LittleEndian.Uint32(out[index*4:])
	}
	want := []uint32{
		1, 2, 0, 0,
		0, 0, 0, 0,
		1, 0, 0, 1,
		1, 2, 2, 1,
	}
	if !reflect.DeepEqual(words, want) {
		t.Fatalf("GPU retry schedule = %v, want %v", words, want)
	}
	if !reflect.DeepEqual(clearedHistogram, make([]byte, len(clearedHistogram))) {
		t.Fatal("GPU retry schedule did not consume the seed histogram")
	}
}
