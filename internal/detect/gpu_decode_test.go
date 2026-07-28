//go:build !js

package detect

import (
	"bytes"
	"errors"
	"math"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/encode"
)

func TestGPUDecodeWorkspaceInitialFinderParity(t *testing.T) {
	rendered, err := encode.Render(encode.Config{
		Colors:       8,
		ModuleSize:   12,
		SymbolNumber: 1,
	}, []byte("resident GPU decode finder parity"))
	if err != nil {
		t.Fatalf("encode finder parity symbol: %v", err)
	}
	base := core.BitmapFromImage(rendered.Image)
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	kernels := newGPUDecodeKernels(device)
	workspace, err := newGPUDecodeWorkspace(device, kernels, base.Width, base.Height, 1)
	if err != nil {
		_ = kernels.Close()
		_ = device.Close()
		t.Fatalf("new GPU decode workspace: %v", err)
	}
	workspace.ownsKernels = true
	t.Cleanup(func() {
		if err := workspace.Close(); err != nil {
			t.Errorf("close GPU decode workspace: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close GPU decode device: %v", err)
		}
	})
	if err := workspace.ladder.UploadAndBuild(base); err != nil {
		t.Fatalf("upload GPU decode workspace: %v", err)
	}
	ctx, err := workspace.contexts.acquire(base.Width, base.Height, nil)
	if err != nil {
		t.Fatalf("acquire GPU route context: %v", err)
	}
	defer workspace.contexts.release(ctx)

	wantBitmap := cloneGPUResidentBitmap(base)
	BalanceRGB(wantBitmap)
	wantDetector := &PrimaryDetector{
		BM: wantBitmap, Ch: BinarizerRGB(wantBitmap, nil), Mode: IntensiveDetect,
	}
	wantFound := wantDetector.LocateInitialFinderFamilies(FinderFamilyCurrent.Mask())
	gotDetector, err := ctx.bufferDetector(
		workspace.ladder.levels[0].buffer,
		base.Width,
		base.Height,
		IntensiveDetect,
		FinderFamilyCurrent.Mask(),
		nil,
		nil,
	)
	if err != nil {
		t.Fatalf("prepare initial GPU finder detector: %v", err)
	}
	gotFound := gotDetector.LocateInitialFinderFamilies(FinderFamilyCurrent.Mask())
	gotDetector, gotFound, err = finishGPUDetector(gotDetector, gotFound, nil)
	if err != nil {
		t.Fatalf("locate initial GPU finder pass: %v", err)
	}
	if gotFound != wantFound {
		t.Fatalf("GPU initial finder families = %#x, want %#x", gotFound, wantFound)
	}
	if !gotFound.Has(FinderFamilyCurrent) {
		t.Fatal("finder parity symbol was not detected")
	}
	if !bytes.Equal(gotDetector.BM.Pix, wantBitmap.Pix) {
		t.Fatal("materialized GPU balanced image differs from CPU output")
	}
	for index := range 4 {
		got, want := gotDetector.FPs[index], wantDetector.FPs[index]
		if got.Center != want.Center || got.ModuleSize != want.ModuleSize || got.Typ != want.Typ {
			t.Fatalf("GPU finder %d = %+v, want %+v", index, got, want)
		}
	}
	for _, fps := range [][]FinderPattern{
		wantDetector.FPs,
		func() []FinderPattern {
			copy := append([]FinderPattern(nil), wantDetector.FPs...)
			copy[2].FoundCount = 0
			return copy
		}(),
	} {
		gotAverage, err := ctx.preparer.averagePixelValue(fps)
		if err != nil {
			t.Fatalf("GPU finder average: %v", err)
		}
		wantAverage := averagePixelValue(wantBitmap, fps)
		if gotAverage != wantAverage {
			t.Fatalf("GPU finder average = %v, want %v", gotAverage, wantAverage)
		}
	}
	thresholds := averagePixelValue(wantBitmap, wantDetector.FPs)
	_, gotRetry, _, materializeRetry, err := ctx.preparer.prepare(0, 0, thresholds[:], false, 0)
	if err != nil {
		t.Fatalf("prepare GPU fixed-threshold retry: %v", err)
	}
	if err := materializeRetry(); err != nil {
		t.Fatalf("materialize GPU fixed-threshold retry masks: %v", err)
	}
	assertGPUResidentMasksEqual(t, gotRetry, BinarizerRGB(wantBitmap, thresholds[:]))
	_, gotPrint, _, materializePrint, err := ctx.preparer.prepare(0, 0, nil, true, 0)
	if err != nil {
		t.Fatalf("prepare GPU print retry: %v", err)
	}
	if err := materializePrint(); err != nil {
		t.Fatalf("materialize GPU print retry masks: %v", err)
	}
	assertGPUResidentMasksEqual(t, gotPrint, BinarizerRGBPrint(wantBitmap))
	gotPitchX, gotPitchY, err := ctx.preparer.estimatePitch()
	if err != nil {
		t.Fatalf("estimate GPU pitch: %v", err)
	}
	wantPitchX, wantPitchY := EstimatePitch(wantBitmap)
	if gotPitchX != wantPitchX || gotPitchY != wantPitchY {
		t.Fatalf(
			"GPU pitch = (%d,%d), want (%d,%d)",
			gotPitchX,
			gotPitchY,
			wantPitchX,
			wantPitchY,
		)
	}
	if err := kernels.compilePitchLag(); err != nil {
		t.Fatalf("compile GPU pitch-lag kernels: %v", err)
	}
	minDim := min(base.Width, base.Height)
	gotRows, gotColumns, gotMaxLag, err := ctx.preparer.pitchResidentACF(minDim)
	if err != nil {
		t.Fatalf("resident GPU pitch autocorrelation: %v", err)
	}
	maxLag := max(2, minDim/8)
	if gotMaxLag != maxLag {
		t.Fatalf("resident GPU pitch maxLag = %d, want %d", gotMaxLag, maxLag)
	}
	wantRows := acfAccumulate(sampleRows(wantBitmap), maxLag)
	wantColumns := acfAccumulate(sampleCols(wantBitmap), maxLag)
	for lag := 0; lag <= maxLag; lag++ {
		if math.Float64bits(gotRows[lag]) != math.Float64bits(wantRows[lag]) {
			t.Fatalf("resident GPU row autocorrelation lag %d = %x, want %x",
				lag, math.Float64bits(gotRows[lag]), math.Float64bits(wantRows[lag]))
		}
		if math.Float64bits(gotColumns[lag]) != math.Float64bits(wantColumns[lag]) {
			t.Fatalf("resident GPU column autocorrelation lag %d = %x, want %x",
				lag, math.Float64bits(gotColumns[lag]), math.Float64bits(wantColumns[lag]))
		}
	}
	residentPitchX, residentPitchY, err := ctx.preparer.estimatePitchResident(minDim)
	if err != nil {
		t.Fatalf("resident GPU pitch estimate: %v", err)
	}
	if residentPitchX != wantPitchX || residentPitchY != wantPitchY {
		t.Fatalf(
			"resident GPU pitch = (%d,%d), want (%d,%d)",
			residentPitchX,
			residentPitchY,
			wantPitchX,
			wantPitchY,
		)
	}
	ctx.preparer.trace = true
	gotFiltered, gotDescreen, _, materializeDescreen, err := ctx.preparer.prepare(2, 3, nil, false, 0)
	if err != nil {
		t.Fatalf("prepare GPU descreen retry: %v", err)
	}
	if err := materializeDescreen(); err != nil {
		t.Fatalf("materialize GPU descreen retry masks: %v", err)
	}
	wantFiltered := descreen(wantBitmap, 2, 3, nil)
	differing, maxDelta := gpuCanvasDifference(gotFiltered, wantFiltered)
	t.Logf("GPU descreen has %d differing components, maximum delta %d", differing, maxDelta)
	if maxDelta > 1 {
		t.Fatalf("GPU descreen maximum component delta = %d, want at most 1", maxDelta)
	}
	assertGPUResidentMasksEqual(t, gotDescreen, BinarizerRGB(wantFiltered, nil))

	// A flat image walks the complete no-finder ladder. It verifies that the GPU
	// preparation stages preserve one shared detector state instead of starting
	// a second finder traversal after the raw pass.
	flat := core.NewBitmap(base.Width, base.Height, 4)
	for pixel := 0; pixel < flat.Width*flat.Height; pixel++ {
		flat.Pix[pixel*4+0] = 127
		flat.Pix[pixel*4+1] = 127
		flat.Pix[pixel*4+2] = 127
		flat.Pix[pixel*4+3] = 255
	}
	if err := workspace.ladder.UploadAndBuild(flat); err != nil {
		t.Fatalf("upload flat GPU decode workspace: %v", err)
	}
	wantFlat := cloneGPUResidentBitmap(flat)
	BalanceRGB(wantFlat)
	wantRetryDetector := &PrimaryDetector{
		BM: wantFlat, Ch: BinarizerRGB(wantFlat, nil), Mode: IntensiveDetect,
	}
	wantRetryFound := wantRetryDetector.LocateFinderFamilies(FinderFamilyCurrent.Mask())
	gotRetryDetector, err := ctx.bufferDetector(
		workspace.ladder.levels[0].buffer,
		flat.Width,
		flat.Height,
		IntensiveDetect,
		FinderFamilyCurrent.Mask(),
		nil,
		nil,
	)
	if err != nil {
		t.Fatalf("prepare complete GPU finder detector: %v", err)
	}
	gotRetryFound, err := gotRetryDetector.locateFinderFamilies(FinderFamilyCurrent.Mask(), ctx.preparer)
	if err != nil {
		t.Fatalf("run complete GPU finder ladder: %v", err)
	}
	gotRetryDetector, gotRetryFound, err = finishGPUDetector(gotRetryDetector, gotRetryFound, nil)
	if err != nil {
		t.Fatalf("locate complete GPU finder ladder: %v", err)
	}
	if gotRetryFound != wantRetryFound {
		t.Fatalf("complete GPU finder families = %#x, want %#x", gotRetryFound, wantRetryFound)
	}
	if !reflect.DeepEqual(gotRetryDetector.Stats, wantRetryDetector.Stats) {
		t.Fatalf("complete GPU finder stats = %+v, want %+v", gotRetryDetector.Stats, wantRetryDetector.Stats)
	}
}

// TestGPUDecodeSessionCloseWaitsForOperations pins the session operation
// gate: Close must wait for a method that has passed entry but not yet
// acquired a route context, and afterwards the session rejects new entries.
func TestGPUDecodeSessionCloseWaitsForOperations(t *testing.T) {
	session := &GPUDecodeSession{
		workspace: &gpuDecodeWorkspace{contexts: newGPURouteContextPool(nil, nil, nil)},
	}
	if _, err := session.enter(); err != nil {
		t.Fatalf("enter open session: %v", err)
	}
	closed := make(chan error, 1)
	go func() { closed <- session.Close() }()
	select {
	case err := <-closed:
		t.Fatalf("Close returned (%v) with a registered operation in flight", err)
	case <-time.After(50 * time.Millisecond):
	}
	if _, err := session.enter(); err == nil {
		t.Fatal("a closing session accepted a new operation")
	}
	session.leave()
	select {
	case err := <-closed:
		if err != nil {
			t.Fatalf("close quiesced session: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Close did not return after the last operation left")
	}
}

// TestGPUDecodeSessionConcurrentCloseRace exercises Close concurrently with
// session methods on a real device so the race detector can see a straddling
// operation touch a released workspace. The operation gate makes every
// interleaving either a completed operation or a closed-session error.
func TestGPUDecodeSessionConcurrentCloseRace(t *testing.T) {
	rendered, err := encode.Render(encode.Config{
		Colors:       8,
		ModuleSize:   12,
		SymbolNumber: 1,
	}, []byte("session close race"))
	if err != nil {
		t.Fatalf("encode close-race symbol: %v", err)
	}
	base := core.BitmapFromImage(rendered.Image)
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	defer func() {
		if err := device.Close(); err != nil {
			t.Errorf("close close-race device: %v", err)
		}
	}()
	for round := range 4 {
		session, err := NewGPUDecodeSessionWithDevice(device, base, 2)
		if err != nil {
			t.Fatalf("new close-race session %d: %v", round, err)
		}
		var workers sync.WaitGroup
		for worker := range 2 {
			workers.Add(1)
			go func() {
				defer workers.Done()
				for {
					switch worker % 2 {
					case 0:
						if _, err := session.DownloadLevel(0); err != nil {
							return
						}
					default:
						if _, _, err := session.LocateLevelFamilies(
							0, FinderFamilyCurrent.Mask(), IntensiveDetect, nil, nil,
						); err != nil {
							return
						}
					}
				}
			}()
		}
		if err := session.Close(); err != nil {
			t.Fatalf("close close-race session %d: %v", round, err)
		}
		workers.Wait()
	}
}

func TestGPUDecodeRuntimeUnavailableFallsBack(t *testing.T) {
	openCalls := 0
	wantErr := errors.New("Vulkan unavailable")
	runtime := newGPUDecodeRuntime(newGPUDeviceCache(func() (*vulki.Device, error) {
		openCalls++
		return nil, wantErr
	}))
	base := &core.Bitmap{Width: 1024, Height: 1024, Channels: 4}
	for range 2 {
		session, err := runtime.begin(base, 1)
		if err != nil {
			t.Fatalf("automatic GPU fallback: %v", err)
		}
		if session != nil {
			t.Fatal("unavailable automatic GPU returned a session")
		}
	}
	if openCalls != 1 {
		t.Fatalf("unavailable automatic GPU opened Vulkan %d times, want once", openCalls)
	}
}
