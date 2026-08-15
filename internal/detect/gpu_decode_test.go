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
		// The device folds the per-lane partials itself, in f32, where the host
		// sums them in f64. That is a last-ulp difference on an intensity in
		// 0..255 and it cannot change a binarization except on an exact tie, so
		// the equality this used to assert was a property of both arms doing
		// the same arithmetic rather than a requirement on the value. The
		// capture census is what holds the threshold to its behaviour.
		const averageTolerance = 1.0 / 1024
		for channel := range gotAverage {
			if math.Abs(float64(gotAverage[channel]-wantAverage[channel])) > averageTolerance {
				t.Fatalf("GPU finder average = %v, want %v", gotAverage, wantAverage)
			}
		}
	}
	thresholds := averagePixelValue(wantBitmap, wantDetector.FPs)
	averagePass, err := ctx.preparer.prepareAverage(wantDetector.FPs, 0)
	if err != nil {
		t.Fatalf("prepare fused GPU finder-average retry: %v", err)
	}
	if err := averagePass.materialize(); err != nil {
		t.Fatalf("materialize fused GPU finder-average retry masks: %v", err)
	}
	assertGPUResidentMasksEqual(t, averagePass.channels, BinarizerRGB(wantBitmap, thresholds[:]))

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
	if err := kernels.compileFinderChains(); err != nil {
		t.Fatalf("compile GPU finder chain kernels: %v", err)
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
	// The device folds this in f32 and the host in float64. What the estimate
	// uses is which lag peaks, so the comparison bounds relative error and the
	// chosen lag is asserted exactly below.
	const acfTolerance = 1e-5
	close := func(got, want float64) bool {
		return math.Abs(got-want) <= acfTolerance*math.Max(math.Abs(want), 1)
	}
	for lag := 0; lag <= maxLag; lag++ {
		if !close(gotRows[lag], wantRows[lag]) {
			t.Fatalf("resident GPU row autocorrelation lag %d = %v, want %v",
				lag, gotRows[lag], wantRows[lag])
		}
		if !close(gotColumns[lag], wantColumns[lag]) {
			t.Fatalf("resident GPU column autocorrelation lag %d = %v, want %v",
				lag, gotColumns[lag], wantColumns[lag])
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
	// The resident retry schedule walks a fixed slot order and keeps admission on
	// the device, so the GPU ladder records a pass for every slot where the host
	// ladder records only the retries its own seed evidence admits. Parity is
	// therefore over the shared passes plus a resident tail that has to be the
	// bounded slot list with no finders in it.
	residentRetryLabels := []string{
		"resident descreen retry 1",
		"resident descreen retry 2",
		"resident print retry 1",
		"resident print retry 2",
	}
	shared := len(wantRetryDetector.Stats.Passes)
	if len(gotRetryDetector.Stats.Passes) != shared+len(residentRetryLabels) {
		t.Fatalf(
			"complete GPU finder passes = %d, want %d",
			len(gotRetryDetector.Stats.Passes),
			shared+len(residentRetryLabels),
		)
	}
	for index, want := range wantRetryDetector.Stats.Passes {
		if got := gotRetryDetector.Stats.Passes[index]; !reflect.DeepEqual(got, want) {
			t.Fatalf("complete GPU finder pass %d = %+v, want %+v", index, got, want)
		}
	}
	// A flat image produces nothing anywhere, so the host's own raw pass is what
	// an unproductive pass looks like.
	noFinders := wantRetryDetector.Stats.Passes[0].FinderFamilyPassStats
	for offset, label := range residentRetryLabels {
		got := gotRetryDetector.Stats.Passes[shared+offset]
		if got.Label != label {
			t.Fatalf("resident GPU retry pass %d = %q, want %q", offset, got.Label, label)
		}
		if !reflect.DeepEqual(got.FinderFamilyPassStats, noFinders) {
			t.Fatalf(
				"resident GPU retry pass %q = %+v, want %+v",
				label, got.FinderFamilyPassStats, noFinders,
			)
		}
	}
	if gotRetryDetector.Stats.RGBAvg != wantRetryDetector.Stats.RGBAvg {
		t.Fatalf(
			"complete GPU finder thresholds = %v, want %v",
			gotRetryDetector.Stats.RGBAvg, wantRetryDetector.Stats.RGBAvg,
		)
	}
	if gotRetryDetector.Stats.Consensus != wantRetryDetector.Stats.Consensus {
		t.Fatalf(
			"complete GPU finder consensus = %+v, want %+v",
			gotRetryDetector.Stats.Consensus, wantRetryDetector.Stats.Consensus,
		)
	}
}

func TestGPUMaskSnapshotDeferredExpansion(t *testing.T) {
	rendered, err := encode.Render(encode.Config{
		Colors:       8,
		ModuleSize:   12,
		SymbolNumber: 1,
	}, []byte("deferred mask snapshot"))
	if err != nil {
		t.Fatalf("encode deferred-snapshot symbol: %v", err)
	}
	base := core.BitmapFromImage(rendered.Image)
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	session, err := NewGPUDecodeSessionWithDevice(device, base, 2)
	if err != nil {
		_ = device.Close()
		t.Fatalf("new deferred-snapshot GPU decode session: %v", err)
	}
	// The assertion below requires the borrowed session's device replay path.
	// Make its background kernel warmup complete instead of letting a cold
	// driver cache select the bit-identical CPU replay for the first route.
	if err := session.workspace.kernels.compileFinderChains(); err != nil {
		_ = session.Close()
		_ = device.Close()
		t.Fatalf("compile deferred-snapshot finder chains: %v", err)
	}
	t.Cleanup(func() {
		if err := session.Close(); err != nil {
			t.Errorf("close deferred-snapshot GPU decode session: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close deferred-snapshot GPU decode device: %v", err)
		}
	})

	detector, found, release, err := session.LocateLevelFamilies(
		0, FinderFamilyCurrent.Mask(), IntensiveDetect, nil, nil,
	)
	if err != nil {
		t.Fatalf("locate deferred-snapshot route: %v", err)
	}
	if !found.Has(FinderFamilyCurrent) {
		t.Fatal("deferred-snapshot symbol was not detected")
	}
	defer release()
	for channel, ch := range detector.Ch {
		if ch == nil || ch.Pix != nil {
			t.Fatalf("located channel %d expanded eagerly; want deferred packed masks", channel)
		}
	}
	if got := detector.ChannelExpansionCount(); got != 0 {
		t.Fatalf("located detector expanded channels before a consumer: %d", got)
	}

	// A later route on the same session overwrites the context's shared
	// packed-mask host buffer; the located detector's snapshot must not care.
	_, _, overwritingRelease, err := session.LocateLevelFamilies(
		1, FinderFamilyCurrent.Mask(), IntensiveDetect, nil, nil,
	)
	if err != nil {
		t.Fatalf("locate overwriting route: %v", err)
	}
	overwritingRelease()
	// The retained lease is what keeps the resident pixels reachable, so the
	// deferred download must still succeed after that second route ran.
	if !detector.EnsureBalanced() {
		t.Fatalf("deferred balanced download failed under a retained lease: %v", detector.materializeErr)
	}
	if len(detector.BM.Pix) == 0 {
		t.Fatal("EnsureBalanced reported success without pixels")
	}
	channelWidth := detector.Ch[0].Width
	channelHeight := detector.Ch[0].Height
	probes := []int{0, channelWidth / 3, channelWidth * channelHeight / 2, channelWidth*channelHeight - 1}
	deferredPixels := make([]byte, len(probes))
	for i, pixel := range probes {
		deferredPixels[i] = detector.Ch[0].Pixel(pixel%channelWidth, pixel/channelWidth)
	}

	if !detector.EnsureChannels() {
		t.Fatal("deferred mask expansion failed after a later route")
	}
	if got := detector.ChannelExpansionCount(); got != 1 {
		t.Fatalf("channel expansion count = %d, want 1", got)
	}
	expanded := detector.Ch
	for channel, ch := range expanded {
		if ch == nil || len(ch.Pix) == 0 {
			t.Fatalf("channel %d has no pixels after deferred expansion", channel)
		}
	}
	for i, pixel := range probes {
		got := expanded[0].Pix[pixel]
		if got != deferredPixels[i] {
			t.Fatalf("deferred mask pixel %d (%d,%d) = %d, expanded = %d", pixel, pixel%channelWidth, pixel/channelWidth, deferredPixels[i], got)
		}
	}
	if !detector.EnsureChannels() {
		t.Fatal("repeated EnsureChannels failed")
	}
	if got := detector.ChannelExpansionCount(); got != 1 {
		t.Fatalf("repeated EnsureChannels changed expansion count to %d", got)
	}

	// A traced locate expands eagerly through the pass's own materializer;
	// its channels are the authoritative expansion the snapshot must match.
	var trace DetectorTrace
	tracedDetector, tracedFound, tracedRelease, err := session.LocateLevelFamilies(
		0, FinderFamilyCurrent.Mask(), IntensiveDetect, nil, &trace,
	)
	if err != nil {
		t.Fatalf("locate traced reference route: %v", err)
	}
	defer tracedRelease()
	if tracedFound != found {
		t.Fatalf("traced reference found %#x, deferred run found %#x", tracedFound, found)
	}
	for channel, ch := range tracedDetector.Ch {
		if ch == nil || len(ch.Pix) == 0 {
			t.Fatalf("traced reference channel %d was not expanded", channel)
		}
		if !bytes.Equal(expanded[channel].Pix, ch.Pix) {
			t.Fatalf("deferred channel %d differs from the traced eager expansion", channel)
		}
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

// TestGPUDecodeSessionRetireReturnsBeforeOperations pins the successful
// automatic-session handoff: the caller returns immediately, while workspace
// release still waits for every operation that entered before retirement.
func TestGPUDecodeSessionRetireReturnsBeforeOperations(t *testing.T) {
	released := make(chan struct{})
	session := &GPUDecodeSession{
		workspace:   &gpuDecodeWorkspace{contexts: newGPURouteContextPool(nil, nil, nil)},
		retireAsync: true,
		release: func() error {
			close(released)
			return nil
		},
	}
	if _, err := session.enter(); err != nil {
		t.Fatalf("enter open session: %v", err)
	}
	retired := make(chan error, 1)
	go func() { retired <- session.Retire() }()
	select {
	case err := <-retired:
		if err != nil {
			t.Fatalf("retire automatic session: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Retire waited for a registered operation")
	}
	select {
	case <-released:
		t.Fatal("Retire released the workspace with an operation in flight")
	default:
	}
	if _, err := session.enter(); err == nil {
		t.Fatal("a retiring session accepted a new operation")
	}
	session.leave()
	select {
	case <-released:
	case <-time.After(5 * time.Second):
		t.Fatal("retirement did not release after the last operation left")
	}
	if err := session.Close(); err != nil {
		t.Fatalf("join completed retirement: %v", err)
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
						if _, _, _, err := session.LocateLevelFamilies(
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
