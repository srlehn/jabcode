package detect

import (
	"bytes"
	"errors"
	"image"
	"reflect"
	"sync"
	"testing"

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
	_, gotRetry, err := ctx.preparer.prepare(0, 0, thresholds[:], false)
	if err != nil {
		t.Fatalf("prepare GPU fixed-threshold retry: %v", err)
	}
	assertGPUResidentMasksEqual(t, gotRetry, BinarizerRGB(wantBitmap, thresholds[:]))
	_, gotPrint, err := ctx.preparer.prepare(0, 0, nil, true)
	if err != nil {
		t.Fatalf("prepare GPU print retry: %v", err)
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
	ctx.preparer.trace = true
	gotFiltered, gotDescreen, err := ctx.preparer.prepare(2, 3, nil, false)
	if err != nil {
		t.Fatalf("prepare GPU descreen retry: %v", err)
	}
	wantFiltered := descreen(wantBitmap, 2, 3)
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

// TestGPUDecodeSessionConcurrentRouteParity runs the same mixed route set
// sequentially and concurrently on one session and requires identical finder
// families, canvas sizes and materialized pixels. It exercises context reuse
// across canvas sizes, route-buffer growth (the 30-degree whole-frame canvas
// exceeds the base dimensions) and the pool's exclusivity under -race.
func TestGPUDecodeSessionConcurrentRouteParity(t *testing.T) {
	rendered, err := encode.Render(encode.Config{
		Colors:       8,
		ModuleSize:   12,
		SymbolNumber: 1,
	}, []byte("concurrent GPU route parity"))
	if err != nil {
		t.Fatalf("encode concurrent route parity symbol: %v", err)
	}
	base := RotateToBitmap(rendered.Image, -30)
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	session, err := NewGPUDecodeSessionWithDevice(device, base, 2)
	if err != nil {
		_ = device.Close()
		t.Fatalf("new concurrent-route GPU decode session: %v", err)
	}
	t.Cleanup(func() {
		if err := session.Close(); err != nil {
			t.Errorf("close concurrent-route GPU decode session: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close concurrent-route GPU decode device: %v", err)
		}
	})

	fullFrame := image.Rect(0, 0, base.Width, base.Height)
	halfFrame := image.Rect(
		0, 0,
		max(base.Width/2, 1), max(base.Height/2, 1),
	)
	routes := []struct {
		level int
		crop  image.Rectangle
		angle float64
	}{
		{level: 0, crop: fullFrame, angle: 30},
		{level: 0, crop: fullFrame, angle: 120},
		{level: 0, crop: halfFrame, angle: 45},
		{level: 0, crop: fullFrame, angle: 30},
		{level: 1, crop: image.Rect(0, 0, (base.Width+1)/2, (base.Height+1)/2), angle: 30},
		{level: 0, crop: fullFrame, angle: 210},
		{level: 0, crop: halfFrame, angle: 300},
		{level: 0, crop: fullFrame, angle: 30},
	}
	type routeResult struct {
		found FinderFamilySet
		size  image.Point
		pix   []byte
	}
	runRoute := func(route struct {
		level int
		crop  image.Rectangle
		angle float64
	}) (routeResult, error) {
		detector, found, size, err := session.LocateRouteFamilies(
			route.level,
			route.crop,
			route.angle,
			FinderFamilyCurrent.Mask(),
			IntensiveDetect,
			nil,
			nil,
		)
		if err != nil {
			return routeResult{}, err
		}
		result := routeResult{found: found, size: size}
		if found != 0 {
			result.pix = append([]byte(nil), detector.BM.Pix...)
		}
		return result, nil
	}

	want := make([]routeResult, len(routes))
	for index, route := range routes {
		want[index], err = runRoute(route)
		if err != nil {
			t.Fatalf("sequential route %d: %v", index, err)
		}
	}
	if !want[0].found.Has(FinderFamilyCurrent) {
		t.Fatal("counter-rotated parity symbol was not detected")
	}

	got := make([]routeResult, len(routes))
	routeErrs := make([]error, len(routes))
	var routesGroup sync.WaitGroup
	for index, route := range routes {
		routesGroup.Add(1)
		go func() {
			defer routesGroup.Done()
			got[index], routeErrs[index] = runRoute(route)
		}()
	}
	routesGroup.Wait()
	for index := range routes {
		if routeErrs[index] != nil {
			t.Fatalf("concurrent route %d: %v", index, routeErrs[index])
		}
		if got[index].found != want[index].found || got[index].size != want[index].size {
			t.Fatalf(
				"concurrent route %d = %#x %v, sequential = %#x %v",
				index, got[index].found, got[index].size, want[index].found, want[index].size,
			)
		}
		if !bytes.Equal(got[index].pix, want[index].pix) {
			t.Fatalf("concurrent route %d materialized pixels differ from sequential run", index)
		}
	}

	// A quit hook that already fired must abort acquisition without touching
	// the device.
	if _, err := session.workspace.contexts.acquire(64, 64, func() bool { return true }); !errors.Is(err, errGPURouteAborted) {
		t.Fatalf("quit-cancelled acquisition error = %v, want errGPURouteAborted", err)
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
