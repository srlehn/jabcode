//go:build !js

package detect

import (
	"errors"
	"fmt"
	"image"
	"reflect"
	"testing"
	"unsafe"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
)

// TestGPURouteContextPoolNonOOMFailureFailsFast pins the sentinel
// classification: a creation failure that is not device-memory exhaustion (a
// lost device, a programming error) surfaces to the caller immediately so the
// route takes its CPU fallback, and it neither retires healthy idle contexts
// nor latches the pool as exhausted.
func TestGPURouteContextPoolNonOOMFailureFailsFast(t *testing.T) {
	pool := newGPURouteContextPool(nil, nil, nil)
	idle := &gpuRouteContext{capWidth: 256, capHeight: 256}
	pool.free = append(pool.free, idle)
	pool.live = append(pool.live, idle)
	lost := fmt.Errorf("create context: %w", vulki.ErrDeviceLost)
	calls := 0
	pool.create = func(capWidth, capHeight int) (*gpuRouteContext, error) {
		calls++
		return nil, lost
	}
	ctx, err := pool.acquire(512, 512, nil)
	if ctx != nil || !errors.Is(err, vulki.ErrDeviceLost) {
		t.Fatalf("acquire returned (%v, %v), want the device-lost error", ctx, err)
	}
	if calls != 1 {
		t.Fatalf("creation ran %d times, want one attempt without an evict-retry", calls)
	}
	if pool.exhausted {
		t.Fatal("a non-OOM creation failure latched the pool as exhausted")
	}
	if len(pool.free) != 1 || pool.free[0] != idle || len(pool.live) != 1 {
		t.Fatalf("healthy idle context was disturbed: free=%d live=%d", len(pool.free), len(pool.live))
	}
}

// TestGPURouteContextPoolOOMEvictsRetriesAndLatches pins the memory-pressure
// path: an out-of-device-memory creation retires the idle contexts, retries
// once, and a second failure latches the pool as exhausted; with no live
// context able to hold the request the acquisition surfaces an error and the
// route takes its CPU fallback.
func TestGPURouteContextPoolOOMEvictsRetriesAndLatches(t *testing.T) {
	pool := newGPURouteContextPool(nil, nil, nil)
	idle := &gpuRouteContext{capWidth: 256, capHeight: 256}
	pool.free = append(pool.free, idle)
	pool.live = append(pool.live, idle)
	oom := fmt.Errorf("allocate route canvas: %w", vulki.ErrOutOfDeviceMemory)
	calls := 0
	pool.create = func(capWidth, capHeight int) (*gpuRouteContext, error) {
		calls++
		return nil, oom
	}
	ctx, err := pool.acquire(512, 512, nil)
	if ctx != nil || err == nil {
		t.Fatalf("acquire returned (%v, %v), want an error after exhaustion", ctx, err)
	}
	if calls != 2 {
		t.Fatalf("creation ran %d times, want the evict-and-retry pair", calls)
	}
	if !pool.exhausted {
		t.Fatal("repeated OOM creation did not latch the pool as exhausted")
	}
	if len(pool.free) != 0 || len(pool.live) != 0 {
		t.Fatalf("idle context was not retired: free=%d live=%d", len(pool.free), len(pool.live))
	}
}

// TestGPURouteContextPoolBudgetAdmission pins the deterministic admission
// gate: a request whose worst-case context exceeds the pool budget fails
// immediately to its CPU route without touching the driver, and an admitted
// request under a full budget retires idle contexts smallest-first instead of
// falling back or sweeping every idle.
func TestGPURouteContextPoolBudgetAdmission(t *testing.T) {
	pool := newGPURouteContextPool(nil, nil, nil)
	pool.budgetKnown = true
	pool.budget = gpuRouteContextDeviceBytes(512, 512)
	calls := 0
	pool.create = func(capWidth, capHeight int) (*gpuRouteContext, error) {
		calls++
		return &gpuRouteContext{capWidth: capWidth, capHeight: capHeight}, nil
	}

	if ctx, err := pool.acquire(768, 768, nil); ctx != nil || err == nil {
		t.Fatalf("oversized request returned (%v, %v), want a deterministic refusal", ctx, err)
	}
	if calls != 0 {
		t.Fatal("an unadmitted request probed the driver")
	}

	// Fill the budget with two idles: a small one and a mid one.
	small := &gpuRouteContext{capWidth: 256, capHeight: 256,
		deviceBytes: gpuRouteContextDeviceBytes(256, 256)}
	mid := &gpuRouteContext{capWidth: 256, capHeight: 512,
		deviceBytes: gpuRouteContextDeviceBytes(256, 512)}
	pool.free = append(pool.free, mid, small)
	pool.live = append(pool.live, mid, small)
	pool.planned = small.deviceBytes + mid.deviceBytes

	ctx, err := pool.acquire(512, 512, nil)
	if err != nil || ctx == nil || ctx.capWidth != 512 || ctx.capHeight != 512 {
		t.Fatalf("admitted request returned (%v, %v), want a created 512x512 context", ctx, err)
	}
	if calls != 1 {
		t.Fatalf("creation ran %d times, want once after retirement", calls)
	}
	if pool.planned != ctx.deviceBytes {
		t.Fatalf("planned bytes = %d, want only the new context's %d", pool.planned, ctx.deviceBytes)
	}
	if len(pool.free) != 0 || len(pool.live) != 1 {
		t.Fatalf("idle retirement left free=%d live=%d, want the new context alone", len(pool.free), len(pool.live))
	}
}

// TestGPURouteContextPoolBudgetRetiresSmallestFirst pins the selective half:
// when retiring one small idle already makes room, the larger idle survives.
func TestGPURouteContextPoolBudgetRetiresSmallestFirst(t *testing.T) {
	pool := newGPURouteContextPool(nil, nil, nil)
	small := &gpuRouteContext{capWidth: 256, capHeight: 256,
		deviceBytes: gpuRouteContextDeviceBytes(256, 256)}
	mid := &gpuRouteContext{capWidth: 512, capHeight: 512,
		deviceBytes: gpuRouteContextDeviceBytes(512, 512)}
	pool.budgetKnown = true
	// One byte more than the surviving idle plus the new request: retiring
	// the small idle makes exactly enough room, so the mid idle must stay.
	pool.budget = mid.deviceBytes + gpuRouteContextDeviceBytes(512, 768) + 1
	pool.free = append(pool.free, mid, small)
	pool.live = append(pool.live, mid, small)
	pool.planned = small.deviceBytes + mid.deviceBytes
	pool.create = func(capWidth, capHeight int) (*gpuRouteContext, error) {
		return &gpuRouteContext{capWidth: capWidth, capHeight: capHeight}, nil
	}

	// Needs more than the remaining headroom but less than headroom plus the
	// small idle; only the small idle must go.
	ctx, err := pool.acquire(512, 768, nil)
	if err != nil || ctx == nil {
		t.Fatalf("acquire returned (%v, %v), want a created context", ctx, err)
	}
	if len(pool.free) != 1 || pool.free[0] != mid {
		t.Fatalf("free list = %d entries, want the larger idle kept", len(pool.free))
	}
}

// TestGPURouteContextPoolOOMRetryRecovers pins the recovery half of the
// memory-pressure path: when retiring the idle contexts frees enough device
// memory, the retried creation serves the request and the pool stays open.
func TestGPURouteContextPoolOOMRetryRecovers(t *testing.T) {
	pool := newGPURouteContextPool(nil, nil, nil)
	idle := &gpuRouteContext{capWidth: 256, capHeight: 256}
	pool.free = append(pool.free, idle)
	pool.live = append(pool.live, idle)
	grown := &gpuRouteContext{capWidth: 512, capHeight: 512}
	calls := 0
	pool.create = func(capWidth, capHeight int) (*gpuRouteContext, error) {
		calls++
		if calls == 1 {
			return nil, fmt.Errorf("allocate route canvas: %w", vulki.ErrOutOfDeviceMemory)
		}
		return grown, nil
	}
	ctx, err := pool.acquire(512, 512, nil)
	if ctx != grown || err != nil {
		t.Fatalf("acquire returned (%v, %v), want the retried context", ctx, err)
	}
	if pool.exhausted {
		t.Fatal("a recovered creation left the pool latched as exhausted")
	}
	if len(pool.live) != 1 || pool.live[0] != grown || len(pool.free) != 0 {
		t.Fatalf("pool bookkeeping wrong after recovery: free=%d live=%d", len(pool.free), len(pool.live))
	}
	pool.release(grown)
	if len(pool.free) != 1 || pool.free[0] != grown {
		t.Fatal("released context did not return to the free list")
	}
}

// TestGPURouteContextPoolHostBudgetFailsFastToCPU pins the anti-stall rule:
// with the host-scratch budget spent and every context leased, no retirement
// can free scratch, so the route takes its CPU fallback immediately rather
// than parking on a lease.
func TestGPURouteContextPoolHostBudgetFailsFastToCPU(t *testing.T) {
	pool := newGPURouteContextPool(nil, nil, nil)
	const small = 256
	pool.hostBudget = 8 * gpuRouteContextHostBytes(small, small)
	for range 8 {
		ctx := &gpuRouteContext{
			capWidth: small, capHeight: small,
			hostBytes: gpuRouteContextHostBytes(small, small),
		}
		pool.live = append(pool.live, ctx)
		pool.hostPlanned += ctx.hostBytes
	}
	pool.outstanding = len(pool.live)
	calls := 0
	pool.create = func(capWidth, capHeight int) (*gpuRouteContext, error) {
		calls++
		return &gpuRouteContext{capWidth: capWidth, capHeight: capHeight}, nil
	}
	ctx, err := pool.acquire(512, 512, nil)
	if ctx != nil || err == nil {
		t.Fatalf("acquire returned (%v, %v), want an immediate CPU-fallback error", ctx, err)
	}
	if calls != 0 {
		t.Fatalf("creation ran %d times, want none over the host budget", calls)
	}
	if len(pool.live) != 8 {
		t.Fatalf("live list has %d entries, want them untouched", len(pool.live))
	}
	if len(pool.free) != 0 {
		t.Fatalf("free list has %d entries, want none", len(pool.free))
	}
}

// TestGPURouteContextPoolHostBudgetRecyclesIdleScratch pins the fix for the
// route ladder's worst backend loss: a pool whose host budget is spent on
// small idle contexts must retire enough of them to build the large canvas a
// full-resolution rotation needs, instead of sending that route to the CPU
// while the adapter idles.
func TestGPURouteContextPoolHostBudgetRecyclesIdleScratch(t *testing.T) {
	pool := newGPURouteContextPool(nil, nil, nil)
	const small = 256
	smallBytes := gpuRouteContextHostBytes(small, small)
	pool.hostBudget = 8 * smallBytes
	for range 8 {
		ctx := &gpuRouteContext{capWidth: small, capHeight: small, hostBytes: smallBytes}
		pool.live = append(pool.live, ctx)
		pool.free = append(pool.free, ctx)
		pool.hostPlanned += ctx.hostBytes
	}
	calls := 0
	pool.create = func(capWidth, capHeight int) (*gpuRouteContext, error) {
		calls++
		return &gpuRouteContext{capWidth: capWidth, capHeight: capHeight}, nil
	}
	ctx, err := pool.acquire(512, 512, nil)
	if err != nil || ctx == nil {
		t.Fatalf("acquire returned (%v, %v), want a context built from recycled scratch", ctx, err)
	}
	if ctx.capWidth < 512 || ctx.capHeight < 512 {
		t.Fatalf("acquired a %dx%d context, want one covering 512x512", ctx.capWidth, ctx.capHeight)
	}
	if calls != 1 {
		t.Fatalf("creation ran %d times, want exactly one", calls)
	}
	// A 512x512 canvas costs four 256x256 contexts' worth of scratch, so the
	// pool must retire four idles and no more.
	if want := 8 - 4; len(pool.free) != want {
		t.Fatalf("free list has %d entries, want %d after retiring for the request", len(pool.free), want)
	}
	if pool.hostPlanned > pool.hostBudget {
		t.Fatalf("host scratch %d exceeds the budget %d", pool.hostPlanned, pool.hostBudget)
	}
}

// TestGPURouteContextPoolChargesRetainedGrowth pins the growth accounting:
// overflow growth accumulated during a lease is folded into the context's
// budgeted bytes and the pool's planned total when the context returns to
// the free list, and the growth arithmetic matches the buffer sizes.
func TestGPURouteContextPoolChargesRetainedGrowth(t *testing.T) {
	wantGrowth := uint64(gpuFinderScanBufferSize(2*gpuFinderScanCapacity)-
		gpuFinderScanBufferSize(gpuFinderScanCapacity)) +
		uint64(gpuFinderChainBufferSize(2*gpuFinderScanCapacity)-
			gpuFinderChainBufferSize(gpuFinderScanCapacity))
	if got := gpuFinderScanGrowthBytes(gpuFinderScanCapacity, 2*gpuFinderScanCapacity); got != wantGrowth {
		t.Fatalf("gpuFinderScanGrowthBytes = %d, want %d", got, wantGrowth)
	}

	pool := newGPURouteContextPool(nil, nil, nil)
	base := gpuRouteContextDeviceBytes(256, 256)
	ctx := &gpuRouteContext{capWidth: 256, capHeight: 256, deviceBytes: base}
	pool.live = append(pool.live, ctx)
	pool.planned = base
	pool.outstanding = 1
	ctx.retainedExtraBytes.Store(wantGrowth)
	pool.release(ctx)
	if ctx.retainedExtraBytes.Load() != 0 {
		t.Fatal("release did not consume the accumulated growth")
	}
	if ctx.deviceBytes != base+wantGrowth {
		t.Fatalf("context bytes = %d, want %d", ctx.deviceBytes, base+wantGrowth)
	}
	if pool.planned != base+wantGrowth {
		t.Fatalf("planned bytes = %d, want %d", pool.planned, base+wantGrowth)
	}
	if len(pool.free) != 1 || pool.free[0] != ctx {
		t.Fatal("released context did not return to the free list")
	}
}

// vulkiBufferAllocationStats walks the direct *vulki.Buffer fields of the
// given structs and reports their distinct count and total size. New buffer
// fields on these structs are counted automatically, so the budget-coverage
// test fails when an allocation is added without updating either the buffer
// count or gpuRouteContextDeviceBytes.
// vulkiBufferAllocationStats totals the distinct device buffers a set of values
// owns. Buffers the values merely borrow are excluded: the pivot catalog belongs
// to the device-wide kernel set and every route context binds the same one, so
// charging it per context would multiply one allocation by the pyramid depth and
// hide exactly the sharing it exists to provide.
func vulkiBufferAllocationStats(borrowed []*vulki.Buffer, values ...any) (uint64, int) {
	var total uint64
	seen := map[*vulki.Buffer]bool{}
	skip := map[*vulki.Buffer]bool{}
	for _, buffer := range borrowed {
		if buffer != nil {
			skip[buffer] = true
		}
	}
	bufferType := reflect.TypeOf((*vulki.Buffer)(nil))
	for _, value := range values {
		v := reflect.ValueOf(value)
		if v.Kind() != reflect.Pointer || v.IsNil() {
			continue
		}
		v = v.Elem()
		for index := range v.NumField() {
			field := v.Field(index)
			if field.Type() != bufferType {
				continue
			}
			buffer := *(**vulki.Buffer)(unsafe.Pointer(field.UnsafeAddr()))
			if buffer == nil || seen[buffer] || skip[buffer] {
				continue
			}
			seen[buffer] = true
			total += buffer.Size()
		}
	}
	return total, len(seen)
}

// TestGPURouteContextDeviceBytesCoversAllocations pins the admission
// estimate against reality: a fully materialized route context - descreen
// and pitch-lag chains included, route canvas allocated - must fit inside
// gpuRouteContextDeviceBytes, and retained overflow growth must be reported
// through the growth hook with its exact byte cost.
func TestGPURouteContextDeviceBytesCoversAllocations(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	defer func() {
		if err := device.Close(); err != nil {
			t.Errorf("close budget-coverage device: %v", err)
		}
	}()
	for _, capSize := range []image.Point{{X: 256, Y: 256}, {X: 768, Y: 512}} {
		kernels := newGPUDecodeKernels(device)
		workspace, err := newGPUDecodeWorkspace(device, kernels, capSize.X, capSize.Y, 1)
		if err != nil {
			_ = kernels.Close()
			t.Fatalf("new %v budget workspace: %v", capSize, err)
		}
		workspace.ownsKernels = true
		base := core.NewBitmap(capSize.X, capSize.Y, 4)
		if err := workspace.ladder.UploadAndBuild(base); err != nil {
			_ = workspace.Close()
			t.Fatalf("upload %v budget workspace: %v", capSize, err)
		}
		if err := kernels.compilePitchLag(); err != nil {
			_ = workspace.Close()
			t.Fatalf("compile pitch-lag kernels: %v", err)
		}
		ctx, err := newGPURouteContext(device, kernels, workspace.ladder, capSize.X, capSize.Y, false)
		if err != nil {
			_ = workspace.Close()
			t.Fatalf("new %v budget context: %v", capSize, err)
		}
		if err := ctx.preparer.ensureDescreen(); err != nil {
			t.Fatalf("materialize %v descreen chain: %v", capSize, err)
		}
		if err := ctx.preparer.ensurePitchLag(); err != nil {
			t.Fatalf("materialize %v pitch-lag chain: %v", capSize, err)
		}
		allocated, allocationCount := vulkiBufferAllocationStats(
			[]*vulki.Buffer{ctx.resident.ldpcCatalog},
			ctx.resident, ctx.resident.binarizer, ctx.preparer,
		)
		if allocationCount != gpuRouteContextBufferCount {
			t.Fatalf(
				"%v context owns %d device buffers, accounting allows %d",
				capSize, allocationCount, gpuRouteContextBufferCount,
			)
		}
		budget := gpuRouteContextDeviceBytes(capSize.X, capSize.Y)
		if allocated > budget {
			t.Fatalf(
				"%v context allocates %d device bytes, budget %d is short by %d",
				capSize, allocated, budget, allocated-budget,
			)
		}
		if ctx.resident.binarizer.onRetainedAllocation == nil {
			t.Fatal("route context binarizer has no growth hook")
		}
		grownCapacity := 2 * gpuFinderScanCapacity
		if err := ctx.resident.binarizer.growFinderScan(grownCapacity); err != nil {
			t.Fatalf("grow %v finder scan: %v", capSize, err)
		}
		wantGrowth := gpuFinderScanGrowthBytes(gpuFinderScanCapacity, grownCapacity)
		if got := ctx.retainedExtraBytes.Load(); got != wantGrowth {
			t.Fatalf("growth hook charged %d bytes, want %d", got, wantGrowth)
		}
		grownTotal, grownAllocationCount := vulkiBufferAllocationStats(
			[]*vulki.Buffer{ctx.resident.ldpcCatalog},
			ctx.resident, ctx.resident.binarizer, ctx.preparer,
		)
		if grownAllocationCount != gpuRouteContextBufferCount {
			t.Fatalf(
				"grown %v context owns %d device buffers, accounting allows %d",
				capSize, grownAllocationCount, gpuRouteContextBufferCount,
			)
		}
		if grownTotal > budget+wantGrowth {
			t.Fatalf(
				"grown %v context allocates %d device bytes, budget plus growth %d is short",
				capSize, grownTotal, budget+wantGrowth,
			)
		}
		if err := ctx.Close(); err != nil {
			t.Errorf("close %v budget context: %v", capSize, err)
		}
		if err := workspace.Close(); err != nil {
			t.Errorf("close %v budget workspace: %v", capSize, err)
		}
	}
}
