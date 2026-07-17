package detect

import (
	"errors"
	"fmt"
	"testing"

	"github.com/srlehn/vulki"
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
