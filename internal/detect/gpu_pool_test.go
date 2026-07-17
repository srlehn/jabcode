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
