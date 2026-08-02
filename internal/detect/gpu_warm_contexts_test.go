//go:build !js

package detect

import (
	"testing"
	"time"
)

// warmTestWorkspace builds a workspace whose pool creates fake contexts, so
// both warming defects are pinned on a machine with no Vulkan at all. budget
// is the pool's device-byte budget; zero leaves it unbounded.
func warmTestWorkspace(levels [][2]int, budget uint64) *gpuDecodeWorkspace {
	ladder := &gpuCanvasLadder{}
	for _, level := range levels {
		ladder.levels = append(ladder.levels, gpuCanvasLevel{width: level[0], height: level[1]})
	}
	pool := newGPURouteContextPool(nil, nil, nil)
	pool.create = func(capWidth, capHeight int) (*gpuRouteContext, error) {
		return &gpuRouteContext{
			capWidth: capWidth, capHeight: capHeight,
			deviceBytes: gpuRouteContextDeviceBytes(capWidth, capHeight),
		}, nil
	}
	if budget > 0 {
		pool.budget, pool.budgetKnown = budget, true
	}
	return &gpuDecodeWorkspace{ladder: ladder, contexts: pool}
}

// Warming has to leave one context per level. A count is not enough and neither
// is "warming allocated something": the pool answers any request from a free
// context large enough, so a single full-size context satisfies every smaller
// level in turn and passes both of those checks while leaving the other three
// concurrent levels to allocate cold.
func TestWarmRouteContextsCoversEveryLevel(t *testing.T) {
	levels := [][2]int{{2048, 1536}, {1024, 768}, {512, 384}, {256, 192}}
	workspace := warmTestWorkspace(levels, 0)
	pool := workspace.contexts

	warmRouteContexts(workspace)

	if pool.outstanding != 0 {
		t.Errorf("warming left %d contexts leased", pool.outstanding)
	}
	want := make(map[[2]int]bool, len(levels))
	for _, level := range levels {
		want[[2]int{gpuRoutePadded(level[0]), gpuRoutePadded(level[1])}] = true
	}
	if len(pool.free) != len(want) {
		t.Fatalf("warming left %d free contexts for %d distinct level sizes",
			len(pool.free), len(want))
	}
	got := make(map[[2]int]bool, len(pool.free))
	for _, ctx := range pool.free {
		got[[2]int{ctx.capWidth, ctx.capHeight}] = true
	}
	for size := range want {
		if !got[size] {
			t.Errorf("no warmed context for a %dx%d level", size[0], size[1])
		}
	}

	// The point of warming is that the decode's own acquisitions then create
	// nothing. Take every level at once, as the concurrent routes do.
	created := 0
	pool.create = func(capWidth, capHeight int) (*gpuRouteContext, error) {
		created++
		return &gpuRouteContext{capWidth: capWidth, capHeight: capHeight}, nil
	}
	held := make([]*gpuRouteContext, 0, len(levels))
	for _, level := range levels {
		ctx, err := pool.acquire(level[0], level[1], nil)
		if err != nil {
			t.Fatalf("acquire %dx%d after warming: %v", level[0], level[1], err)
		}
		held = append(held, ctx)
	}
	if created != 0 {
		t.Errorf("the decode created %d contexts warming should have provided", created)
	}
	for _, ctx := range held {
		pool.release(ctx)
	}
}

// Each level fits the budget alone or the pool rejects it outright, but all of
// them together need not. Warming holds every lease until they all exist, so
// the only leases that could satisfy the request it cannot afford are its own -
// parking there would never end, and the read that started the warm-up waits on
// it. Warming must return with whatever it managed to allocate.
func TestWarmRouteContextsDoesNotParkOnACumulativeBudget(t *testing.T) {
	levels := [][2]int{{2048, 1536}, {1024, 768}, {512, 384}, {256, 192}}
	// Room for the largest level alone, and nowhere near all four.
	budget := gpuRouteContextDeviceBytes(gpuRoutePadded(2048), gpuRoutePadded(1536)) + 1
	workspace := warmTestWorkspace(levels, budget)

	done := make(chan struct{})
	go func() {
		defer close(done)
		warmRouteContexts(workspace)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("warming parked waiting for leases it holds itself")
	}
	if workspace.contexts.outstanding != 0 {
		t.Errorf("warming left %d contexts leased", workspace.contexts.outstanding)
	}
	if len(workspace.contexts.free) == 0 {
		t.Error("warming gave up without keeping the contexts it did allocate")
	}
}
