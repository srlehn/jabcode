//go:build !js

package detect

import (
	"testing"

	"github.com/srlehn/vulki"
)

// Warming route contexts is the one part of the warm-up that cannot be checked
// against a fake device: the defect it exists to prevent is the pool answering
// a smaller request from a larger free context, which only a real pool with
// real contexts does. A count is not enough either - one context of the right
// size satisfies every smaller level, so a pool holding a single full-size
// context passes any "did warming allocate something" test while leaving every
// concurrent level to allocate cold.
func TestWarmRouteContextsCoversEveryLevel(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Cleanup(func() { _ = device.Close() })

	const width, height, levels = 2048, 1536, 4
	kernels := newGPUDecodeKernels(device)
	workspace, err := newGPUDecodeWorkspace(device, kernels, width, height, levels)
	if err != nil {
		t.Skipf("no workspace: %v", err)
	}
	t.Cleanup(func() { _ = workspace.Close() })

	pool := workspace.contexts
	if got := len(pool.live); got != 0 {
		t.Fatalf("a fresh pool already holds %d contexts", got)
	}
	warmRouteContexts(workspace)

	if pool.outstanding != 0 {
		t.Errorf("warming left %d contexts leased", pool.outstanding)
	}
	// One free context per level, each at least that level's padded size, and
	// no two of the same capacity: reuse would show up as a short list.
	want := make(map[[2]int]bool, levels)
	for _, level := range workspace.ladder.levels {
		want[[2]int{gpuRoutePadded(level.width), gpuRoutePadded(level.height)}] = true
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
	held := make([]*gpuRouteContext, 0, levels)
	for _, level := range workspace.ladder.levels {
		ctx, err := pool.acquire(level.width, level.height, nil)
		if err != nil {
			t.Fatalf("acquire %dx%d after warming: %v", level.width, level.height, err)
		}
		held = append(held, ctx)
	}
	if len(pool.free) != 0 {
		t.Errorf("%d warmed contexts went unused by the levels they were warmed for",
			len(pool.free))
	}
	for _, ctx := range held {
		pool.release(ctx)
	}
}
