//go:build !js

package detect

import (
	"testing"

	"github.com/srlehn/vulki"
)

// The ballot kernels are selected from advertised capabilities plus a measured
// partitioning probe. This reports what this adapter decided and holds the
// decision to two rules: a device that passes selection must actually be able to
// hand back a working kernel, and falling back for a reason that is not a
// capability limit is a defect.
//
// The probe itself lives in production code, not here, because Vulkan defines no
// relationship between SubgroupLocalInvocationId and LocalInvocationIndex. A
// probe that only tests would prove the assumption on the development adapter
// and leave every other one to fail silently, emitting boundaries in the wrong
// order with nothing to catch it.
func TestGPUFullSubgroupPartitioning(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	kernels := newGPUDecodeKernels(device)
	t.Cleanup(func() {
		_ = kernels.Close()
		_ = device.Close()
	})
	limits := device.Info().Limits
	t.Logf("adapter %s: subgroup size %d (min %d, max %d), operations %#x",
		device.Info().AdapterName, limits.SubgroupSize,
		limits.MinSubgroupSize, limits.MaxSubgroupSize, limits.SubgroupOperations)

	advertised := kernels.subgroupBallotUsable()
	layout, err := kernels.subgroupLayoutUsable()
	if err != nil {
		t.Fatalf("subgroup partitioning probe failed: %v", err)
	}
	selected, err := kernels.subgroupKernelsUsable()
	if err != nil {
		t.Fatalf("device advertises ballot support but the ballot kernel did not build: %v", err)
	}
	t.Logf("ballot advertised = %t, partitioning usable = %t, kernels selected = %t",
		advertised, layout, selected)
	if !selected {
		t.Log("this adapter runs the portable fused kernel by design")
	}

	// The selector must hand back a working fused kernel either way, because
	// the alternative it must never fall back to is a boundary buffer.
	for _, layout := range []finderScanLayout{finderScanInterleaved, finderScanBitplane} {
		if _, err := kernels.finderWindows(layout); err != nil {
			t.Fatalf("select fused window kernel for %s: %v", layout.name(), err)
		}
	}

	// Selecting the portable kernel for a reason that is not a capability limit
	// is a defect, and it costs about 30% on every read forever. Without this
	// check an editing mistake in the ballot shader would look exactly like an
	// adapter that never had subgroups.
	if err := kernels.ballotFallbackError(); err != nil {
		t.Fatalf("fell back off the ballot kernel for a non-capability reason: %v", err)
	}
}
