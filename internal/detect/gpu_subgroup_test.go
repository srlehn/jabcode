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

	advertised := kernels.finderBallotUsable()
	t.Logf("full subgroups %t, size pinnable %t",
		limits.FullSubgroupsSupported, limits.RequiredSubgroupSizeSupported)
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
	// the alternative it must never fall back to is a boundary buffer. The one
	// exception is a device too small for the workgroup every variant declares,
	// where there is nothing to select.
	if !finderScanWorkgroupSupported(limits) {
		t.Skipf("adapter caps compute workgroups at %d invocations, %d in x",
			limits.MaxComputeWorkGroupInvocations, limits.MaxComputeWorkGroupSize[0])
	}
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

// The selection rule decides which devices keep the fast compaction, and the
// development adapter presents exactly one point in that space: a size of 32.
// Every other class is reachable only from limits.
//
// The rule reads SubgroupSize and never the size-control range, and the two
// cases below that hold MinSubgroupSize at 2 are what pin that down. A compute
// pipeline runs at SubgroupSize unless it opts into ALLOW_VARYING_SUBGROUP_SIZE,
// which vulki does not set, so a low range end is not a size the kernels can be
// handed.
func TestFinderBallotSelection(t *testing.T) {
	const ops = finderBallotOperations
	// Every case is given a workgroup budget the kernels fit in, so only the
	// cases about that budget vary it.
	roomy := func(l vulki.Limits) vulki.Limits {
		if l.MaxComputeWorkGroupInvocations == 0 {
			l.MaxComputeWorkGroupInvocations = finderBallotWorkgroup
		}
		if l.MaxComputeWorkGroupSize[0] == 0 {
			l.MaxComputeWorkGroupSize[0] = finderBallotWorkgroup
		}
		return l
	}
	for _, tc := range []struct {
		name   string
		limits vulki.Limits
		ok     bool
	}{{
		name:   "a size within bounds is usable",
		limits: vulki.Limits{SubgroupSize: 32, MinSubgroupSize: 32, MaxSubgroupSize: 32, SubgroupOperations: ops, FullSubgroupsSupported: true},
		ok:     true,
	}, {
		name:   "a range reaching below the array bound does not matter",
		limits: vulki.Limits{SubgroupSize: 32, MinSubgroupSize: 2, MaxSubgroupSize: 64, SubgroupOperations: ops, FullSubgroupsSupported: true, RequiredSubgroupSizeSupported: true},
		ok:     true,
	}, {
		name:   "nor does it matter without size control",
		limits: vulki.Limits{SubgroupSize: 32, MinSubgroupSize: 2, MaxSubgroupSize: 64, SubgroupOperations: ops, FullSubgroupsSupported: true},
		ok:     true,
	}, {
		// The per-subgroup total array holds WORKGROUP / 4 entries, so a subgroup
		// of 2 would produce 128 subgroups and index past it.
		name:   "a size under the array bound is out",
		limits: vulki.Limits{SubgroupSize: 2, MinSubgroupSize: 2, MaxSubgroupSize: 2, SubgroupOperations: ops, FullSubgroupsSupported: true, RequiredSubgroupSizeSupported: true},
		ok:     false,
	}, {
		// Full subgroups require the workgroup to be a whole number of subgroups.
		name:   "a size that does not divide the workgroup is out",
		limits: vulki.Limits{SubgroupSize: 512, MinSubgroupSize: 512, MaxSubgroupSize: 512, SubgroupOperations: ops, FullSubgroupsSupported: true},
		ok:     false,
	}, {
		name:   "ballot support without full subgroups is out",
		limits: vulki.Limits{SubgroupSize: 32, MinSubgroupSize: 32, MaxSubgroupSize: 32, SubgroupOperations: ops},
		ok:     false,
	}, {
		name:   "full subgroups without the ballot class is out",
		limits: vulki.Limits{SubgroupSize: 32, MinSubgroupSize: 32, MaxSubgroupSize: 32, SubgroupOperations: vulki.SubgroupBasic, FullSubgroupsSupported: true},
		ok:     false,
	}, {
		name:   "unreported subgroup properties are unknown, not supported",
		limits: vulki.Limits{SubgroupOperations: ops, FullSubgroupsSupported: true, RequiredSubgroupSizeSupported: true},
		ok:     false,
	}, {
		// Vulkan Core guarantees only 128, so this is a conformant device and
		// not a broken one. Selecting it would build a probe that cannot launch.
		name: "a workgroup budget under the kernels' own size is out",
		limits: vulki.Limits{
			SubgroupSize: 32, MinSubgroupSize: 32, MaxSubgroupSize: 32,
			SubgroupOperations: ops, FullSubgroupsSupported: true,
			MaxComputeWorkGroupInvocations: 128,
			MaxComputeWorkGroupSize:        [3]uint32{128, 128, 64},
		},
		ok: false,
	}, {
		name: "an x dimension under it is equally out",
		limits: vulki.Limits{
			SubgroupSize: 32, MinSubgroupSize: 32, MaxSubgroupSize: 32,
			SubgroupOperations: ops, FullSubgroupsSupported: true,
			MaxComputeWorkGroupInvocations: 1024,
			MaxComputeWorkGroupSize:        [3]uint32{128, 1024, 64},
		},
		ok: false,
	}} {
		t.Run(tc.name, func(t *testing.T) {
			if ok := finderBallotUsableFor(roomy(tc.limits)); ok != tc.ok {
				t.Fatalf("got usable %t, want %t", ok, tc.ok)
			}
		})
	}
}
