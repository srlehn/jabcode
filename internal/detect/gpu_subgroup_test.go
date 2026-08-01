//go:build !js

package detect

import (
	"encoding/binary"
	"testing"

	"github.com/srlehn/vulki"
)

// subgroupProbeWGSL reports each lane's subgroup size and its index within the
// subgroup, which is all the host needs to check how the workgroup was
// partitioned.
const subgroupProbeWGSL = `
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_index) lane: u32,
    @builtin(subgroup_size) size: u32,
    @builtin(subgroup_invocation_id) id: u32,
) {
    out[lane * 2u] = size;
    out[lane * 2u + 1u] = id;
}
`

// fullSubgroupsUsable reports whether the ballot kernels may run here, by
// building a probe under the same RequireFullSubgroups guarantee they use and
// then checking that the guarantee actually holds.
//
// The check is not redundant with the flag. RequireFullSubgroups is what makes
// deriving a subgroup index from the local invocation index legal, but nothing
// in the Go API observes the resulting partition, so this is where that promise
// is held to account on whatever adapter the suite runs on.
func fullSubgroupsUsable(t testing.TB, device *vulki.Device) (bool, string) {
	t.Helper()
	kernel, err := device.NewKernel(vulki.KernelOptions{
		WGSL:                 wgslEnableSubgroups + subgroupProbeWGSL,
		Bindings:             []vulki.BindingLayout{{Binding: 0, Access: vulki.BufferReadWrite}},
		RequireFullSubgroups: true,
	})
	if err != nil {
		return false, "full subgroups are unavailable: " + err.Error()
	}
	defer func() { _ = kernel.Close() }()
	const lanes = 256
	buf, err := device.NewBuffer(lanes * 2 * 4)
	if err != nil {
		t.Fatalf("allocate subgroup probe buffer: %v", err)
	}
	defer func() { _ = buf.Close() }()
	bindings, err := kernel.NewBindings(vulki.BindBuffer(0, buf))
	if err != nil {
		t.Fatalf("bind subgroup probe: %v", err)
	}
	defer func() { _ = bindings.Close() }()
	recorder, err := device.NewRecorder()
	if err != nil {
		t.Fatalf("new recorder: %v", err)
	}
	defer recorder.Abort()
	if err := recorder.Update(buf, 0, make([]byte, lanes*2*4)); err != nil {
		t.Fatalf("clear subgroup probe: %v", err)
	}
	if err := recorder.Dispatch(kernel, bindings, vulki.Workgroups{X: 1, Y: 1, Z: 1}); err != nil {
		t.Fatalf("dispatch subgroup probe: %v", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		t.Fatalf("run subgroup probe: %v", err)
	}
	raw := make([]byte, lanes*2*4)
	if err := buf.Download(raw); err != nil {
		t.Fatalf("download subgroup probe: %v", err)
	}
	size := binary.LittleEndian.Uint32(raw[0:])
	if size == 0 || lanes%int(size) != 0 {
		return false, "subgroup size does not divide the workgroup"
	}
	for lane := range lanes {
		gotSize := binary.LittleEndian.Uint32(raw[lane*8:])
		gotID := binary.LittleEndian.Uint32(raw[lane*8+4:])
		if gotSize != size {
			return false, "subgroup size is not uniform across the workgroup"
		}
		if gotID != uint32(lane)%size {
			return false, "subgroups are not full and linearly assigned"
		}
	}
	return true, ""
}

// The ballot kernels are selected from a reported capability and built under a
// pipeline guarantee. This checks that the two agree with what the device
// actually does, so a wrong capability bit or an unhonoured guarantee shows up
// here rather than as silently misordered boundaries.
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

	claimed := kernels.subgroupBallotUsable()
	usable, reason := fullSubgroupsUsable(t, device)
	t.Logf("ballot selected = %t, full linear subgroups observed = %t", claimed, usable)
	if claimed && !usable {
		t.Fatalf("device advertises ballot support but its partitioning is unusable: %s", reason)
	}
	if !usable {
		t.Logf("ballot kernels must not be used here: %s", reason)
	}

	// The selector must hand back a working fused kernel either way, because
	// the alternative it must never fall back to is a boundary buffer.
	for _, layout := range []finderScanLayout{finderScanInterleaved, finderScanBitplane} {
		if _, err := kernels.finderWindows(layout); err != nil {
			t.Fatalf("select fused window kernel for %s: %v", layout.name(), err)
		}
	}

	// A device that advertises ballot support and then cannot build the kernel
	// is a defect, not a capability limit, and falling back costs about 30% on
	// every read forever. Without this check an editing mistake in the ballot
	// shader would look exactly like an adapter that never had subgroups.
	if claimed {
		if err := kernels.ballotFallbackError(); err != nil {
			t.Fatalf("device advertises ballot support but the ballot kernel did not build: %v", err)
		}
	}
}
