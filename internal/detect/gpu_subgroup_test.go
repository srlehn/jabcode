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

// fullSubgroupsUsable reports whether the ballot kernels' partitioning
// assumption holds on this device.
//
// Those kernels derive a lane's subgroup index as
// local_invocation_index / subgroup_size and order the per-subgroup totals by
// it. That is only meaningful when the workgroup is split into full, linearly
// assigned subgroups, which Vulkan guarantees only under a pipeline flag vulki
// cannot request. So it is measured rather than assumed, and the scan variant
// covers the case where it does not hold.
func fullSubgroupsUsable(t testing.TB, device *vulki.Device) (bool, string) {
	t.Helper()
	kernel, err := device.NewKernel(vulki.KernelOptions{
		WGSL:     wgslEnableSubgroups + subgroupProbeWGSL,
		Bindings: []vulki.BindingLayout{{Binding: 0, Access: vulki.BufferReadWrite}},
	})
	if err != nil {
		return false, "subgroup operations are unavailable: " + err.Error()
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

// The ballot kernels stand on an assumption Vulkan does not let vulki request,
// so this pins it down on whatever adapter the suite runs against. It is not a
// test of the kernels: it is the test that decides whether they may be used at
// all, and the scan variants exist for when it says no.
func TestGPUFullSubgroupPartitioning(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Cleanup(func() { _ = device.Close() })
	usable, reason := fullSubgroupsUsable(t, device)
	t.Logf("adapter %s: full linear subgroups usable = %t", device.Info().AdapterName, usable)
	if !usable {
		t.Logf("ballot kernels must not be used here: %s", reason)
	}
}
