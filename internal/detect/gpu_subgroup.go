//go:build !js

package detect

import (
	"encoding/binary"
	"fmt"

	"github.com/srlehn/vulki"
)

// The ballot kernels compact boundaries in lane order, which requires knowing
// which lanes precede a given lane across subgroup boundaries. They get that by
// deriving a subgroup index as local_invocation_index / subgroup_size and
// ordering the per-subgroup totals by it.
//
// **Vulkan does not define any relationship between SubgroupLocalInvocationId
// and LocalInvocationIndex.** RequireFullSubgroups guarantees that subgroups are
// fully populated; it says nothing about how invocations are assigned to them.
// An implementation is free to partition a workgroup in an order that makes the
// derivation meaningless, and the result would not be a failure to build or a
// crash - it would be boundaries emitted out of order, which the window stage
// would read as runs of nonsense lengths. Silent wrong answers, on a path with
// no payload integrity check behind it.
//
// So the relationship is measured on the device before the kernels that depend
// on it are selected, not assumed from the pipeline flag. This is the probe that
// measures it.

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

// finderBallotMinSubgroupSize is the smallest subgroup the ballot kernels can
// run on. They size their per-subgroup total array as WORKGROUP / 4, so a
// smaller subgroup yields more subgroups than the array holds and indexes past
// it. Vulkan permits sizes 1 and 2, so this is a real device class and not a
// theoretical one.
const finderBallotMinSubgroupSize = 4

// subgroupLayoutUsable reports whether this device partitions a workgroup the
// way the ballot kernels need: uniform subgroup size, at least
// finderBallotMinSubgroupSize, dividing the workgroup, and with
// subgroup_invocation_id equal to local_invocation_index modulo that size.
//
// It dispatches once per kernel set and caches the answer. The error is a
// defect - a probe that would not build or run - and is nil when the device
// simply partitions differently, which is a capability limit.
func (set *gpuDecodeKernels) subgroupLayoutUsable() (bool, error) {
	if set == nil || set.device == nil {
		return false, nil
	}
	set.subgroupProbeOnce.Do(func() {
		set.subgroupProbeOK, set.subgroupProbeErr = set.probeSubgroupLayout()
	})
	return set.subgroupProbeOK, set.subgroupProbeErr
}

func (set *gpuDecodeKernels) probeSubgroupLayout() (bool, error) {
	const lanes = 256
	kernel, err := set.device.NewKernel(vulki.KernelOptions{
		WGSL:                 wgslEnableSubgroups + subgroupProbeWGSL,
		Bindings:             []vulki.BindingLayout{{Binding: 0, Access: vulki.BufferReadWrite}},
		RequireFullSubgroups: true,
	})
	if err != nil {
		// A device that cannot promise full subgroups is not defective; it just
		// runs the portable kernels.
		return false, nil
	}
	defer func() { _ = kernel.Close() }()

	buffer, err := set.device.NewBuffer(lanes * 2 * 4)
	if err != nil {
		return false, fmt.Errorf("jabcode: allocate GPU subgroup probe buffer: %w", err)
	}
	defer func() { _ = buffer.Close() }()
	bindings, err := kernel.NewBindings(vulki.BindBuffer(0, buffer))
	if err != nil {
		return false, fmt.Errorf("jabcode: bind GPU subgroup probe: %w", err)
	}
	defer func() { _ = bindings.Close() }()
	recorder, err := set.device.NewRecorder()
	if err != nil {
		return false, fmt.Errorf("jabcode: create GPU subgroup probe recorder: %w", err)
	}
	defer recorder.Abort()
	if err := recorder.Fill(buffer, 0, lanes*2*4, 0); err != nil {
		return false, fmt.Errorf("jabcode: clear GPU subgroup probe: %w", err)
	}
	if err := recorder.Dispatch(kernel, bindings, vulki.Workgroups{X: 1, Y: 1, Z: 1}); err != nil {
		return false, fmt.Errorf("jabcode: dispatch GPU subgroup probe: %w", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		return false, fmt.Errorf("jabcode: run GPU subgroup probe: %w", err)
	}
	raw := make([]byte, lanes*2*4)
	if err := buffer.Download(raw); err != nil {
		return false, fmt.Errorf("jabcode: download GPU subgroup probe: %w", err)
	}

	size := binary.LittleEndian.Uint32(raw[0:])
	if size < finderBallotMinSubgroupSize || lanes%int(size) != 0 {
		return false, nil
	}
	for lane := range lanes {
		if binary.LittleEndian.Uint32(raw[lane*8:]) != size {
			return false, nil
		}
		if binary.LittleEndian.Uint32(raw[lane*8+4:]) != uint32(lane)%size {
			return false, nil
		}
	}
	return true, nil
}
