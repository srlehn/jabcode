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
// on it are selected, not assumed from the pipeline flag. The probe is
// shaders/subgroup_probe.wgsl.

// finderBallotMinSubgroupSize is the smallest subgroup the ballot kernels can
// run on. They size their per-subgroup total array as WORKGROUP / 4, so a
// smaller subgroup yields more subgroups than the array holds and indexes past
// it. Vulkan permits sizes 1 and 2, so this is a real device class and not a
// theoretical one; where the device can pin a size, such a device keeps the
// ballot route by being run at this size instead of its own.
const finderBallotMinSubgroupSize = 4

// finderBallotWorkgroup mirrors WORKGROUP in the ballot shaders. The workgroup
// size is declared in WGSL, but full subgroups require it to divide into whole
// subgroups, so the host needs the same number to decide that before building
// anything.
const finderBallotWorkgroup = 256

// finderBallotSubgroupSize reports how the ballot kernels have to be built
// here: the subgroup size to pin, zero when the device's own partitioning is
// already inside the shaders' bounds, and whether they can run at all.
//
// Pinning exists for the device class whose subgroups may come out smaller than
// the per-subgroup array holds. Such a device is entirely capable of running the
// ballot form at a larger size, and without a pinned size it would take the
// portable twin and its permanent slowdown for no reason.
func (set *gpuDecodeKernels) finderBallotSubgroupSize() (uint32, bool) {
	if set == nil || set.device == nil {
		return 0, false
	}
	return finderBallotSubgroupSizeFor(set.device.Info().Limits)
}

// finderBallotSubgroupSizeFor decides the same thing from limits alone, so the
// device classes the development hardware cannot present - a size range reaching
// below the array bound, a device without size control - stay reachable in a
// test.
func finderBallotSubgroupSizeFor(limits vulki.Limits) (uint32, bool) {
	// A zero size means the implementation reported nothing, which is unknown
	// rather than supported.
	if limits.SubgroupSize == 0 {
		return 0, false
	}
	if limits.SubgroupOperations&finderBallotOperations != finderBallotOperations ||
		!limits.FullSubgroupsSupported {
		return 0, false
	}
	// Both the advertised size and the low end of the size-control range are
	// checked, because without a pinned size a pipeline may run anywhere in that
	// range.
	if limits.SubgroupSize >= finderBallotMinSubgroupSize &&
		limits.MinSubgroupSize >= finderBallotMinSubgroupSize &&
		finderBallotWorkgroup%limits.SubgroupSize == 0 {
		return 0, true
	}
	if !limits.RequiredSubgroupSizeSupported {
		return 0, false
	}
	size := max(uint32(finderBallotMinSubgroupSize), limits.MinSubgroupSize)
	if size > limits.MaxSubgroupSize || finderBallotWorkgroup%size != 0 {
		return 0, false
	}
	return size, true
}

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
	const lanes = finderBallotWorkgroup
	pin, ok := set.finderBallotSubgroupSize()
	if !ok {
		return false, nil
	}
	kernel, err := set.device.NewKernel(vulki.KernelOptions{
		WGSL:                 enableSubgroupsWGSL + subgroupProbeWGSL,
		Bindings:             []vulki.BindingLayout{{Binding: 0, Access: vulki.BufferReadWrite}},
		RequireFullSubgroups: true,
		RequiredSubgroupSize: pin,
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
	// A pinned size that did not take is a driver disagreeing with itself, not a
	// device that partitions differently, so it is reported rather than folded
	// into the ordinary fallback.
	if pin != 0 && size != pin {
		return false, fmt.Errorf("jabcode: GPU pinned subgroup size %d but ran at %d", pin, size)
	}
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
