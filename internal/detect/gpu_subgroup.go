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
// it. Vulkan permits sizes 1 and 2, so this is a real device class.
const finderBallotMinSubgroupSize = 4

// finderBallotWorkgroup mirrors WORKGROUP in the ballot shaders. The workgroup
// size is declared in WGSL, but full subgroups require it to divide into whole
// subgroups, so the host needs the same number to decide that before building
// anything.
const finderBallotWorkgroup = 256

// finderScanWorkgroupSupported reports whether this device can launch the
// directional scan kernels at all.
//
// **Every kernel in this stage declares a 256-lane workgroup**, the ballot and
// scan window kernels alike, so this is a prerequisite for the whole stage and
// not a choice between variants. Vulkan Core guarantees only 128 for both
// MaxComputeWorkGroupInvocations and the x dimension of MaxComputeWorkGroupSize,
// so a conformant device may refuse every one of them. Such a device keeps the
// CPU route, which is unaffected; supporting it on the device would mean a
// second set of kernels at 128 lanes, which is worth building only if one turns
// up.
func finderScanWorkgroupSupported(limits vulki.Limits) bool {
	return limits.MaxComputeWorkGroupInvocations >= finderBallotWorkgroup &&
		limits.MaxComputeWorkGroupSize[0] >= finderBallotWorkgroup
}

// finderBallotUsable reports whether the ballot kernels may be built here.
//
// **The size a compute pipeline runs at is the device's own SubgroupSize**, not
// anything in the size-control range. A pipeline only draws from
// MinSubgroupSize..MaxSubgroupSize when it opts in with
// ALLOW_VARYING_SUBGROUP_SIZE, which vulki never sets. So SubgroupSize is the
// only size this decision may look at, and treating the range's low end as
// reachable - as this did briefly - rules out capable devices for a size they
// will never be given.
func (set *gpuDecodeKernels) finderBallotUsable() bool {
	if set == nil || set.device == nil {
		return false
	}
	return finderBallotUsableFor(set.device.Info().Limits)
}

// finderBallotUsableFor decides the same thing from limits alone, so the device
// classes the development hardware cannot present stay reachable in a test.
func finderBallotUsableFor(limits vulki.Limits) bool {
	if !finderScanWorkgroupSupported(limits) {
		return false
	}
	// A zero size means the implementation reported nothing, which is unknown
	// rather than supported.
	if limits.SubgroupSize == 0 {
		return false
	}
	if limits.SubgroupOperations&finderBallotOperations != finderBallotOperations ||
		!limits.FullSubgroupsSupported {
		return false
	}
	// Full subgroups require the workgroup to be a whole number of subgroups,
	// and the per-subgroup array bounds how small they may be.
	return limits.SubgroupSize >= finderBallotMinSubgroupSize &&
		finderBallotWorkgroup%limits.SubgroupSize == 0
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
	if !set.finderBallotUsable() {
		return false, nil
	}
	kernel, err := set.device.NewKernel(vulki.KernelOptions{
		WGSL:                 enableSubgroupsWGSL + subgroupProbeWGSL,
		Bindings:             []vulki.BindingLayout{{Binding: 0, Access: vulki.BufferReadWrite}},
		RequireFullSubgroups: true,
	})
	if err != nil {
		// Not a capability answer: the device has already reported that it
		// supports full subgroups and a usable size, so a probe that will not
		// build is a defect. Swallowing it here would turn a broken probe shader
		// into a device that merely partitions differently, and the ballot route
		// would disappear with nothing to say why.
		return false, fmt.Errorf("jabcode: build GPU subgroup probe: %w", err)
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
