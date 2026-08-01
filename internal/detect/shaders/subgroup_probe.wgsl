// Reports each lane's subgroup size and its index within the subgroup, which is
// all the host needs to see how the implementation partitioned the workgroup.
//
// The ballot kernels derive a subgroup index as
// local_invocation_index / subgroup_size. Vulkan defines no relationship between
// SubgroupLocalInvocationId and LocalInvocationIndex, so that derivation has to
// be measured on the device before those kernels are selected rather than
// inferred from the full-subgroups pipeline flag, which only promises the
// subgroups are populated. See subgroupLayoutUsable.
//
// The workgroup size matches the kernels this probes for, because a partitioning
// observed at one size says nothing about another.

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
