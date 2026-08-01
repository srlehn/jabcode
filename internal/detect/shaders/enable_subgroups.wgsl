// The subgroup enable directive, alone in a file because it has to come first.
//
// WGSL requires every enable directive to precede all declarations in the
// module. The kernels that need this one are assembled by concatenating a
// shared parameter block, a mask accessor, the geometry helpers and a body, so
// no shader file is the start of its module and none of them can carry it; the
// Go side prepends this fragment instead.
//
// `subgroups` is an approved WGSL enable-extension, alongside f16,
// clip_distances, dual_source_blending, primitive_index and
// subgroup_size_control. It is what admits subgroupBallot and the
// subgroup_size and subgroup_invocation_id builtins. subgroup_size_control is a
// separate extension and is deliberately not enabled: nothing here varies the
// subgroup size.
//
// The vendored naga accepts an enable directive appearing after declarations.
// The specification does not, so relying on that would be building on one
// compiler's tolerance rather than on the language.

enable subgroups;
