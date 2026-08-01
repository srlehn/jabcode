// The subgroup enable directive, alone in a file because it has to come first.
//
// WGSL requires enable directives to precede every declaration in the module,
// and the kernels that need this one are assembled by concatenating a shared
// parameter block, a mask accessor, the geometry helpers and a body - so no
// shader file is the start of its module and none of them can carry it. The Go
// side prepends this fragment instead. The vendored naga tolerates the directive
// appearing mid-module; the specification does not, and a stricter compiler
// would reject it.

enable subgroups;
