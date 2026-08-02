// Shared parameter block and mask binding for the directional scan kernels.
//
// The directional prototypes differ only in how they compact and in how the
// masks are stored, so everything they agree on lives here and is concatenated
// ahead of a mask accessor, the geometry helpers and one body. Duplicating the
// clip and the addressing per prototype would make them incomparable the first
// time one of the copies was fixed.

struct Params {
    width: u32,
    height: u32,
    channel_mask: u32,
    // Samples per line, uniform across lines so a run slot is a plain
    // (line, channel, index) triple and needs no prefix over earlier lines.
    line_length: u32,
    dx: f32,
    dy: f32,
    nx: f32,
    ny: f32,
    q_lo: f32,
    q_step: f32,
    line_count: u32,
    // Boundary slots available per line and channel. Writes past it are
    // dropped, but the count reported is the true one so the host can tell a
    // complete list from a truncated one and grow or reroute.
    run_capacity: u32,
    // Words per channel plane in the bitplane layout. The interleaved layout
    // ignores it.
    plane_words: u32,
}

// Neither address space is a style choice. `masks` is a runtime-sized array,
// which WGSL permits only in the storage address space; no host API could make
// it a uniform buffer. Params is a fixed 52-byte block read identically by every
// invocation, which is what the uniform address space is for and what puts it on
// the constant path where hardware has one. Uniform raises the struct's required
// alignment to 16 and leaves its size at 52, so the buffer is exactly the
// struct.
//
// For params alone the choice would not have affected correctness either way:
// the analysis treats read-only module-scope variables as uniform, storage or
// not. But read-only is not the whole story, and the distinction matters for the
// barrier-carrying loops below. A *fixed* access such as params.line_length is
// uniform, which is what lets those loops be proved uniform. An *indexed* access
// such as masks[pixel] is not: indexing propagates the uniformity of the index,
// and the mask reads here are indexed by sample position, which is per lane.
// That is fine because no barrier or subgroup operation is ever reached through
// a branch on a mask value - the sampled bits only ever feed data, never control
// flow that a collective operation sits inside.
@group(0) @binding(0) var<storage, read> masks: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
