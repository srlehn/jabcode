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

@group(0) @binding(0) var<storage, read> masks: array<u32>;
@group(0) @binding(2) var<storage, read> params: Params;
