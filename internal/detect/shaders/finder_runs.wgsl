// Directional run extraction: the parallel replacement for the serial per-line
// walk in finder_line_scan.wgsl.
//
// The serial baseline makes one lane responsible for a whole scan line, which
// wastes the machine twice over. The work per line is unbounded and uneven, so
// lanes in a workgroup diverge for as long as the longest line takes, and the
// state machine forces every sample of a line to be visited in order by one
// lane. Neither is inherent. What that machine actually computes is where the
// sampled bit changes - the run boundaries - and whether sample i is a boundary
// depends on sample i-1 and nothing else.
//
// So this kernel answers only that question, for every sample of every line at
// once, and compacts the boundary positions. Run lengths are differences of
// consecutive boundaries, so the window test that follows reads them without
// this kernel computing them. Deciding which five-run window is a finder is
// then an independent test per window.
//
// There is deliberately no carried state between blocks. A lane needing its
// predecessor's value re-samples it rather than receiving it, which costs one
// extra load and removes every cross-block dependency, the workgroup-shared
// carry it would otherwise need, and the ordering hazards that come with it.
//
// Sampling is f32 and the proportion tests downstream are exact integer ratios.
// Nothing here reproduces host float64: see the binding rule in the plan.

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
    // Boundaries recorded per line and channel. A line producing more is
    // truncated rather than allowed to write into the next line's slots.
    run_capacity: u32,
}

@group(0) @binding(0) var<storage, read> packed_masks: array<u32>;
@group(0) @binding(1) var<storage, read_write> boundaries: array<u32>;
@group(0) @binding(2) var<storage, read> params: Params;
@group(0) @binding(3) var<storage, read_write> boundary_counts: array<u32>;

const WORKGROUP: u32 = 256u;

var<workgroup> flags: array<u32, WORKGROUP>;
// emitted is the workgroup's running total across blocks. It is shared state,
// not a per-lane var, which is the bug the first draft of this kernel had.
var<workgroup> emitted: u32;

fn mask_bit(pixel: u32, channel: u32) -> u32 {
    let word = packed_masks[pixel / 8u];
    return (word >> ((pixel % 8u) * 3u + channel)) & 1u;
}

// sample_at returns the mask bit at sample i along one line. Out-of-frame
// samples return a value no in-frame sample can take, so the frame edge reads
// as a run boundary and needs no separate clip pass.
fn sample_at(origin: vec2<f32>, i: u32, channel: u32) -> u32 {
    let p = origin + f32(i) * vec2<f32>(params.dx, params.dy);
    let x = i32(p.x);
    let y = i32(p.y);
    if x < 0 || x >= i32(params.width) || y < 0 || y >= i32(params.height) {
        return 2u;
    }
    return mask_bit(u32(y) * params.width + u32(x), channel);
}

// Hillis-Steele inclusive scan over flags. log2(WORKGROUP) rounds of uniform
// work, against a serial walk whose cost is the longest line in the group.
fn scan_inclusive(lane: u32) -> u32 {
    var offset = 1u;
    loop {
        if offset >= WORKGROUP {
            break;
        }
        var addend = 0u;
        if lane >= offset {
            addend = flags[lane - offset];
        }
        workgroupBarrier();
        if lane >= offset {
            flags[lane] = flags[lane] + addend;
        }
        workgroupBarrier();
        offset = offset * 2u;
    }
    return flags[lane];
}

// One workgroup covers one (line, channel) pair and strides the line in
// WORKGROUP-sized blocks.
@compute @workgroup_size(WORKGROUP)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let line = group.x;
    let channel = group.y;
    if line >= params.line_count || (params.channel_mask & (1u << channel)) == 0u {
        return;
    }
    let q = params.q_lo + f32(line) * params.q_step;
    let origin = q * vec2<f32>(params.nx, params.ny);
    let slot_base = (line * 3u + channel) * params.run_capacity;

    if lane == 0u {
        emitted = 0u;
    }
    workgroupBarrier();

    var block = 0u;
    loop {
        if block >= params.line_length {
            break;
        }
        let i = block + lane;

        // Sample 0 opens the first run; every other sample is a boundary when
        // it differs from its predecessor, which it re-samples itself.
        var starts = 0u;
        if i < params.line_length {
            let value = sample_at(origin, i, channel);
            if i == 0u {
                starts = 1u;
            } else if value != sample_at(origin, i - 1u, channel) {
                starts = 1u;
            }
        }
        flags[lane] = starts;
        workgroupBarrier();
        let index = scan_inclusive(lane);
        let block_total = flags[WORKGROUP - 1u];

        if starts == 1u {
            let slot = emitted + index - 1u;
            if slot < params.run_capacity {
                boundaries[slot_base + slot] = i;
            }
        }
        workgroupBarrier();
        if lane == 0u {
            emitted = emitted + block_total;
        }
        workgroupBarrier();
        block += WORKGROUP;
    }

    if lane == 0u {
        boundary_counts[line * 3u + channel] = min(emitted, params.run_capacity);
    }
}
