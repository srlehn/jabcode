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
// once, and compacts the boundary positions. Deciding which five-run window is
// a finder is then an independent test per window.
//
// Output contract, which the window stage depends on:
//
//   - Boundaries are ascending sample indices, the first being the line's first
//     in-frame sample and the last being one past its last. A run's length is
//     therefore always boundary[k+1] - boundary[k], with no special case for the
//     final run and no implicit end position to remember.
//   - N boundaries describe N-1 runs. A line with no in-frame samples reports
//     zero boundaries, not one.
//   - Every run described is entirely inside the frame. Out-of-frame samples are
//     clipped away rather than marked, so no run in this list needs excluding.
//   - The reported count is the true one even when it exceeds run_capacity, in
//     which case the stored list is truncated and must not be used as though it
//     were whole.
//
// The only state carried between blocks is the running emitted count, which
// lane 0 owns. The predecessor *value* dependency is gone: each lane publishes
// its own sample to workgroup memory and lanes 1..255 read their neighbour's
// from there, so only lane 0 re-samples across a block edge. An earlier version
// had every lane sample both i and i-1, which nearly doubled the scattered
// reads to avoid a dependency that shared memory removes for one extra load per
// block.
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
    // Boundary slots available per line and channel. Writes past it are
    // dropped, but the count reported is the true one so the host can tell a
    // complete list from a truncated one and grow or reroute.
    run_capacity: u32,
}

@group(0) @binding(0) var<storage, read> packed_masks: array<u32>;
@group(0) @binding(1) var<storage, read_write> boundaries: array<u32>;
@group(0) @binding(2) var<storage, read> params: Params;
@group(0) @binding(3) var<storage, read_write> boundary_counts: array<u32>;

const WORKGROUP: u32 = 256u;

var<workgroup> flags: array<u32, WORKGROUP>;
// values holds each lane's sample so its right-hand neighbour can read it
// instead of loading it again.
var<workgroup> values: array<u32, WORKGROUP>;
// emitted is the workgroup's running total across blocks, and is the one piece
// of state that genuinely is carried. It is shared, not a per-lane var, which
// is the bug the first draft of this kernel had.
var<workgroup> emitted: u32;

fn mask_bit(pixel: u32, channel: u32) -> u32 {
    let word = packed_masks[pixel / 8u];
    return (word >> ((pixel % 8u) * 3u + channel)) & 1u;
}

// sample_at returns the mask bit at sample i along one line. Callers only ask
// about samples inside the clipped span, so the bounds test is a guard rather
// than the mechanism that finds the frame edge.
//
// Coordinates floor rather than truncate toward zero. Truncation maps a
// coordinate in (-1, 0) onto row or column 0, reading a pixel on the far side
// of the frame edge as though the line were still inside - and not by one
// sample but by 1/|component| of them, about 3.7 at 15 degrees. The CPU walk
// has that artifact; nothing requires reproducing it.
fn sample_at(origin: vec2<f32>, i: i32, channel: u32) -> u32 {
    let p = floor(origin + f32(i) * vec2<f32>(params.dx, params.dy));
    let x = i32(p.x);
    let y = i32(p.y);
    if x < 0 || x >= i32(params.width) || y < 0 || y >= i32(params.height) {
        return 2u;
    }
    return mask_bit(u32(y) * params.width + u32(x), channel);
}

// clip_line restricts a line to the frame, returning the first and last in-frame
// sample index and whether any exist. Each axis contributes the interval of i
// keeping that coordinate in range and the span is their intersection; a zero
// step means that coordinate never moves, so the line either holds throughout or
// misses the frame entirely.
//
// Clipping here rather than letting an out-of-frame sentinel mark the edge is
// what keeps a sentinel run out of the output. If the boundary list contained
// the transition into and out of the frame, the window stage could not tell that
// run from a real one, and an out-of-frame run could take part in a five-run
// finder signature as though the mask had produced it.
fn clip_line(origin: vec2<f32>) -> vec3<i32> {
    var lo = -3.4e38;
    var hi = 3.4e38;
    let step = vec2<f32>(params.dx, params.dy);
    // Exclusive upper bounds: with floor addressing a sample is in frame when
    // its coordinate is in [0, dim), so the interval must be half-open. Using
    // dim-1 here excluded every sample whose coordinate fell in the last pixel,
    // which is 1/|component| samples at the far edge - two of them at 75
    // degrees, not the one that widening the span by a constant would cover.
    let limit = vec2<f32>(f32(params.width), f32(params.height));
    for (var axis = 0u; axis < 2u; axis++) {
        let p = origin[axis];
        let s = step[axis];
        if s == 0.0 {
            if p < 0.0 || p >= limit[axis] {
                return vec3<i32>(0, 0, 0);
            }
            continue;
        }
        var t0 = -p / s;
        var t1 = (limit[axis] - p) / s;
        if t0 > t1 {
            let swap = t0;
            t0 = t1;
            t1 = swap;
        }
        lo = max(lo, t0);
        hi = min(hi, t1);
    }
    // The half-open interval and the floor addressing now agree, so the trim
    // below only absorbs the rounding at each endpoint. It is kept because a
    // span that claims a sample the sampler rejects would break the contract
    // that every emitted run lies inside the frame.
    var start = max(i32(ceil(lo)) - 1, 0);
    var end = min(i32(floor(hi)) + 1, i32(params.line_length) - 1);
    loop {
        if start > end || in_frame(origin, start) {
            break;
        }
        start += 1;
    }
    loop {
        if end < start || in_frame(origin, end) {
            break;
        }
        end -= 1;
    }
    if end < start {
        return vec3<i32>(0, 0, 0);
    }
    return vec3<i32>(start, end, 1);
}

fn in_frame(origin: vec2<f32>, i: i32) -> bool {
    let p = floor(origin + f32(i) * vec2<f32>(params.dx, params.dy));
    let x = i32(p.x);
    let y = i32(p.y);
    return x >= 0 && x < i32(params.width) && y >= 0 && y < i32(params.height);
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
    let clip = clip_line(origin);

    if lane == 0u {
        emitted = 0u;
    }
    workgroupBarrier();
    if clip.z == 0 {
        if lane == 0u {
            boundary_counts[line * 3u + channel] = 0u;
        }
        return;
    }
    let span_start = clip.x;
    let span_end = clip.y;

    var block = span_start;
    loop {
        if block > span_end {
            break;
        }
        let i = block + i32(lane);
        let inside = i <= span_end;

        var value = 3u;
        if inside {
            value = sample_at(origin, i, channel);
        }
        values[lane] = value;
        workgroupBarrier();

        // The span's first sample opens the first run; every later one is a
        // boundary when it differs from its predecessor. Lanes 1..255 read that
        // predecessor from workgroup memory; only lane 0 crosses a block edge
        // and loads, and its predecessor is always inside the span because it
        // is only consulted when i > span_start.
        var starts = 0u;
        if inside {
            if i == span_start {
                starts = 1u;
            } else {
                var prev = 3u;
                if lane == 0u {
                    prev = sample_at(origin, i - 1, channel);
                } else {
                    prev = values[lane - 1u];
                }
                if value != prev {
                    starts = 1u;
                }
            }
        }
        workgroupBarrier();
        flags[lane] = starts;
        workgroupBarrier();
        let index = scan_inclusive(lane);
        let block_total = flags[WORKGROUP - 1u];

        if starts == 1u {
            let slot = emitted + index - 1u;
            if slot < params.run_capacity {
                boundaries[slot_base + slot] = u32(i);
            }
        }
        workgroupBarrier();
        if lane == 0u {
            emitted = emitted + block_total;
        }
        workgroupBarrier();
        block += i32(WORKGROUP);
    }

    // Terminal boundary one past the span's last sample, so a run length is
    // always the difference of consecutive entries. Without it the final run
    // has no closing index and every consumer would need the span's end as a
    // special case, which is exactly the kind of implicit contract that gets
    // read wrong once and silently drops the last run.
    if lane == 0u {
        if emitted < params.run_capacity {
            boundaries[slot_base + emitted] = u32(span_end + 1);
        }
        emitted = emitted + 1u;
    }
    workgroupBarrier();

    // The true count, not the clamped one. Writes past capacity were dropped,
    // so a count above capacity is the host's signal that this line's list is
    // incomplete; reporting the clamped value instead would make a truncated
    // result indistinguishable from a complete one.
    if lane == 0u {
        boundary_counts[line * 3u + channel] = emitted;
    }
}
