// Prototype 1: boundary extraction compacted by a workgroup Hillis-Steele scan.
//
// The serial baseline in finder_line_scan.wgsl makes one lane responsible for a
// whole scan line, which wastes the machine twice over. The work per line is
// unbounded and uneven, so lanes in a workgroup diverge for as long as the
// longest line takes, and the state machine forces every sample of a line to be
// visited in order by one lane. Neither is inherent. What that machine actually
// computes is where the sampled bit changes - the run boundaries - and whether
// sample i is a boundary depends on sample i-1 and nothing else.
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

@group(0) @binding(1) var<storage, read_write> boundaries: array<u32>;
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
    let origin = line_origin(line);
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
                boundaries[slot_base + slot] = bitcast<u32>(i);
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
            boundaries[slot_base + emitted] = bitcast<u32>(span_end + 1);
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
