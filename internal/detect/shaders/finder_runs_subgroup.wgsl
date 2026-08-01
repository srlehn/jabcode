// Prototype 2: the same boundary extraction and the same output contract as
// finder_runs_hillis.wgsl, compacted with subgroup ballot and a bit-count
// prefix instead of a workgroup scan.
//
// The Hillis-Steele scan costs log2(WORKGROUP) rounds and two workgroup
// barriers per round to answer one question: how many boundaries precede mine
// in this block. A subgroup answers it in one ballot and one countOneBits, with
// no barrier at all, leaving only a scan across the handful of subgroups in the
// workgroup. Whether that difference survives the memory traffic of the mask
// reads is the whole reason both exist.
//
// Everything else is deliberately identical to prototype 1, including the
// workgroup array that shares each lane's sample with its right-hand neighbour,
// so the measurement isolates the compaction primitive rather than comparing
// two differently written kernels.
//
// Portability condition the host must establish, not assume: the subgroup index
// is derived as local_invocation_index / subgroup_size, which is meaningful only
// when the workgroup is partitioned into full, linearly assigned subgroups.
// RequireFullSubgroups gets the "full" half; Vulkan defines no relationship
// between SubgroupLocalInvocationId and LocalInvocationIndex, so the "linearly
// assigned" half is measured on the device before this kernel is selected. See
// subgroupLayoutUsable.


@group(0) @binding(1) var<storage, read_write> boundaries: array<u32>;
@group(0) @binding(3) var<storage, read_write> boundary_counts: array<u32>;

const WORKGROUP: u32 = 256u;
// Smallest subgroup any target reports, so the per-subgroup total array is
// always long enough however the workgroup is partitioned.
const MAX_SUBGROUPS: u32 = WORKGROUP / 4u;

var<workgroup> values: array<u32, WORKGROUP>;
var<workgroup> subgroup_totals: array<u32, MAX_SUBGROUPS>;
var<workgroup> emitted: u32;

fn ballot_count(b: vec4<u32>) -> u32 {
    return countOneBits(b.x) + countOneBits(b.y) + countOneBits(b.z) + countOneBits(b.w);
}

// ballot_prefix counts set bits strictly below id. The shift is never 32, which
// would be undefined: the middle branch only runs when id - base is 1..31, and
// id == base contributes nothing.
fn ballot_prefix(b: vec4<u32>, id: u32) -> u32 {
    var total = 0u;
    for (var w = 0u; w < 4u; w++) {
        let base = w * 32u;
        if id >= base + 32u {
            total = total + countOneBits(b[w]);
        } else if id > base {
            total = total + countOneBits(b[w] & ((1u << (id - base)) - 1u));
        }
    }
    return total;
}

@compute @workgroup_size(WORKGROUP)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
    @builtin(subgroup_size) sg_size: u32,
    @builtin(subgroup_invocation_id) sg_lane: u32,
) {
    let line = group.x;
    let channel = group.y;
    if line >= params.line_count || (params.channel_mask & (1u << channel)) == 0u {
        return;
    }
    let origin = line_origin(line);
    let slot_base = (line * 3u + channel) * params.run_capacity;
    let clip = clip_line(origin);
    let sg_index = lane / sg_size;
    let sg_count = WORKGROUP / sg_size;

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

        var starts = false;
        if inside {
            if i == span_start {
                starts = true;
            } else {
                var prev = 3u;
                if lane == 0u {
                    prev = sample_at(origin, i - 1, channel);
                } else {
                    prev = values[lane - 1u];
                }
                starts = value != prev;
            }
        }

        // One ballot replaces the whole scan inside a subgroup: the prefix is a
        // masked population count, and the subgroup's own total is the full one.
        let ballot = subgroupBallot(starts);
        let within = ballot_prefix(ballot, sg_lane);
        if sg_lane == 0u {
            subgroup_totals[sg_index] = ballot_count(ballot);
        }
        workgroupBarrier();

        // Only the handful of subgroup totals still needs a scan, and it is
        // short enough to sum in place: eight entries on a 32-lane subgroup.
        var before = 0u;
        for (var s = 0u; s < sg_count; s++) {
            if s < sg_index {
                before = before + subgroup_totals[s];
            }
        }
        var block_total = 0u;
        for (var s = 0u; s < sg_count; s++) {
            block_total = block_total + subgroup_totals[s];
        }

        if starts {
            let slot = emitted + before + within;
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

    if lane == 0u {
        if emitted < params.run_capacity {
            boundaries[slot_base + emitted] = u32(span_end + 1);
        }
        emitted = emitted + 1u;
    }
    workgroupBarrier();
    if lane == 0u {
        boundary_counts[line * 3u + channel] = emitted;
    }
}
