// Fused windows compacted by subgroup ballot: the measured fastest of the two,
// and the route this pipeline is built on where the device supports it.
//
// The workgroup scan costs log2(WORKGROUP) rounds and two barriers per round to
// answer one question: how many boundaries precede mine in this block. A
// subgroup answers it in one ballot and one countOneBits with no barrier,
// leaving only a scan across the handful of subgroups in the workgroup.
//
// It carries a portability condition the scan variant does not. The subgroup
// index is derived as local_invocation_index / subgroup_size, which is only
// meaningful when the workgroup is partitioned into full, linearly assigned
// subgroups. Vulkan guarantees that under a pipeline flag vulki cannot request,
// so the assumption is verified against the device instead of assumed - see the
// full-subgroup test - and the scan variant exists for anything that fails it.

const MAX_SUBGROUPS: u32 = WORKGROUP / 4u;

var<workgroup> subgroup_totals: array<u32, MAX_SUBGROUPS>;

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
    let clip = clip_line(origin);
    if clip.z == 0 {
        return;
    }
    let sg_index = lane / sg_size;
    let sg_count = WORKGROUP / sg_size;
    let key = line * 3u + channel;
    let span_start = clip.x;
    let span_end = clip.y;

    if lane == 0u {
        carry = 0u;
    }
    workgroupBarrier();

    var block = span_start;
    loop {
        if block > span_end {
            break;
        }
        let i = block + i32(lane);
        let starts = block_flag(origin, channel, i, span_start, i <= span_end, lane);

        let ballot = subgroupBallot(starts);
        let within = ballot_prefix(ballot, sg_lane);
        if sg_lane == 0u {
            subgroup_totals[sg_index] = ballot_count(ballot);
        }
        workgroupBarrier();
        var before = 0u;
        var n = 0u;
        for (var s = 0u; s < sg_count; s++) {
            if s < sg_index {
                before = before + subgroup_totals[s];
            }
            n = n + subgroup_totals[s];
        }
        if starts {
            bpos[5u + before + within] = u32(i);
        }
        workgroupBarrier();

        flush_block(key, n, lane);
        block += i32(WORKGROUP);
    }

    // The terminal boundary closes the last run, and closes the last windows
    // with it, so it goes through the same path as a one-boundary block.
    if lane == 0u {
        bpos[5u] = u32(span_end + 1);
    }
    workgroupBarrier();
    flush_block(key, 1u, lane);
}
