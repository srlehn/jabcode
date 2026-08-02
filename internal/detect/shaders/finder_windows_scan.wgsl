// Fused windows compacted by a workgroup Hillis-Steele scan: the same kernel as
// the ballot variant, and the same output, using nothing beyond core WGSL.
//
// It is slower - the scan costs log2(WORKGROUP) rounds and two barriers each
// where a ballot costs one instruction - but it is the fallback that makes the
// fused design deployable rather than adapter-specific. The boundary prototypes
// cannot serve that purpose: falling back to them would reinstate the per-line
// slot layout this design exists to remove, trading a portability problem for a
// hundred megabytes per angle.

var<workgroup> flags: array<u32, WORKGROUP>;

// Hillis-Steele inclusive scan over flags.
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
    let clip = clip_line(origin);
    if clip.z == 0 {
        return;
    }
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

        workgroupBarrier();
        flags[lane] = select(0u, 1u, starts);
        workgroupBarrier();
        let index = scan_inclusive(lane);
        let n = flags[WORKGROUP - 1u];
        if starts {
            bpos[5u + index - 1u] = u32(i);
        }
        workgroupBarrier();

        flush_block(origin, channel, key, n, lane);
        block += i32(WORKGROUP);
    }

    if lane == 0u {
        bpos[5u] = u32(span_end + 1);
    }
    workgroupBarrier();
    flush_block(origin, channel, key, 1u, lane);
}
