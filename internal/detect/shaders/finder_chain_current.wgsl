// Current-family fragment of the finder cross-check chain: appended to the
// shared prelude to form the current-family kernel module. One lane replays
// the complete per-hit chain of one raw green-row record and writes its
// fixed-slot outcome. Duplicated call sites of the CPU orchestration are
// folded into loops (the horizontal branch probes, the channel pair of the
// full cross-check, the core color checks) so the driver's pipeline compiler
// inlines each machine once; IEEE addition commutes and the folded checks
// are symmetric, so every computed value is bit-identical to the CPU chain's
// per-branch operand order.

struct CrossPat { cx: F64, cy: F64, ms: F64, dir: i32, ok: bool }

// cross_check_pattern mirrors crossCheckPattern for horizontal current-family
// candidates: the green channel and the type pair's second channel through
// one cross_check_pattern_ch site, then the core color check on the third
// channel in every direction.
fn cross_check_pattern(typ: i32, cx0: F64, cy0: F64, module_size0: F64, slack: i32) -> CrossPat {
    let module_size_max = sf_scale_pow2(module_size0, 1);
    var second_channel = 2u;
    var color_channel = 0u;
    var color_bit = chain_params.cross_color_bits & 1u;
    if typ == 1 || typ == 2 {
        second_channel = 0u;
        color_channel = 2u;
        color_bit = (chain_params.cross_color_bits >> 1u) & 1u;
    }
    var ms = array<F64, 2>(F64(0u, 0u), F64(0u, 0u));
    var cx = array<F64, 2>(F64(0u, 0u), F64(0u, 0u));
    var cy = array<F64, 2>(F64(0u, 0u), F64(0u, 0u));
    var dir = array<i32, 2>(0, 0);
    var dcc = array<i32, 2>(0, 0);
    for (var k = 0; k < 2; k++) {
        var channel = 1u;
        if k == 1 { channel = second_channel; }
        let res = cross_check_pattern_ch(channel, typ, module_size_max, cx0, cy0, slack);
        if !res.ok { return CrossPat(cx0, cy0, module_size0, 0, false); }
        ms[k] = res.ms;
        cx[k] = res.cx;
        cy[k] = res.cy;
        dir[k] = res.dir;
        dcc[k] = res.dcc;
    }
    if !check_module_size2(ms[0], ms[1]) { return CrossPat(cx0, cy0, module_size0, 0, false); }
    let msf = sf_scale_pow2(sf_add(ms[0], ms[1]), -1);
    let cxf = sf_scale_pow2(sf_add(cx[0], cx[1]), -1);
    let cyf = sf_scale_pow2(sf_add(cy[0], cy[1]), -1);
    for (var d: i32 = 0; d < 3; d++) {
        if !cross_check_color(color_channel, color_bit, sf_trunc_i32(msf), sf_trunc_i32(cxf), sf_trunc_i32(cyf), d, slack) {
            return CrossPat(cx0, cy0, module_size0, 0, false);
        }
    }
    var direction: i32 = -1;
    if dcc[0] == 2 || dcc[1] == 2 {
        direction = 2;
    } else if dir[0] + dir[1] > 0 {
        direction = 1;
    }
    return CrossPat(cxf, cyf, msf, direction, true);
}

// process_current_hit mirrors processCurrentFamilyHit for one raw green-row
// hit: the blue-then-red horizontal probes share one call site (the first
// success selects the branch, exactly the CPU's else-if), and both branches'
// core color checks run at the seed center and module through one site.
fn process_current_hit(y: i32, end_pos: i32, s2: i32, s3: i32, s4: i32, inside: i32) -> Outcome {
    var outc = zero_outcome();
    let w = i32(chain_params.width);
    let center_g = sf_sub(sf_from_i32(end_pos - s4 - s3), sf_scale_pow2(sf_from_i32(s2), -1));
    let module_g = sf_div_small(sf_from_i32(inside), 3u);
    let row_offset = y * w;

    let type_g = mask_bit_at(row_offset + sf_trunc_i32(center_g), 1u);
    let slack = chain_slack(module_g);
    let module_g_x2 = sf_scale_pow2(module_g, 1);

    // Branch 0 probes blue, branch 1 red.
    var branch: i32 = -1;
    var probe_center = center_g;
    var probe_ms = F64(0u, 0u);
    for (var b: i32 = 0; b < 2; b++) {
        var probe_channel = 2u;
        if b == 1 { probe_channel = 0u; }
        let h = cross_check_pattern_horizontal(probe_channel, module_g_x2, center_g, sf_from_i32(y), slack);
        if h.ok {
            branch = b;
            probe_center = h.centerx;
            probe_ms = h.ms;
            break;
        }
    }
    if branch < 0 { return outc; }
    var color_channel = 0u;
    var color_bit = chain_params.cross_color_bits & 1u;
    if branch == 1 {
        outc.flags = outc.flags | 2u; // branch red
        color_channel = 2u;
        color_bit = (chain_params.cross_color_bits >> 1u) & 1u;
    } else {
        outc.flags = outc.flags | 1u; // branch blue
    }
    if !cross_check_color(color_channel, color_bit, sf_trunc_i32(module_g), sf_trunc_i32(center_g), y, 0, slack) {
        return outc;
    }
    if branch == 1 {
        outc.flags = outc.flags | 4u; // red color
    }
    if !check_module_size2(module_g, probe_ms) { return outc; }
    let cx = sf_scale_pow2(sf_add(center_g, probe_center), -1);
    let ms = sf_scale_pow2(sf_add(module_g, probe_ms), -1);

    var type_r = 0u;
    var type_b = 0u;
    var t0: i32 = 0;
    var t1: i32 = 3;
    if branch == 1 {
        type_r = mask_bit_at(row_offset + sf_trunc_i32(probe_center), 0u);
        t0 = 1;
        t1 = 2;
    } else {
        type_b = mask_bit_at(row_offset + sf_trunc_i32(probe_center), 2u);
    }
    var typ: i32;
    if classify_match(chain_params.classify_current, t0, type_r, type_g, type_b) {
        typ = t0;
    } else if classify_match(chain_params.classify_current, t1, type_r, type_g, type_b) {
        typ = t1;
    } else {
        return outc;
    }
    if branch == 1 {
        outc.flags = outc.flags | 8u; // red classified
    }
    let pat = cross_check_pattern(typ, cx, sf_from_i32(y), ms, chain_slack(ms));
    if !pat.ok { return outc; }
    outc.flags = outc.flags | 16u; // survivor
    outc.typ = typ;
    outc.dir = pat.dir;
    outc.cx = pat.cx;
    outc.cy = pat.cy;
    outc.ms = pat.ms;
    return outc;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    // An overflowed pass leaves holes in the scattered walk-order records;
    // the host retries it with grown buffers, so no lane may read the
    // incomplete data (a stale hole could carry unbounded run lengths).
    if records.count > chain_params.capacity { return; }
    if idx >= chain_params.capacity || idx >= records.count { return; }
    let base = idx * 8u;
    if records.data[base] != 1u { return; }
    write_outcome(idx, process_current_hit(
        i32(records.data[base + 1u]),
        i32(records.data[base + 3u]),
        i32(records.data[base + 4u]),
        i32(records.data[base + 5u]),
        i32(records.data[base + 6u]),
        i32(records.data[base + 7u]),
    ));
}
