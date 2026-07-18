// BSI-family fragment of the finder cross-check chain: appended to the
// shared prelude to form the BSI kernel module, present only in builds whose
// decoder compiles the BSI family in. One lane replays the complete per-hit
// chain of one raw red-row record and writes its fixed-slot outcome. The
// horizontal probes of the green and blue channels share one call site, as
// the three-channel full cross-check already does.

fn check_module_size3(r: F64, g: F64, b: F64) -> bool {
    let mean = sf_div_small(sf_add(sf_add(r, g), b), 3u);
    let tol = sf_div_small(sf_scale_pow2(mean, 1), 5u);
    return sf_less(sf_abs(sf_sub(mean, r)), tol) &&
        sf_less(sf_abs(sf_sub(mean, g)), tol) &&
        sf_less(sf_abs(sf_sub(mean, b)), tol);
}

struct CrossPatBSI { cx: F64, cy: F64, ms: F64, dir: i32, ok: bool }

// cross_check_pattern_bsi mirrors crossCheckPatternBSIFamily for horizontal
// candidates (hv 0).
fn cross_check_pattern_bsi(typ: i32, cx0: F64, cy0: F64, module_size0: F64, slack: i32) -> CrossPatBSI {
    let module_size_max = sf_scale_pow2(module_size0, 1);
    var module_size = array<F64, 3>(F64(0u, 0u), F64(0u, 0u), F64(0u, 0u));
    var center_x = array<F64, 3>(F64(0u, 0u), F64(0u, 0u), F64(0u, 0u));
    var center_y = array<F64, 3>(F64(0u, 0u), F64(0u, 0u), F64(0u, 0u));
    var direction = array<i32, 3>(0, 0, 0);
    var diagonal = array<i32, 3>(0, 0, 0);
    for (var c: i32 = 0; c < 3; c++) {
        let res = cross_check_pattern_ch(u32(c), typ, module_size_max, cx0, cy0, slack);
        if !res.ok { return CrossPatBSI(cx0, cy0, module_size0, 0, false); }
        module_size[c] = res.ms;
        center_x[c] = res.cx;
        center_y[c] = res.cy;
        direction[c] = res.dir;
        diagonal[c] = res.dcc;
    }
    if !check_module_size3(module_size[0], module_size[1], module_size[2]) {
        return CrossPatBSI(cx0, cy0, module_size0, 0, false);
    }
    let ms = sf_div_small(sf_add(sf_add(module_size[0], module_size[1]), module_size[2]), 3u);
    let cx = sf_div_small(sf_add(sf_add(center_x[0], center_x[1]), center_x[2]), 3u);
    let cy = sf_div_small(sf_add(sf_add(center_y[0], center_y[1]), center_y[2]), 3u);
    var dir: i32 = -1;
    if diagonal[0] == 2 || diagonal[1] == 2 || diagonal[2] == 2 {
        dir = 2;
    } else if direction[0] + direction[1] + direction[2] > 0 {
        dir = 1;
    }
    return CrossPatBSI(cx, cy, ms, dir, true);
}

// process_bsi_hit mirrors processBSIFamilyHit for one raw red-row hit; the
// green and blue horizontal probes share one call site in a loop with the
// CPU's early exit.
fn process_bsi_hit(y: i32, end_pos: i32, s2: i32, s3: i32, s4: i32, inside: i32) -> Outcome {
    var outc = zero_outcome();
    let w = i32(chain_params.width);
    let center0 = sf_sub(sf_from_i32(end_pos - s4 - s3), sf_scale_pow2(sf_from_i32(s2), -1));
    let module0 = sf_div_small(sf_from_i32(inside), 3u);
    let row_offset = y * w;
    let slack = chain_slack(module0);
    let module0_x2 = sf_scale_pow2(module0, 1);

    var center = array<F64, 3>(center0, center0, center0);
    var module_size = array<F64, 3>(module0, F64(0u, 0u), F64(0u, 0u));
    for (var c: i32 = 1; c < 3; c++) {
        let h = cross_check_pattern_horizontal(u32(c), module0_x2, center[c], sf_from_i32(y), slack);
        if !h.ok { return outc; }
        center[c] = h.centerx;
        module_size[c] = h.ms;
    }
    if !check_module_size3(module_size[0], module_size[1], module_size[2]) { return outc; }

    let cx = sf_div_small(sf_add(sf_add(center[0], center[1]), center[2]), 3u);
    let ms = sf_div_small(sf_add(sf_add(module_size[0], module_size[1]), module_size[2]), 3u);
    let type_r = mask_bit_at(row_offset + sf_trunc_i32(center[0]), 0u);
    let type_g = mask_bit_at(row_offset + sf_trunc_i32(center[1]), 1u);
    let type_b = mask_bit_at(row_offset + sf_trunc_i32(center[2]), 2u);
    var typ: i32 = -1;
    for (var t: i32 = 0; t < 4; t++) {
        if classify_match(chain_params.classify_bsi, t, type_r, type_g, type_b) {
            typ = t;
            break;
        }
    }
    if typ < 0 { return outc; }
    let pat = cross_check_pattern_bsi(typ, cx, sf_from_i32(y), ms, chain_slack(ms));
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
    if records.data[base] != 0u { return; }
    write_outcome(idx, process_bsi_hit(
        i32(records.data[base + 1u]),
        i32(records.data[base + 3u]),
        i32(records.data[base + 4u]),
        i32(records.data[base + 5u]),
        i32(records.data[base + 6u]),
        i32(records.data[base + 7u]),
    ));
}
