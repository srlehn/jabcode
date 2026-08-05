// BSI-family fragment of the finder cross-check chain: appended to the
// shared prelude to form the BSI kernel module, present only in builds whose
// decoder compiles the BSI family in. One lane replays the complete per-hit
// chain of one raw red-row record and writes its fixed-slot outcome. The
// horizontal probes of the green and blue channels share one call site, as
// the three-channel full cross-check already does.

struct CrossPatBSI { cx: f32, cy: f32, ms: f32, dir: i32, ok: bool }

// cross_check_pattern_bsi mirrors crossCheckPatternBSIFamily for horizontal
// candidates (hv 0).
fn cross_check_pattern_bsi(typ: i32, cx0: f32, cy0: f32, module_size0: f32, slack: i32) -> CrossPatBSI {
    let module_size_max = (module_size0 * 2.0);
    var module_size = array<f32, 3>(0.0, 0.0, 0.0);
    var center_x = array<f32, 3>(0.0, 0.0, 0.0);
    var center_y = array<f32, 3>(0.0, 0.0, 0.0);
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
    let ms = (((module_size[0] + module_size[1]) + module_size[2]) / f32(3u));
    let cx = (((center_x[0] + center_x[1]) + center_x[2]) / f32(3u));
    let cy = (((center_y[0] + center_y[1]) + center_y[2]) / f32(3u));
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
    let center0 = (f32(end_pos - s4 - s3) - (f32(s2) * 0.5));
    let module0 = (f32(inside) / f32(3u));
    let row_offset = y * w;
    let slack = chain_slack(module0);
    let module0_x2 = (module0 * 2.0);

    var center = array<f32, 3>(center0, center0, center0);
    var module_size = array<f32, 3>(module0, 0.0, 0.0);
    for (var c: i32 = 1; c < 3; c++) {
        let h = cross_check_pattern_horizontal(u32(c), module0_x2, center[c], f32(y), slack);
        if !h.ok { return outc; }
        center[c] = h.centerx;
        module_size[c] = h.ms;
    }
    if !check_module_size3(module_size[0], module_size[1], module_size[2]) { return outc; }

    let cx = (((center[0] + center[1]) + center[2]) / f32(3u));
    let ms = (((module_size[0] + module_size[1]) + module_size[2]) / f32(3u));
    let type_r = mask_bit_at(row_offset + i32(center[0]), 0u);
    let type_g = mask_bit_at(row_offset + i32(center[1]), 1u);
    let type_b = mask_bit_at(row_offset + i32(center[2]), 2u);
    var typ: i32 = -1;
    for (var t: i32 = 0; t < 4; t++) {
        if classify_match(chain_params.classify_bsi, t, type_r, type_g, type_b) {
            typ = t;
            break;
        }
    }
    if typ < 0 { return outc; }
    let pat = cross_check_pattern_bsi(typ, cx, f32(y), ms, chain_slack(ms));
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
    // The host retries an overflowed pass with grown buffers, so skip chain
    // work whose outcomes would be discarded.
    if records.count > chain_params.capacity { return; }
    if idx >= chain_params.capacity || idx >= records.count { return; }
    let base = idx * 8u;
    if records.data[base] != 0u { return; }
    let y = records.data[base + 1u];
    // A row the consumer's walk never visits contributes to nothing, so it is
    // dropped here rather than counted and then filtered on the host.
    if row_stride_skips(y) { return; }
    let outc = process_bsi_hit(
        i32(y),
        i32(records.data[base + 3u]),
        i32(records.data[base + 4u]),
        i32(records.data[base + 5u]),
        i32(records.data[base + 6u]),
        i32(records.data[base + 7u]),
    );
    write_outcome(idx, outc);
    summarize_row(0u, y, records.data[base + 2u], base, outc, row_module_size(records.data[base + 7u]));
}
