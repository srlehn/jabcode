// Shared prelude of the finder-pattern cross-check chain kernels: the
// run-length machines, the color check and the per-channel cross-check
// driver over the packed binary masks. Each finder family's chain is a
// separate kernel module built from softfloat64.wgsl plus this prelude plus
// its family fragment, so a build without a family compiles no trace of its
// chain and no module carries another family's code through the driver's
// pipeline compiler. The float64 arithmetic of the CPU chain is reproduced
// exactly by the integer softfloat routines of softfloat64.wgsl; every
// machine function here has a Go twin in gpu_finder_chain_ref_test.go; keep
// the arithmetic in sync (the kernels fold duplicated call sites into loops
// to bound pipeline compile time, computing the identical per-hit sequence).

struct ChainParams {
    width: u32,
    height: u32,
    capacity: u32,
    // Bit 0 selects the print-level slack rule of ccSlack.
    flags: u32,
    // Binarized palette bits of the four current-family finder cores,
    // three bits (R, G, B) per type at bit type*3.
    classify_current: u32,
    // The same table for the four BSI-era finder cores.
    classify_bsi: u32,
    // Bit 0: expected red-channel core bit of the blue-branch color check;
    // bit 1: expected blue-channel core bit of the red-branch color check.
    cross_color_bits: u32,
    pad: u32,
}

struct ScanRecords {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    data: array<u32>,
}

@group(0) @binding(0) var<storage, read> packed_masks: array<u32>;
@group(0) @binding(1) var<storage, read> records: ScanRecords;
@group(0) @binding(2) var<storage, read_write> outcomes: array<u32>;
@group(0) @binding(3) var<storage, read> chain_params: ChainParams;

// diag_length_const is float64(5) / (2.0 * 1.41421), crossCheckColor's
// diagonal length factor. Structured constants live behind functions:
// module-scope struct consts miscompile to zero on measured drivers.
fn diag_length_const() -> F64 { return F64(0x3ffc48cau, 0xab7554e4u); }

// half_f64 is 0.5, the rounding addend of the print-slack rule.
fn half_f64() -> F64 { return F64(0x3fe00000u, 0u); }

// mask_bit_at reads a binary mask bit; out-of-range indexes read as zero (the
// CPU chain never survives to read one on decodable inputs).
fn mask_bit_at(pixel: i32, channel: u32) -> u32 {
    if pixel < 0 || u32(pixel) >= chain_params.width * chain_params.height {
        return 0u;
    }
    let p = u32(pixel);
    let word = packed_masks[p / 8u];
    return (word >> ((p % 8u) * 3u + channel)) & 1u;
}

// chain_slack is ccSlack: the ported constant 3 normally, half a module in
// the print-level passes.
fn chain_slack(module_size: F64) -> i32 {
    if (chain_params.flags & 1u) != 0u {
        let s = sf_trunc_i32(sf_add(sf_scale_pow2(module_size, -1), half_f64()));
        return max(3, s);
    }
    return 3;
}

struct CrossMs { ms: F64, ok: bool }

// check_pattern_cross mirrors checkPatternCross through the softfloat ops.
fn check_pattern_cross(sc: array<i32, 5>) -> CrossMs {
    var s = sc;
    var inside = 0;
    for (var i = 1; i < 4; i++) {
        if s[i] == 0 { return CrossMs(F64(0u, 0u), false); }
        inside = inside + s[i];
    }
    let layer = sf_div_small(sf_from_i32(inside), 3u);
    let tol = sf_scale_pow2(layer, -1);
    let half_tol = sf_scale_pow2(tol, -1);
    let ok = sf_less(sf_abs(sf_sub(layer, sf_from_i32(s[1]))), tol) &&
        sf_less(sf_abs(sf_sub(layer, sf_from_i32(s[2]))), tol) &&
        sf_less(sf_abs(sf_sub(layer, sf_from_i32(s[3]))), tol) &&
        sf_less(half_tol, sf_from_i32(s[0])) &&
        sf_less(half_tol, sf_from_i32(s[4])) &&
        sf_less(sf_abs(sf_from_i32(s[1] - s[3])), tol);
    return CrossMs(layer, ok);
}

fn check_module_size2(s1: F64, s2: F64) -> bool {
    let mean = sf_scale_pow2(sf_add(s1, s2), -1);
    let tol = sf_div_small(sf_scale_pow2(mean, 1), 5u);
    return sf_less(sf_abs(sf_sub(mean, s1)), tol) && sf_less(sf_abs(sf_sub(mean, s2)), tol);
}

struct CrossV { centery: F64, ms: F64, ok: bool }

// cross_check_pattern_vertical mirrors crossCheckPatternVertical.
fn cross_check_pattern_vertical(
    channel: u32, module_size_max: i32, centerx: F64, centery: F64, slack: i32,
) -> CrossV {
    var sc = array<i32, 5>(0, 0, 0, 0, 0);
    let w = i32(chain_params.width);
    let h = i32(chain_params.height);
    let cx = sf_trunc_i32(centerx);
    let cy = sf_trunc_i32(centery);

    var i: i32 = 1;
    var state_index: i32 = 0;
    sc[1] = sc[1] + 1;
    loop {
        if !(i <= cy && state_index <= 2) { break; }
        if mask_bit_at((cy - i) * w + cx, channel) == mask_bit_at((cy - (i - 1)) * w + cx, channel) {
            sc[2 - state_index] = sc[2 - state_index] + 1;
        } else if state_index > 0 && sc[2 - state_index] < slack {
            sc[2 - (state_index - 1)] = sc[2 - (state_index - 1)] + sc[2 - state_index];
            sc[2 - state_index] = 0;
            state_index = state_index - 1;
            sc[2 - state_index] = sc[2 - state_index] + 1;
        } else {
            state_index = state_index + 1;
            if state_index > 2 { break; }
            sc[2 - state_index] = sc[2 - state_index] + 1;
        }
        continuing { i = i + 1; }
    }
    if state_index < 2 { return CrossV(centery, F64(0u, 0u), false); }
    state_index = 0;
    i = 1;
    loop {
        if !(cy + i < h && state_index <= 2) { break; }
        if mask_bit_at((cy + i) * w + cx, channel) == mask_bit_at((cy + (i - 1)) * w + cx, channel) {
            sc[2 + state_index] = sc[2 + state_index] + 1;
        } else if state_index > 0 && sc[2 + state_index] < slack {
            sc[2 + (state_index - 1)] = sc[2 + (state_index - 1)] + sc[2 + state_index];
            sc[2 + state_index] = 0;
            state_index = state_index - 1;
            sc[2 + state_index] = sc[2 + state_index] + 1;
        } else {
            state_index = state_index + 1;
            if state_index > 2 { break; }
            sc[2 + state_index] = sc[2 + state_index] + 1;
        }
        continuing { i = i + 1; }
    }
    if state_index < 2 { return CrossV(centery, F64(0u, 0u), false); }
    let cross = check_pattern_cross(sc);
    if cross.ok && sf_less_eq(cross.ms, sf_from_i32(module_size_max)) {
        let new_cy = sf_sub(sf_from_i32(cy + i - sc[4] - sc[3]), sf_scale_pow2(sf_from_i32(sc[2]), -1));
        return CrossV(new_cy, cross.ms, true);
    }
    return CrossV(centery, cross.ms, false);
}

struct CrossH { centerx: F64, ms: F64, ok: bool }

// cross_check_pattern_horizontal mirrors crossCheckPatternHorizontal.
fn cross_check_pattern_horizontal(
    channel: u32, module_size_max: F64, centerx: F64, centery: F64, slack: i32,
) -> CrossH {
    var sc = array<i32, 5>(0, 0, 0, 0, 0);
    let w = i32(chain_params.width);
    let startx = sf_trunc_i32(centerx);
    let row_offset = sf_trunc_i32(centery) * w;

    var i: i32 = 1;
    var state_index: i32 = 0;
    sc[2] = sc[2] + 1;
    loop {
        if !(i <= startx && state_index <= 2) { break; }
        if mask_bit_at(row_offset + (startx - i), channel) == mask_bit_at(row_offset + (startx - (i - 1)), channel) {
            sc[2 - state_index] = sc[2 - state_index] + 1;
        } else if state_index > 0 && sc[2 - state_index] < slack {
            sc[2 - (state_index - 1)] = sc[2 - (state_index - 1)] + sc[2 - state_index];
            sc[2 - state_index] = 0;
            state_index = state_index - 1;
            sc[2 - state_index] = sc[2 - state_index] + 1;
        } else {
            state_index = state_index + 1;
            if state_index > 2 { break; }
            sc[2 - state_index] = sc[2 - state_index] + 1;
        }
        continuing { i = i + 1; }
    }
    if state_index < 2 { return CrossH(centerx, F64(0u, 0u), false); }
    state_index = 0;
    i = 1;
    loop {
        if !(startx + i < w && state_index <= 2) { break; }
        if mask_bit_at(row_offset + (startx + i), channel) == mask_bit_at(row_offset + (startx + (i - 1)), channel) {
            sc[2 + state_index] = sc[2 + state_index] + 1;
        } else if state_index > 0 && sc[2 + state_index] < slack {
            sc[2 + (state_index - 1)] = sc[2 + (state_index - 1)] + sc[2 + state_index];
            sc[2 + state_index] = 0;
            state_index = state_index - 1;
            sc[2 + state_index] = sc[2 + state_index] + 1;
        } else {
            state_index = state_index + 1;
            if state_index > 2 { break; }
            sc[2 + state_index] = sc[2 + state_index] + 1;
        }
        continuing { i = i + 1; }
    }
    if state_index < 2 { return CrossH(centerx, F64(0u, 0u), false); }
    let cross = check_pattern_cross(sc);
    if cross.ok && sf_less_eq(cross.ms, module_size_max) {
        let new_cx = sf_sub(sf_from_i32(startx + i - sc[4] - sc[3]), sf_scale_pow2(sf_from_i32(sc[2]), -1));
        return CrossH(new_cx, cross.ms, true);
    }
    return CrossH(centerx, cross.ms, false);
}

struct CrossD { cx: F64, cy: F64, ms: F64, confirmed: i32, dir: i32 }

// cross_check_pattern_diagonal mirrors crossCheckPatternDiagonal, including
// its retry flips and the module-size write of a failed second try.
fn cross_check_pattern_diagonal(
    channel: u32, typ: i32, module_size_max: F64,
    centerx0: F64, centery0: F64, module_size0: F64,
    dir0: i32, both_dir: bool, slack: i32,
) -> CrossD {
    var centerx = centerx0;
    var centery = centery0;
    var module_size = module_size0;
    var dir = dir0;
    let w = i32(chain_params.width);
    let h = i32(chain_params.height);
    var offset_x: i32;
    var offset_y: i32;
    var fix_dir = false;
    if dir != 0 {
        if dir > 0 {
            offset_x = -1; offset_y = -1; dir = 1;
        } else {
            offset_x = 1; offset_y = -1; dir = -1;
        }
        fix_dir = true;
    } else if typ == 0 || typ == 1 {
        offset_x = -1; offset_y = -1; dir = 1;
    } else {
        offset_x = 1; offset_y = -1; dir = -1;
    }

    var confirmed: i32 = 0;
    var try_count: i32 = 0;
    var tmp_module_size = F64(0u, 0u);
    loop {
        var flag = false;
        try_count = try_count + 1;
        var i: i32 = 0;
        var state_index: i32 = 0;
        var sc = array<i32, 5>(0, 0, 0, 0, 0);
        let startx = sf_trunc_i32(centerx);
        let starty = sf_trunc_i32(centery);

        sc[2] = sc[2] + 1;
        var j: i32 = 1;
        loop {
            if !(starty + j * offset_y >= 0 && starty + j * offset_y < h &&
                startx + j * offset_x >= 0 && startx + j * offset_x < w &&
                state_index <= 2) { break; }
            if mask_bit_at((starty + j * offset_y) * w + (startx + j * offset_x), channel) ==
                mask_bit_at((starty + (j - 1) * offset_y) * w + (startx + (j - 1) * offset_x), channel) {
                sc[2 - state_index] = sc[2 - state_index] + 1;
            } else if state_index > 0 && sc[2 - state_index] < slack {
                sc[2 - (state_index - 1)] = sc[2 - (state_index - 1)] + sc[2 - state_index];
                sc[2 - state_index] = 0;
                state_index = state_index - 1;
                sc[2 - state_index] = sc[2 - state_index] + 1;
            } else {
                state_index = state_index + 1;
                if state_index > 2 { break; }
                sc[2 - state_index] = sc[2 - state_index] + 1;
            }
            continuing { j = j + 1; }
        }
        if state_index < 2 {
            if try_count == 1 {
                flag = true;
                offset_x = -offset_x;
                dir = -dir;
            } else {
                return CrossD(centerx, centery, module_size, confirmed, dir);
            }
        }

        if !flag {
            state_index = 0;
            i = 1;
            loop {
                if !(starty - i * offset_y >= 0 && starty - i * offset_y < h &&
                    startx - i * offset_x >= 0 && startx - i * offset_x < w &&
                    state_index <= 2) { break; }
                if mask_bit_at((starty - i * offset_y) * w + (startx - i * offset_x), channel) ==
                    mask_bit_at((starty - (i - 1) * offset_y) * w + (startx - (i - 1) * offset_x), channel) {
                    sc[2 + state_index] = sc[2 + state_index] + 1;
                } else if state_index > 0 && sc[2 + state_index] < slack {
                    sc[2 + (state_index - 1)] = sc[2 + (state_index - 1)] + sc[2 + state_index];
                    sc[2 + state_index] = 0;
                    state_index = state_index - 1;
                    sc[2 + state_index] = sc[2 + state_index] + 1;
                } else {
                    state_index = state_index + 1;
                    if state_index > 2 { break; }
                    sc[2 + state_index] = sc[2 + state_index] + 1;
                }
                continuing { i = i + 1; }
            }
            if state_index < 2 {
                if try_count == 1 {
                    flag = true;
                    offset_x = -offset_x;
                    dir = -dir;
                } else {
                    return CrossD(centerx, centery, module_size, confirmed, dir);
                }
            }
        }

        if !flag {
            let cross = check_pattern_cross(sc);
            module_size = cross.ms;
            if cross.ok && sf_less_eq(module_size, module_size_max) {
                if sf_less(F64(0u, 0u), tmp_module_size) {
                    module_size = sf_scale_pow2(sf_add(module_size, tmp_module_size), -1);
                } else {
                    tmp_module_size = module_size;
                }
                centerx = sf_sub(sf_from_i32(startx + i - sc[4] - sc[3]), sf_scale_pow2(sf_from_i32(sc[2]), -1));
                centery = sf_sub(sf_from_i32(starty + i - sc[4] - sc[3]), sf_scale_pow2(sf_from_i32(sc[2]), -1));
                confirmed = confirmed + 1;
                if !both_dir || try_count == 2 || fix_dir {
                    if confirmed == 2 { dir = 2; }
                    return CrossD(centerx, centery, module_size, confirmed, dir);
                }
            } else {
                offset_x = -offset_x;
                dir = -dir;
            }
        }
        if !(try_count < 2 && !fix_dir) { break; }
    }
    return CrossD(centerx, centery, module_size, confirmed, dir);
}

// cross_check_color mirrors crossCheckColor with moduleNumber fixed at 5,
// the only value the chain uses. color_bit is the expected mask bit.
fn cross_check_color(
    channel: u32, color_bit: u32,
    module_size: i32, centerx: i32, centery: i32, dir_mode: i32, tol: i32,
) -> bool {
    let w = i32(chain_params.width);
    let h = i32(chain_params.height);
    if centerx < 0 || centerx >= w || centery < 0 || centery >= h { return false; }
    if dir_mode == 0 {
        let length = module_size * 4;
        let startx = max(centerx - length / 2, 0);
        var unmatch: i32 = 0;
        for (var j = startx; j < startx + length && j < w; j++) {
            if mask_bit_at(centery * w + j, channel) != color_bit {
                unmatch = unmatch + 1;
            } else if unmatch <= tol {
                unmatch = 0;
            }
            if unmatch > tol { return false; }
        }
        return true;
    }
    if dir_mode == 1 {
        let length = module_size * 4;
        let starty = max(centery - length / 2, 0);
        var unmatch: i32 = 0;
        for (var i = starty; i < starty + length && i < h; i++) {
            if mask_bit_at(w * i + centerx, channel) != color_bit {
                unmatch = unmatch + 1;
            } else if unmatch <= tol {
                unmatch = 0;
            }
            if unmatch > tol { return false; }
        }
        return true;
    }
    if dir_mode == 2 {
        let offset = sf_trunc_i32(sf_mul_u16(u32(module_size), diag_length_const()));
        let length = offset * 2;
        var unmatch: i32 = 0;
        var startx = max(centerx - offset, 0);
        var starty = max(centery - offset, 0);
        for (var i = 0; i < length && starty + i < h && startx + i < w; i++) {
            if mask_bit_at(w * (starty + i) + (startx + i), channel) != color_bit {
                unmatch = unmatch + 1;
            } else if unmatch <= tol {
                unmatch = 0;
            }
            if unmatch > tol { break; }
        }
        if unmatch < tol { return true; }
        unmatch = 0;
        startx = max(centerx - offset, 0);
        starty = min(centery + offset, h - 1);
        for (var i = 0; i < length && starty - i >= 0 && startx + i < w; i++) {
            if mask_bit_at(w * (starty - i) + (startx + i), channel) != color_bit {
                unmatch = unmatch + 1;
            } else if unmatch <= tol {
                unmatch = 0;
            }
            if unmatch > tol { return false; }
        }
        return true;
    }
    return false;
}

struct CrossCh { ms: F64, cx: F64, cy: F64, dir: i32, dcc: i32, ok: bool }

// cross_check_pattern_ch mirrors crossCheckPatternCh for horizontal
// candidates (hv 0), the only orientation the device chain replays.
fn cross_check_pattern_ch(
    channel: u32, typ: i32, module_size_max: F64, centerx: F64, centery: F64, slack: i32,
) -> CrossCh {
    var cx = centerx;
    var cy = centery;
    var ms_v = F64(0u, 0u);
    var ms_h = F64(0u, 0u);
    var ms_d = F64(0u, 0u);
    var dir: i32 = 0;
    var vcc = false;
    let v = cross_check_pattern_vertical(channel, sf_trunc_i32(module_size_max), cx, cy, slack);
    if v.ok {
        vcc = true;
        cy = v.centery;
        ms_v = v.ms;
        let hres = cross_check_pattern_horizontal(channel, module_size_max, cx, cy, slack);
        if !hres.ok { return CrossCh(F64(0u, 0u), cx, cy, dir, 0, false); }
        cx = hres.centerx;
        ms_h = hres.ms;
    }
    let d = cross_check_pattern_diagonal(channel, typ, module_size_max, cx, cy, ms_d, dir, !vcc, slack);
    let dcc = d.confirmed;
    cx = d.cx;
    cy = d.cy;
    ms_d = d.ms;
    dir = d.dir;
    if vcc && dcc > 0 {
        let ms = sf_div_small(sf_add(sf_add(ms_v, ms_h), ms_d), 3u);
        return CrossCh(ms, cx, cy, dir, dcc, true);
    }
    if dcc == 2 {
        let hres = cross_check_pattern_horizontal(channel, module_size_max, cx, cy, slack);
        if !hres.ok { return CrossCh(F64(0u, 0u), cx, cy, dir, dcc, false); }
        cx = hres.centerx;
        ms_h = hres.ms;
        let ms = sf_div_small(sf_add(ms_h, sf_scale_pow2(ms_d, 1)), 3u);
        return CrossCh(ms, cx, cy, dir, dcc, true);
    }
    return CrossCh(F64(0u, 0u), cx, cy, dir, dcc, false);
}

// classify_match tests one finder type's palette bits from a 12-bit table.
fn classify_match(table: u32, t: i32, type_r: u32, type_g: u32, type_b: u32) -> bool {
    let bits = table >> (u32(t) * 3u);
    return type_r == (bits & 1u) && type_g == ((bits >> 1u) & 1u) && type_b == ((bits >> 2u) & 1u);
}

struct Outcome { flags: u32, typ: i32, dir: i32, cx: F64, cy: F64, ms: F64 }

fn zero_outcome() -> Outcome {
    return Outcome(0u, 0, 0, F64(0u, 0u), F64(0u, 0u), F64(0u, 0u));
}

// write_outcome stores one hit's outcome in its fixed record slot. Each
// family kernel writes only its own channel's slots, so concurrently
// dispatched family chains never touch the same record.
fn write_outcome(idx: u32, outc: Outcome) {
    let slot = idx * 10u;
    outcomes[slot] = outc.flags;
    outcomes[slot + 1u] = u32(outc.typ);
    outcomes[slot + 2u] = bitcast<u32>(outc.dir);
    outcomes[slot + 3u] = outc.cx.hi;
    outcomes[slot + 4u] = outc.cx.lo;
    outcomes[slot + 5u] = outc.cy.hi;
    outcomes[slot + 6u] = outc.cy.lo;
    outcomes[slot + 7u] = outc.ms.hi;
    outcomes[slot + 8u] = outc.ms.lo;
    outcomes[slot + 9u] = 0u;
}
