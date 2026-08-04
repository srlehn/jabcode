// Shared arithmetic of the finder-pattern cross-check chain kernels: the
// run-length machines, color check and per-channel driver over packed masks.
// Binding declarations are prepended separately because row and directional
// records have different layouts. The arithmetic is native f32: the host chain
// runs float64, so device decisions can differ at a boundary, and the gate for
// that is decoded corpus rows rather than agreement with host rounding. Every
// machine function here has a Go twin in gpu_finder_chain_ref_test.go.

// diag_length_const is float64(5) / (2.0 * 1.41421), crossCheckColor's
// diagonal length factor. Structured constants live behind functions:
// module-scope struct consts miscompile to zero on measured drivers.
fn diag_length_const() -> f32 { return 1.7677713632583618; }



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
fn chain_slack(module_size: f32) -> i32 {
    if (chain_params.flags & 1u) != 0u {
        let s = i32(module_size * 0.5 + 0.5);
        return max(3, s);
    }
    return 3;
}

struct CrossMs { ms: f32, ok: bool }

// check_pattern_cross mirrors checkPatternCross.
fn check_pattern_cross(sc: array<i32, 5>) -> CrossMs {
    var s = sc;
    var inside = 0;
    for (var i = 1; i < 4; i++) {
        if s[i] == 0 { return CrossMs(0.0, false); }
        inside = inside + s[i];
    }
    let layer = (f32(inside) / f32(3u));
    let tol = (layer * 0.5);
    let half_tol = (tol * 0.5);
    let ok = (abs((layer - f32(s[1]))) < tol) &&
        (abs((layer - f32(s[2]))) < tol) &&
        (abs((layer - f32(s[3]))) < tol) &&
        (half_tol < f32(s[0])) &&
        (half_tol < f32(s[4])) &&
        (abs(f32(s[1] - s[3])) < tol);
    return CrossMs(layer, ok);
}

fn check_module_size2(s1: f32, s2: f32) -> bool {
    let mean = ((s1 + s2) * 0.5);
    let tol = ((mean * 2.0) / f32(5u));
    return (abs((mean - s1)) < tol) && (abs((mean - s2)) < tol);
}

// The three-channel form, which the BSI-era signature needs because it must
// hold in every channel rather than in a chosen pair.
fn check_module_size3(r: f32, g: f32, b: f32) -> bool {
    let mean = (((r + g) + b) / f32(3u));
    let tol = ((mean * 2.0) / f32(5u));
    return (abs((mean - r)) < tol) &&
        (abs((mean - g)) < tol) &&
        (abs((mean - b)) < tol);
}

struct CrossV { centery: f32, ms: f32, ok: bool }

// cross_check_pattern_vertical mirrors crossCheckPatternVertical.
fn cross_check_pattern_vertical(
    channel: u32, module_size_max: i32, centerx: f32, centery: f32, slack: i32,
) -> CrossV {
    var sc = array<i32, 5>(0, 0, 0, 0, 0);
    let w = i32(chain_params.width);
    let h = i32(chain_params.height);
    let cx = i32(centerx);
    let cy = i32(centery);

    var i: i32 = 1;
    var state_index: i32 = 0;
    sc[2] = sc[2] + 1;
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
    if state_index < 2 { return CrossV(centery, 0.0, false); }
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
    if state_index < 2 { return CrossV(centery, 0.0, false); }
    let cross = check_pattern_cross(sc);
    if cross.ok && (cross.ms <= f32(module_size_max)) {
        let new_cy = (f32(cy + i - sc[4] - sc[3]) - (f32(sc[2]) * 0.5));
        return CrossV(new_cy, cross.ms, true);
    }
    return CrossV(centery, cross.ms, false);
}

struct CrossH { centerx: f32, ms: f32, ok: bool }

// cross_check_pattern_horizontal mirrors crossCheckPatternHorizontal.
fn cross_check_pattern_horizontal(
    channel: u32, module_size_max: f32, centerx: f32, centery: f32, slack: i32,
) -> CrossH {
    var sc = array<i32, 5>(0, 0, 0, 0, 0);
    let w = i32(chain_params.width);
    let startx = i32(centerx);
    let row_offset = i32(centery) * w;

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
    if state_index < 2 { return CrossH(centerx, 0.0, false); }
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
    if state_index < 2 { return CrossH(centerx, 0.0, false); }
    let cross = check_pattern_cross(sc);
    if cross.ok && (cross.ms <= module_size_max) {
        let new_cx = (f32(startx + i - sc[4] - sc[3]) - (f32(sc[2]) * 0.5));
        return CrossH(new_cx, cross.ms, true);
    }
    return CrossH(centerx, cross.ms, false);
}

struct CrossD { cx: f32, cy: f32, ms: f32, confirmed: i32, dir: i32 }

// cross_check_pattern_diagonal mirrors crossCheckPatternDiagonal, including
// its retry flips and the module-size write of a failed second try.
fn cross_check_pattern_diagonal(
    channel: u32, typ: i32, module_size_max: f32,
    centerx0: f32, centery0: f32, module_size0: f32,
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
    var tmp_module_size = 0.0;
    loop {
        var flag = false;
        try_count = try_count + 1;
        var i: i32 = 0;
        var state_index: i32 = 0;
        var sc = array<i32, 5>(0, 0, 0, 0, 0);
        let startx = i32(centerx);
        let starty = i32(centery);

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
            if cross.ok && (module_size <= module_size_max) {
                if (0.0 < tmp_module_size) {
                    module_size = ((module_size + tmp_module_size) * 0.5);
                } else {
                    tmp_module_size = module_size;
                }
                // Mirrors the walk-direction split in crossCheckPatternDiagonal:
                // y always advances, x retreats wherever offset_x is +1.
                let edge = i - sc[4] - sc[3];
                let half = (f32(sc[2]) * 0.5);
                if offset_x < 0 {
                    centerx = (f32(startx + edge) - half);
                } else {
                    centerx = (f32(startx - edge + 1) + half);
                }
                centery = (f32(starty + edge) - half);
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
        let offset = i32((f32(u32(module_size)) * diag_length_const()));
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

struct CrossCh { ms: f32, cx: f32, cy: f32, dir: i32, dcc: i32, ok: bool }

// cross_check_pattern_ch mirrors crossCheckPatternCh for horizontal
// candidates (hv 0), the only orientation the device chain replays.
fn cross_check_pattern_ch(
    channel: u32, typ: i32, module_size_max: f32, centerx: f32, centery: f32, slack: i32,
) -> CrossCh {
    var cx = centerx;
    var cy = centery;
    var ms_v = 0.0;
    var ms_h = 0.0;
    var ms_d = 0.0;
    var dir: i32 = 0;
    var vcc = false;
    let v = cross_check_pattern_vertical(channel, i32(module_size_max), cx, cy, slack);
    if v.ok {
        vcc = true;
        cy = v.centery;
        ms_v = v.ms;
        let hres = cross_check_pattern_horizontal(channel, module_size_max, cx, cy, slack);
        if !hres.ok { return CrossCh(0.0, cx, cy, dir, 0, false); }
        cx = hres.centerx;
        ms_h = hres.ms;
    }
    let axisx = cx;
    let axisy = cy;
    let d = cross_check_pattern_diagonal(channel, typ, module_size_max, cx, cy, ms_d, dir, !vcc, slack);
    let dcc = d.confirmed;
    cx = d.cx;
    cy = d.cy;
    ms_d = d.ms;
    dir = d.dir;
    if vcc && dcc > 0 {
        // The diagonal is neither a module-scale nor a position measurement;
        // see crossCheckPatternCh for why its centre carries a parity bias the
        // axis walks do not have.
        let ms = ((ms_v + ms_h) / f32(2u));
        return CrossCh(ms, axisx, axisy, dir, dcc, true);
    }
    if dcc == 2 {
        let hres = cross_check_pattern_horizontal(channel, module_size_max, cx, cy, slack);
        if !hres.ok { return CrossCh(0.0, cx, cy, dir, dcc, false); }
        cx = hres.centerx;
        ms_h = hres.ms;
        return CrossCh(ms_h, cx, cy, dir, dcc, true);
    }
    return CrossCh(0.0, cx, cy, dir, dcc, false);
}

// classify_match tests one finder type's palette bits from a 12-bit table.
fn classify_match(table: u32, t: i32, type_r: u32, type_g: u32, type_b: u32) -> bool {
    let bits = table >> (u32(t) * 3u);
    return type_r == (bits & 1u) && type_g == ((bits >> 1u) & 1u) && type_b == ((bits >> 2u) & 1u);
}

struct Outcome { flags: u32, typ: i32, dir: i32, cx: f32, cy: f32, ms: f32 }

fn zero_outcome() -> Outcome {
    return Outcome(0u, 0, 0, 0.0, 0.0, 0.0);
}

// write_outcome stores one hit's outcome in its fixed record slot. Each
// family kernel writes only its own channel's slots, so concurrently
// dispatched family chains never touch the same record.
fn write_outcome(idx: u32, outc: Outcome) {
    let slot = idx * 6u;
    outcomes[slot] = outc.flags;
    outcomes[slot + 1u] = u32(outc.typ);
    outcomes[slot + 2u] = bitcast<u32>(outc.dir);
    outcomes[slot + 3u] = bitcast<u32>(outc.cx);
    outcomes[slot + 4u] = bitcast<u32>(outc.cy);
    outcomes[slot + 5u] = bitcast<u32>(outc.ms);
}
