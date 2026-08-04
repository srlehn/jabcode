// Current-family per-hit chain for arbitrary-direction window records. This
// mirrors processDirectionalFamilyHit. Each lane reads one raw window, walks
// the other mask channels in the scan basis, and writes the same fixed outcome
// shape as the row chain. The source-color signal remains on the host because
// it reads balanced RGB pixels rather than packed masks.

const CHAIN_CONTEXTUAL_SEED: u32 = 32u;
const EVIDENCE_MASK: u32 = 0x3fffffffu;

fn sqrt2_const() -> f32 { return 1.4142135381698608; }

struct AlongResult { cx: f32, cy: f32, ms: f32, ok: bool }

fn direction_pixel(cx: f32, cy: f32, step: i32, d: ChainDirection, channel: u32) -> i32 {
    let i = f32(step);
    let x = i32((cx + (i * d.dx)));
    let y = i32((cy + (i * d.dy)));
    if x < 0 || x >= i32(chain_params.width) || y < 0 || y >= i32(chain_params.height) {
        return -1;
    }
    return i32(mask_bit_at(y * i32(chain_params.width) + x, channel));
}

// cross_check_along mirrors crossCheckAlong, including short-run folding and
// the early inner-width limit. px_per_run is the conversion from counted
// samples to physical pixels for this particular basis direction.
fn cross_check_along(
    channel: u32,
    d: ChainDirection,
    module_size_max: f32,
    px_per_run: f32,
    cx0: f32,
    cy0: f32,
    slack: i32,
) -> AlongResult {
    var sc = array<i32, 5>(0, 0, 0, 0, 0);
    let first = direction_pixel(cx0, cy0, 0, d, channel);
    if first < 0 { return AlongResult(cx0, cy0, 0.0, false); }

    var inside_limit = i32(chain_params.width + chain_params.height + 1u);
    let lim = ((module_size_max * f32(3)) / px_per_run);
    if (lim < f32(inside_limit)) {
        inside_limit = i32(lim) + 2;
    }
    var inside = 1;
    sc[2] = 1;

    var state_index = 0;
    var prev = first;
    for (var i = 1; state_index <= 2; i++) {
        let cur = direction_pixel(cx0, cy0, -i, d, channel);
        if cur < 0 { break; }
        if cur == prev {
            let state = 2 - state_index;
            sc[state] = sc[state] + 1;
            if state > 0 {
                inside = inside + 1;
                if inside >= inside_limit { return AlongResult(cx0, cy0, 0.0, false); }
            }
            continue;
        }
        if state_index > 0 && sc[2 - state_index] < slack {
            sc[2 - (state_index - 1)] = sc[2 - (state_index - 1)] + sc[2 - state_index];
            sc[2 - state_index] = 0;
            state_index = state_index - 1;
            sc[2 - state_index] = sc[2 - state_index] + 1;
            inside = sc[1] + sc[2] + sc[3];
            if inside >= inside_limit { return AlongResult(cx0, cy0, 0.0, false); }
        } else {
            state_index = state_index + 1;
            if state_index > 2 { break; }
            sc[2 - state_index] = sc[2 - state_index] + 1;
            if 2 - state_index > 0 {
                inside = inside + 1;
                if inside >= inside_limit { return AlongResult(cx0, cy0, 0.0, false); }
            }
        }
        prev = cur;
    }
    if state_index < 2 { return AlongResult(cx0, cy0, 0.0, false); }

    state_index = 0;
    prev = first;
    var last = 0;
    for (var i = 1; state_index <= 2; i++) {
        let cur = direction_pixel(cx0, cy0, i, d, channel);
        if cur < 0 { break; }
        last = i;
        if cur == prev {
            let state = 2 + state_index;
            sc[state] = sc[state] + 1;
            if state < 4 {
                inside = inside + 1;
                if inside >= inside_limit { return AlongResult(cx0, cy0, 0.0, false); }
            }
            continue;
        }
        if state_index > 0 && sc[2 + state_index] < slack {
            sc[2 + (state_index - 1)] = sc[2 + (state_index - 1)] + sc[2 + state_index];
            sc[2 + state_index] = 0;
            state_index = state_index - 1;
            sc[2 + state_index] = sc[2 + state_index] + 1;
            inside = sc[1] + sc[2] + sc[3];
            if inside >= inside_limit { return AlongResult(cx0, cy0, 0.0, false); }
        } else {
            state_index = state_index + 1;
            if state_index > 2 { break; }
            sc[2 + state_index] = sc[2 + state_index] + 1;
            if 2 + state_index < 4 {
                inside = inside + 1;
                if inside >= inside_limit { return AlongResult(cx0, cy0, 0.0, false); }
            }
        }
        prev = cur;
    }
    if state_index < 2 { return AlongResult(cx0, cy0, 0.0, false); }

    let cross = check_pattern_cross(sc);
    let ms = (cross.ms * px_per_run);
    if !cross.ok || !(ms <= module_size_max) {
        return AlongResult(cx0, cy0, ms, false);
    }
    let offset = (f32(last - sc[4] - sc[3]) - (f32(sc[2]) * 0.5));
    return AlongResult(
        (cx0 + (offset * d.dx)),
        (cy0 + (offset * d.dy)),
        ms,
        true,
    );
}

struct DirectionalDiagonal { cx: f32, cy: f32, ms: f32, confirmed: i32, dir: i32 }

fn cross_check_directional_diagonal(
    channel: u32,
    typ: i32,
    module_size_max: f32,
    cx0: f32,
    cy0: f32,
    dir0: i32,
    both_dir: bool,
    slack: i32,
) -> DirectionalDiagonal {
    var cx = cx0;
    var cy = cy0;
    var d = chain_params.diagonal_right;
    var dir = 1;
    var fix_dir = false;
    if dir0 != 0 {
        fix_dir = true;
        if dir0 < 0 { d = chain_params.diagonal_left; dir = -1; }
    } else if typ != 0 && typ != 1 {
        d = chain_params.diagonal_left;
        dir = -1;
    }
    var confirmed = 0;
    var tmp_ms = 0.0;
    var module_size = 0.0;
    for (var try_count = 1; try_count <= 2; try_count++) {
        let px_per_run = (d.px_per_sample / sqrt2_const());
        let result = cross_check_along(channel, d, module_size_max, px_per_run, cx, cy, slack);
        if result.ok {
            module_size = result.ms;
            if (0.0 < tmp_ms) {
                module_size = ((module_size + tmp_ms) * 0.5);
            } else {
                tmp_ms = module_size;
            }
            cx = result.cx;
            cy = result.cy;
            confirmed = confirmed + 1;
            if !both_dir || try_count == 2 || fix_dir {
                if confirmed == 2 { dir = 2; }
                return DirectionalDiagonal(cx, cy, module_size, confirmed, dir);
            }
            continue;
        }
        if try_count == 2 || fix_dir {
            return DirectionalDiagonal(cx, cy, module_size, confirmed, dir);
        }
        if dir > 0 {
            d = chain_params.diagonal_left;
            dir = -1;
        } else {
            d = chain_params.diagonal_right;
            dir = 1;
        }
    }
    return DirectionalDiagonal(cx, cy, module_size, confirmed, dir);
}

fn cross_check_color_along(
    channel: u32,
    color_bit: u32,
    module_size: f32,
    cx: f32,
    cy: f32,
    d: ChainDirection,
    tol: i32,
) -> bool {
    let n = i32(((module_size * 4.0) / d.px_per_sample));
    if n <= 0 { return false; }
    let half = (f32(n) * 0.5);
    let x0 = (cx - (half * d.dx));
    let y0 = (cy - (half * d.dy));
    var unmatch = 0;
    for (var i = 0; i < n; i++) {
        let step = f32(i);
        let x = i32((x0 + (step * d.dx)));
        let y = i32((y0 + (step * d.dy)));
        if x < 0 || x >= i32(chain_params.width) || y < 0 || y >= i32(chain_params.height) { break; }
        if mask_bit_at(y * i32(chain_params.width) + x, channel) != color_bit {
            unmatch = unmatch + 1;
        } else if unmatch <= tol {
            unmatch = 0;
        }
        if unmatch > tol { return false; }
    }
    return true;
}

fn cross_check_color_basis(
    channel: u32, color_bit: u32, module_size: f32, cx: f32, cy: f32, slack: i32,
) -> bool {
    if !cross_check_color_along(channel, color_bit, module_size, cx, cy, chain_params.base, slack) ||
        !cross_check_color_along(channel, color_bit, module_size, cx, cy, chain_params.perpendicular, slack) {
        return false;
    }
    return cross_check_color_along(channel, color_bit, module_size, cx, cy, chain_params.diagonal_right, slack) ||
        cross_check_color_along(channel, color_bit, module_size, cx, cy, chain_params.diagonal_left, slack);
}

struct DirectionalCrossCh { ms: f32, cx: f32, cy: f32, dir: i32, dcc: i32, ok: bool }

fn cross_check_directional_ch(
    channel: u32, typ: i32, module_size_max: f32, cx0: f32, cy0: f32, slack: i32,
) -> DirectionalCrossCh {
    var cx = cx0;
    var cy = cy0;
    var ms_p = 0.0;
    var ms_a = 0.0;
    var pcc = false;
    let p = cross_check_along(
        channel, chain_params.perpendicular, module_size_max,
        chain_params.perpendicular.px_per_sample, cx, cy, slack,
    );
    if p.ok {
        pcc = true;
        cx = p.cx;
        cy = p.cy;
        ms_p = p.ms;
        let a = cross_check_along(
            channel, chain_params.base, module_size_max,
            chain_params.base.px_per_sample, cx, cy, slack,
        );
        if !a.ok { return DirectionalCrossCh(0.0, cx, cy, 0, 0, false); }
        cx = a.cx;
        cy = a.cy;
        ms_a = a.ms;
    }
    let diagonal = cross_check_directional_diagonal(
        channel, typ, module_size_max, cx, cy, 0, !pcc, slack,
    );
    cx = diagonal.cx;
    cy = diagonal.cy;
    if pcc && diagonal.confirmed > 0 {
        return DirectionalCrossCh(((ms_p + ms_a) * 0.5), cx, cy, diagonal.dir, diagonal.confirmed, true);
    }
    if diagonal.confirmed == 2 {
        let a = cross_check_along(
            channel, chain_params.base, module_size_max,
            chain_params.base.px_per_sample, cx, cy, slack,
        );
        if !a.ok { return DirectionalCrossCh(0.0, cx, cy, diagonal.dir, diagonal.confirmed, false); }
        return DirectionalCrossCh(a.ms, a.cx, a.cy, diagonal.dir, diagonal.confirmed, true);
    }
    return DirectionalCrossCh(0.0, cx, cy, diagonal.dir, diagonal.confirmed, false);
}

struct DirectionalPattern { cx: f32, cy: f32, ms: f32, dir: i32, ok: bool }

fn cross_check_directional_pattern(
    typ: i32, cx0: f32, cy0: f32, module_size0: f32, slack: i32,
) -> DirectionalPattern {
    let module_size_max = (module_size0 * 2.0);
    let green = cross_check_directional_ch(1u, typ, module_size_max, cx0, cy0, slack);
    if !green.ok { return DirectionalPattern(cx0, cy0, module_size0, 0, false); }

    var other_channel = 2u;
    var color_channel = 0u;
    var color_bit = chain_params.cross_color_bits & 1u;
    if typ == 1 || typ == 2 {
        other_channel = 0u;
        color_channel = 2u;
        color_bit = (chain_params.cross_color_bits >> 1u) & 1u;
    }
    let other = cross_check_directional_ch(other_channel, typ, module_size_max, cx0, cy0, slack);
    if !other.ok || !check_module_size2(green.ms, other.ms) {
        return DirectionalPattern(cx0, cy0, module_size0, 0, false);
    }
    let ms = ((green.ms + other.ms) * 0.5);
    let cx = ((green.cx + other.cx) * 0.5);
    let cy = ((green.cy + other.cy) * 0.5);
    if !cross_check_color_basis(color_channel, color_bit, ms, cx, cy, slack) {
        return DirectionalPattern(cx0, cy0, module_size0, 0, false);
    }
    var direction = -1;
    if green.dcc == 2 || other.dcc == 2 {
        direction = 2;
    } else if green.dir + other.dir > 0 {
        direction = 1;
    }
    return DirectionalPattern(cx, cy, ms, direction, true);
}

fn process_directional_hit(idx: u32) -> Outcome {
    var outc = zero_outcome();
    let at = idx * 8u;
    let key = records.data[at] & EVIDENCE_MASK;
    let line = key / 3u;
    let q = (chain_params.geom_q_lo + (f32(line) * chain_params.geom_q_step));
    let along = (f32(records.data[at + 3u] + records.data[at + 4u]) * 0.5);
    let center_g_x = ((q * chain_params.geom_nx) + (along * chain_params.geom_dx));
    let center_g_y = ((q * chain_params.geom_ny) + (along * chain_params.geom_dy));
    let inner = records.data[at + 5u] - records.data[at + 2u];
    let module_g = ((f32(inner) / f32(3u)) * chain_params.base.px_per_sample);

    let gx = i32(center_g_x);
    let gy = i32(center_g_y);
    if gx < 0 || gx >= i32(chain_params.width) || gy < 0 || gy >= i32(chain_params.height) { return outc; }
    let type_g = mask_bit_at(gy * i32(chain_params.width) + gx, 1u);
    let slack = chain_slack(module_g);
    let module_max = (module_g * 2.0);

    var branch = -1;
    var probe = AlongResult(center_g_x, center_g_y, 0.0, false);
    for (var b = 0; b < 2; b++) {
        var channel = 2u;
        if b == 1 { channel = 0u; }
        let current = cross_check_along(
            channel, chain_params.base, module_max, chain_params.base.px_per_sample,
            center_g_x, center_g_y, slack,
        );
        if current.ok {
            branch = b;
            probe = current;
            break;
        }
    }
    if branch < 0 { return outc; }

    var type_r = 0u;
    var type_b = 0u;
    var t0 = 0;
    var t1 = 3;
    if branch == 0 {
        outc.flags = outc.flags | 1u;
        let bx = i32(probe.cx);
        let by = i32(probe.cy);
        if bx < 0 || bx >= i32(chain_params.width) || by < 0 || by >= i32(chain_params.height) { return outc; }
        type_b = mask_bit_at(by * i32(chain_params.width) + bx, 2u);
        if !cross_check_color_along(
            0u, chain_params.cross_color_bits & 1u, module_g,
            center_g_x, center_g_y, chain_params.base, slack,
        ) { return outc; }
    } else {
        outc.flags = outc.flags | 2u;
        let rx = i32(probe.cx);
        let ry = i32(probe.cy);
        if rx < 0 || rx >= i32(chain_params.width) || ry < 0 || ry >= i32(chain_params.height) { return outc; }
        type_r = mask_bit_at(ry * i32(chain_params.width) + rx, 0u);
        if !cross_check_color_along(
            2u, (chain_params.cross_color_bits >> 1u) & 1u, module_g,
            center_g_x, center_g_y, chain_params.base, slack,
        ) { return outc; }
        outc.flags = outc.flags | 4u;
        t0 = 1;
        t1 = 2;
    }
    if !check_module_size2(module_g, probe.ms) { return outc; }
    let cx = ((center_g_x + probe.cx) * 0.5);
    let cy = ((center_g_y + probe.cy) * 0.5);
    let ms = ((module_g + probe.ms) * 0.5);

    var typ = -1;
    if classify_match(chain_params.classify_current, t0, type_r, type_g, type_b) {
        typ = t0;
    } else if classify_match(chain_params.classify_current, t1, type_r, type_g, type_b) {
        typ = t1;
    } else {
        return outc;
    }
    if branch == 1 { outc.flags = outc.flags | 8u; }

    let pat = cross_check_directional_pattern(typ, cx, cy, ms, chain_slack(ms));
    outc.typ = typ;
    if !pat.ok {
        outc.flags = outc.flags | CHAIN_CONTEXTUAL_SEED;
        outc.cx = cx;
        outc.cy = cy;
        outc.ms = ms;
        return outc;
    }
    outc.flags = outc.flags | 16u;
    outc.dir = pat.dir;
    outc.cx = pat.cx;
    outc.cy = pat.cy;
    outc.ms = pat.ms;
    return outc;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= chain_params.capacity { return; }
    write_outcome(id.x, process_directional_hit(id.x));
}
