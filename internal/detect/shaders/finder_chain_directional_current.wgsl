// Current-family fragment of the arbitrary-direction per-hit chain, appended to
// the shared directional machinery to form one kernel module. This mirrors
// processDirectionalFamilyHit: one lane reads one raw window, picks the branch
// its seek channel confirms on, and decides classification, the full pattern
// cross-check and the source-colour signal for that candidate.

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
    let seed = directional_seed(idx);
    let center_g_x = seed.cx;
    let center_g_y = seed.cy;
    let module_g = seed.ms;

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
        return record_color_signal(outc, typ, cx, cy, ms,
        chain_params.base.dx, chain_params.base.dy, chain_params.base.px_per_sample);
    }
    outc.flags = outc.flags | 16u;
    outc.dir = pat.dir;
    outc.cx = pat.cx;
    outc.cy = pat.cy;
    outc.ms = pat.ms;
    return record_color_signal(outc, typ, pat.cx, pat.cy, pat.ms,
        chain_params.base.dx, chain_params.base.dy, chain_params.base.px_per_sample);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= dispatch_args[3] { return; }
    summarize(process_directional_hit(id.x), directional_seed(id.x).ms);
}
