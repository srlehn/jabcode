// Current-family fragment of the arbitrary-direction per-hit chain, appended to
// the shared directional machinery to form one kernel module. This mirrors
// processDirectionalFamilyHit: one lane reads one raw window, picks the branch
// its seek channel confirms on, and decides classification, the full pattern
// cross-check and the source-colour signal for that candidate.
//
// The colour signal is the one stage that reads balanced source intensities
// rather than mask bits, which is why this fragment binds the balanced image
// and the BSI-era one does not.

const CHAIN_COLOR_EVALUATED: u32 = 64u;
const CHAIN_COLOR_OK: u32 = 128u;

// Bit 1 of the parameter flags says the balanced image is bound, which is what
// makes the colour test answerable here at all.
const CHAIN_FLAG_COLOR_SOURCE: u32 = 2u;

// finder_min_channel_contrast is the signed Michelson contrast an FP1 or FP2
// candidate must show between its yellow and black bands.
const FINDER_MIN_CHANNEL_CONTRAST: f32 = 0.1;

// color_signal_ok verifies that an FP1/FP2 mask signature is a source-level
// yellow-to-black transition in both colour-bearing channels. The palette
// classifier gives yellow and black identical red and green masks, so the mask
// walks alone decide this once; sampling the balanced image across the expected
// five-module band restores two independent source observations.
//
// The band is walked as one strided sample set rather than as the host's two
// nested loops: every invocation already owns its candidate, so the work is
// one linear pass with two running sums.
fn color_signal_ok(typ: i32, cx: f32, cy: f32, ms: f32) -> bool {
    if typ != 1 && typ != 2 { return true; }
    if ms <= 0.0 { return false; }
    let d = chain_params.base;
    let sample_count = max(5, i32(ceil(5.0 * ms / d.px_per_sample)));
    var core_bit = 0;
    if typ == 2 { core_bit = 1; }
    var sums = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    var counts = array<i32, 2>(0, 0);
    for (var i = 0; i < sample_count; i++) {
        let offset = (f32(i) + 0.5) / f32(sample_count) * 5.0 - 2.5;
        var bit = core_bit;
        let distance = abs(offset);
        if distance >= 0.5 && distance < 1.5 { bit = 1 - bit; }
        let x = i32(cx + offset * ms * d.dx / d.px_per_sample);
        let y = i32(cy + offset * ms * d.dy / d.px_per_sample);
        if x < 0 || x >= i32(chain_params.width) || y < 0 || y >= i32(chain_params.height) {
            continue;
        }
        let pixel = balanced_pixels[u32(y) * chain_params.width + u32(x)];
        sums[bit] = sums[bit] + f32(pixel & 0xffu);
        sums[2 + bit] = sums[2 + bit] + f32((pixel >> 8u) & 0xffu);
        counts[bit] = counts[bit] + 1;
    }
    if counts[0] == 0 || counts[1] == 0 { return false; }
    for (var channel = 0; channel < 2; channel++) {
        let black = sums[channel * 2] / f32(counts[0]);
        let yellow = sums[channel * 2 + 1] / f32(counts[1]);
        if (yellow - black) / max(yellow + black, 1.0) < FINDER_MIN_CHANNEL_CONTRAST {
            return false;
        }
    }
    return true;
}

// record_color_signal stamps the colour verdict for one candidate. A kernel
// dispatched without the balanced image stamps nothing, and the host runs the
// test itself for those hits.
fn record_color_signal(outc: Outcome, typ: i32, cx: f32, cy: f32, ms: f32) -> Outcome {
    var result = outc;
    if (chain_params.flags & CHAIN_FLAG_COLOR_SOURCE) == 0u {
        return result;
    }
    result.flags = result.flags | CHAIN_COLOR_EVALUATED;
    if color_signal_ok(typ, cx, cy, ms) {
        result.flags = result.flags | CHAIN_COLOR_OK;
    }
    return result;
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
        return record_color_signal(outc, typ, cx, cy, ms);
    }
    outc.flags = outc.flags | 16u;
    outc.dir = pat.dir;
    outc.cx = pat.cx;
    outc.cy = pat.cy;
    outc.ms = pat.ms;
    return record_color_signal(outc, typ, pat.cx, pat.cy, pat.ms);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= dispatch_args[3] { return; }
    summarize(process_directional_hit(id.x), directional_seed(id.x).ms);
}
