// BSI-era fragment of the arbitrary-direction per-hit chain, appended to the
// shared directional machinery to form one kernel module. Present only in
// builds whose decoder compiles the BSI family in.
//
// This mirrors processDirectionalBSIFamilyHit, and it is shorter than the
// current-family fragment for a reason that is a property of the signature
// rather than of this port: the BSI pattern must hold in all three channels,
// so there is no branch to choose between, no core-colour probe on a third
// channel, and no source-level colour signal. Every channel takes the same
// walks, which is also why every stage here is a loop over three channels
// rather than a pair of hand-written cases.

struct BSIDirectionalPattern { cx: f32, cy: f32, ms: f32, dir: i32, ok: bool }

// cross_check_pattern_bsi_directional is crossCheckPatternBSIFamilyAlong: the
// full cross-check in every channel, each starting from the same candidate
// centre, averaged where all three agree.
fn cross_check_pattern_bsi_directional(
    typ: i32, cx0: f32, cy0: f32, module_size0: f32, slack: i32,
) -> BSIDirectionalPattern {
    let module_size_max = (module_size0 * 2.0);
    var ms = array<f32, 3>(0.0, 0.0, 0.0);
    var cx = array<f32, 3>(0.0, 0.0, 0.0);
    var cy = array<f32, 3>(0.0, 0.0, 0.0);
    var dir = array<i32, 3>(0, 0, 0);
    var dcc = array<i32, 3>(0, 0, 0);
    for (var c = 0; c < 3; c++) {
        let res = cross_check_directional_ch(u32(c), typ, module_size_max, cx0, cy0, slack);
        if !res.ok { return BSIDirectionalPattern(cx0, cy0, module_size0, 0, false); }
        ms[c] = res.ms;
        cx[c] = res.cx;
        cy[c] = res.cy;
        dir[c] = res.dir;
        dcc[c] = res.dcc;
    }
    if !check_module_size3(ms[0], ms[1], ms[2]) {
        return BSIDirectionalPattern(cx0, cy0, module_size0, 0, false);
    }
    var direction = -1;
    if dcc[0] == 2 || dcc[1] == 2 || dcc[2] == 2 {
        direction = 2;
    } else if ((dir[0] + dir[1]) + dir[2]) > 0 {
        direction = 1;
    }
    return BSIDirectionalPattern(
        (((cx[0] + cx[1]) + cx[2]) / f32(3u)),
        (((cy[0] + cy[1]) + cy[2]) / f32(3u)),
        (((ms[0] + ms[1]) + ms[2]) / f32(3u)),
        direction,
        true,
    );
}

fn process_bsi_directional_hit(idx: u32) -> Outcome {
    var outc = zero_outcome();
    let seed = directional_seed(idx);
    let slack = chain_slack(seed.ms);
    let module_max = (seed.ms * 2.0);

    // The record was seeded on red; green and blue have to confirm the same
    // signature along the same line before anything else is asked.
    var ms = array<f32, 3>(seed.ms, 0.0, 0.0);
    var cx = array<f32, 3>(seed.cx, 0.0, 0.0);
    var cy = array<f32, 3>(seed.cy, 0.0, 0.0);
    for (var c = 1; c < 3; c++) {
        let res = cross_check_along(
            u32(c), chain_params.base, module_max,
            chain_params.base.px_per_sample, seed.cx, seed.cy, slack,
        );
        if !res.ok { return outc; }
        ms[c] = res.ms;
        cx[c] = res.cx;
        cy[c] = res.cy;
    }
    if !check_module_size3(ms[0], ms[1], ms[2]) { return outc; }

    var colour = array<u32, 3>(0u, 0u, 0u);
    for (var c = 0; c < 3; c++) {
        let x = i32(cx[c]);
        let y = i32(cy[c]);
        if x < 0 || x >= i32(chain_params.width) || y < 0 || y >= i32(chain_params.height) {
            return outc;
        }
        colour[c] = mask_bit_at(y * i32(chain_params.width) + x, u32(c));
    }

    var typ = -1;
    for (var t = 0; t < 4; t++) {
        if classify_match(chain_params.classify_bsi, t, colour[0], colour[1], colour[2]) {
            typ = t;
            break;
        }
    }
    if typ < 0 { return outc; }

    let centre_x = (((cx[0] + cx[1]) + cx[2]) / f32(3u));
    let centre_y = (((cy[0] + cy[1]) + cy[2]) / f32(3u));
    let centre_ms = (((ms[0] + ms[1]) + ms[2]) / f32(3u));
    let pat = cross_check_pattern_bsi_directional(
        typ, centre_x, centre_y, centre_ms, chain_slack(centre_ms),
    );
    if !pat.ok { return outc; }
    outc.flags = outc.flags | 16u;
    outc.typ = typ;
    outc.dir = pat.dir;
    outc.cx = pat.cx;
    outc.cy = pat.cy;
    outc.ms = pat.ms;
    return outc;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= dispatch_args[3] { return; }
    summarize(process_bsi_directional_hit(id.x), directional_seed(id.x).ms);
}
