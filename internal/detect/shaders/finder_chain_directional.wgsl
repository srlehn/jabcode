// Shared machinery for the arbitrary-direction per-hit chains: the walks, the
// summary reduction, and the seed each family's fragment starts from. A family
// fragment is appended to this to form one kernel module, exactly as the row
// chains append theirs to finder_chain_prelude.wgsl.
//
// Each lane owns one raw window record. Nothing here reads source intensities;
// a family that needs them binds the balanced image itself.

const EVIDENCE_MASK: u32 = 0x3fffffffu;

// The contextual-seed flag lives here rather than in a family fragment because
// summarize decides compaction on it. Only the current family raises it.
const CHAIN_CONTEXTUAL_SEED: u32 = 32u;

fn sqrt2_const() -> f32 { return 1.4142135381698608; }

struct DirectionalSeed { cx: f32, cy: f32, ms: f32 }

// directional_seed resolves one raw record to the point and module size the
// per-hit chain starts from: the midpoint of the middle run projected back
// through the sweep basis, and the mean inner run in pixels. Both families
// seek the same five-run signature, so they share this and differ only in
// which channel produced it and what has to confirm it.
fn directional_seed(idx: u32) -> DirectionalSeed {
    let at = idx * 8u;
    let line = (records.data[at] & EVIDENCE_MASK) / 3u;
    let q = (chain_params.geom_q_lo + (f32(line) * chain_params.geom_q_step));
    let along = (f32(records.data[at + 3u] + records.data[at + 4u]) * 0.5);
    let inner = records.data[at + 5u] - records.data[at + 2u];
    return DirectionalSeed(
        ((q * chain_params.geom_nx) + (along * chain_params.geom_dx)),
        ((q * chain_params.geom_ny) + (along * chain_params.geom_dy)),
        ((f32(inner) / f32(3u)) * chain_params.base.px_per_sample),
    );
}

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

// Summary word offsets, shared with the host parser.
const SUMMARY_COMPACTED: u32 = 0u;
const SUMMARY_RAW_HITS: u32 = 1u;
const SUMMARY_BRANCH_BLUE: u32 = 2u;
const SUMMARY_BRANCH_RED: u32 = 3u;
const SUMMARY_RED_COLOR: u32 = 4u;
const SUMMARY_RED_CLASSIFIED: u32 = 5u;
// Word six is the raw hit count finder_dispatch_args.wgsl publishes for the
// host's overflow check; no atomic here touches it.
const SUMMARY_HISTOGRAM: u32 = 7u;
const SUMMARY_HISTOGRAM_BUCKETS: u32 = 1024u;
// Quarter-pixel buckets, matching the host accumulator this histogram merges
// into.
const SUMMARY_MODULE_SCALE: f32 = 4.0;

// summarize folds one hit into the shared counters and, when the hit survived,
// appends it to the compacted candidate list. Hundreds of thousands of hits
// reduce to a few hundred records this way, which is the difference between
// the host reading a summary and the host reading the whole sweep.
fn summarize(outc: Outcome, module: f32) {
    atomicAdd(&summary[SUMMARY_RAW_HITS], 1u);
    if module > 0.0 {
        var bucket = u32(module * SUMMARY_MODULE_SCALE);
        if bucket >= SUMMARY_HISTOGRAM_BUCKETS { bucket = SUMMARY_HISTOGRAM_BUCKETS - 1u; }
        atomicAdd(&summary[SUMMARY_HISTOGRAM + bucket], 1u);
    }
    if (outc.flags & 1u) != 0u { atomicAdd(&summary[SUMMARY_BRANCH_BLUE], 1u); }
    if (outc.flags & 2u) != 0u { atomicAdd(&summary[SUMMARY_BRANCH_RED], 1u); }
    if (outc.flags & 4u) != 0u { atomicAdd(&summary[SUMMARY_RED_COLOR], 1u); }
    if (outc.flags & 8u) != 0u { atomicAdd(&summary[SUMMARY_RED_CLASSIFIED], 1u); }
    if (outc.flags & (16u | CHAIN_CONTEXTUAL_SEED)) == 0u { return; }
    let slot = atomicAdd(&summary[SUMMARY_COMPACTED], 1u);
    // The count keeps rising past the buffer so the host can see the overflow
    // and walk that direction itself rather than act on a truncated list.
    if slot >= chain_params.compact_capacity { return; }
    write_outcome(slot, outc);
}
