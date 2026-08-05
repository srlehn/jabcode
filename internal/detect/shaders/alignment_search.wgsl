// Locates one anti-diagonal of the alignment-pattern grid against the resident
// packed masks. The host search and this kernel answer the same question and
// accept on the same evidence, but they look for it in opposite shapes.
//
// The host walks outward from the prediction: a radius that doubles until
// something is found, two cursors taking turns along the module axis so
// candidates arrive ordered by distance, and the line abandoned at its first
// acceptance. Every one of those is a way to avoid work on a processor that
// evaluates one candidate at a time, and none of them survives here. A lane
// owns one candidate offset, the whole window is evaluated whether or not an
// early candidate would have satisfied the host, and "nearest accepted
// candidate to the prediction" falls out of a workgroup reduction over the
// distances rather than out of the order the candidates were visited in. The
// doubling radius disappears with it: the widest radius the host could reach is
// simply the window, because covering it costs occupancy rather than time.
//
// The grid is walked one anti-diagonal per dispatch because a cell's prediction
// is extrapolated from the neighbours above and to its left. Cells on a common
// anti-diagonal have no such dependency on each other, so the chain resolves in
// as many dispatches as there are diagonals, all recorded into one submission.
// The host never sees the intermediate cells.

const PARAM_WIDTH: u32 = 0u;
const PARAM_HEIGHT: u32 = 1u;
const PARAM_NAPX: u32 = 2u;
const PARAM_NAPY: u32 = 3u;
const PARAM_DIAGONAL: u32 = 4u;
const PARAM_CORE_R: u32 = 5u;
const PARAM_CORE_G: u32 = 6u;
const PARAM_CORE_B: u32 = 7u;
const PARAM_SIDE_X: u32 = 8u;
const PARAM_SIDE_Y: u32 = 9u;
const PARAM_MODULE_MAX: u32 = 10u;
const PARAM_QUAD: u32 = 11u;
const PARAM_AP_POS_X: u32 = 19u;
const PARAM_AP_POS_Y: u32 = 28u;

// One cell's result, and the state the next diagonal extrapolates from.
const CELL_WORDS: u32 = 6u;
const CELL_FOUND: u32 = 0u;
const CELL_CX: u32 = 1u;
const CELL_CY: u32 = 2u;
const CELL_MODULE: u32 = 3u;
const CELL_DIR: u32 = 4u;

const WORKGROUP: u32 = 256u;

// A run shorter than this is noise rather than a transition, and folds back
// into the run before it. The host row walks use the same bound.
const MIN_RUN: i32 = 3;

@group(0) @binding(0) var<storage, read> masks: array<u32>;
@group(0) @binding(1) var<storage, read_write> cells: array<u32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

fn param_f32(index: u32) -> f32 {
    return bitcast<f32>(params[index]);
}

fn cell_f32(cell: u32, field: u32) -> f32 {
    return bitcast<f32>(cells[cell * CELL_WORDS + field]);
}

fn quad(corner: u32) -> vec2<f32> {
    return vec2<f32>(
        param_f32(PARAM_QUAD + corner * 2u),
        param_f32(PARAM_QUAD + corner * 2u + 1u),
    );
}

fn in_frame(p: vec2<i32>) -> bool {
    return p.x >= 0 && p.x < i32(params[PARAM_WIDTH]) &&
        p.y >= 0 && p.y < i32(params[PARAM_HEIGHT]);
}

fn mask_at(p: vec2<i32>, channel: u32) -> u32 {
    let pixel = u32(p.y) * params[PARAM_WIDTH] + u32(p.x);
    return (masks[pixel / 8u] >> ((pixel % 8u) * 3u + channel)) & 1u;
}

// step_vector normalizes on the major axis, so one step advances a full pixel
// along the dominant direction and the run lengths a walk reports are directly
// comparable with a module size measured in pixels.
fn step_vector(v: vec2<f32>) -> vec2<f32> {
    let major = max(abs(v.x), abs(v.y));
    if major == 0.0 {
        return vec2<f32>(0.0, 0.0);
    }
    return v / major;
}

// Run lengths of the core colour and its two flanks along one axis, measured
// outward from a candidate. Returns the triple as (left, core, right) with the
// refined centre offset, or a zero core when the candidate is not on core
// colour at all.
struct Runs {
    left: i32,
    core: i32,
    right: i32,
    centre: f32,
}

fn axis_runs(origin: vec2<f32>, step: vec2<f32>, channel: u32, core: u32, limit: i32) -> Runs {
    var runs = Runs(0, 0, 0, 0.0);
    let seed = vec2<i32>(i32(floor(origin.x)), i32(floor(origin.y)));
    if !in_frame(seed) || mask_at(seed, channel) != core {
        return runs;
    }
    runs.core = 1;
    // Outward in both directions: the first run of core colour is the pattern's
    // centre run, and the run beyond each end is its flank. A flank shorter
    // than MIN_RUN is folded back, which is how the host tolerates a single
    // mis-binarized sample inside a run.
    for (var side = 0; side < 2; side++) {
        let sign = select(1.0, -1.0, side == 0);
        var flank = 0;
        var in_core = true;
        for (var i = 1; i <= limit; i++) {
            let at = origin + step * (sign * f32(i));
            let p = vec2<i32>(i32(floor(at.x)), i32(floor(at.y)));
            if !in_frame(p) {
                break;
            }
            let bit = mask_at(p, channel);
            if in_core {
                if bit == core {
                    runs.core++;
                    continue;
                }
                in_core = false;
                flank = 1;
                continue;
            }
            if bit != core {
                flank++;
                continue;
            }
            // Core colour again: a short excursion folds back into the flank,
            // anything longer ends it.
            if flank < MIN_RUN {
                runs.core += flank + 1;
                flank = 0;
                in_core = true;
                continue;
            }
            break;
        }
        if side == 0 {
            runs.right = flank;
        } else {
            runs.left = flank;
        }
    }
    return runs;
}

// accept_runs is the host's ratio test: a core run under the module ceiling
// with both flanks at least half its width.
fn accept_runs(runs: Runs, module_max: f32) -> bool {
    if runs.core <= 0 {
        return false;
    }
    let core = f32(runs.core);
    return core < module_max &&
        f32(runs.left) > 0.5 * core &&
        f32(runs.right) > 0.5 * core;
}

// colour_holds walks the green channel across the candidate the way the host's
// colour cross-check does, requiring the core's own green bit over the span the
// pattern core occupies.
fn colour_holds(centre: vec2<f32>, step: vec2<f32>, module: f32, core: u32) -> bool {
    let reach = i32(max(module, 1.0));
    for (var i = -reach; i <= reach; i++) {
        let at = centre + step * f32(i);
        let p = vec2<i32>(i32(floor(at.x)), i32(floor(at.y)));
        if !in_frame(p) {
            return false;
        }
        if mask_at(p, 1u) != core {
            return false;
        }
    }
    return true;
}

// One candidate's verdict and how far it sits from the prediction.
struct Candidate {
    ok: bool,
    module: f32,
    dir: i32,
    distance: f32,
    centre: vec2<f32>,
}

fn evaluate(
    at: vec2<f32>,
    prediction: vec2<f32>,
    u: vec2<f32>,
    v: vec2<f32>,
    module_max: f32,
    limit: i32,
) -> Candidate {
    var out = Candidate(false, 0.0, 0, 0.0, at);
    let core_r = params[PARAM_CORE_R];
    let core_g = params[PARAM_CORE_G];
    let core_b = params[PARAM_CORE_B];

    // Red and blue must both carry the signature along both module axes. The
    // host interleaves the channels so each walk refines the centre the next
    // one measures from; a lane has the whole neighbourhood in reach at once,
    // so it measures all four from the same candidate and averages instead.
    let ur = axis_runs(at, u, 0u, core_r, limit);
    if !accept_runs(ur, module_max) {
        return out;
    }
    let ub = axis_runs(at, u, 2u, core_b, limit);
    if !accept_runs(ub, module_max) {
        return out;
    }
    let vr = axis_runs(at, v, 0u, core_r, limit);
    if !accept_runs(vr, module_max) {
        return out;
    }
    let vb = axis_runs(at, v, 2u, core_b, limit);
    if !accept_runs(vb, module_max) {
        return out;
    }
    let module = f32(ur.core + ub.core + vr.core + vb.core) / 4.0;
    if !colour_holds(at, u, module, core_g) || !colour_holds(at, v, module, core_g) {
        return out;
    }

    // The diagonals decide the pattern's class: one runs along the join between
    // the two square references and the other crosses the data quadrants, so
    // which of them carries the longer core run is the direction.
    let d_plus = step_vector(u + v);
    let d_minus = step_vector(u - v);
    let dp = axis_runs(at, d_plus, 0u, core_r, limit);
    let dm = axis_runs(at, d_minus, 0u, core_r, limit);
    if dp.core <= 0 && dm.core <= 0 {
        return out;
    }

    out.ok = true;
    out.module = module;
    out.dir = select(-1, 1, dp.core >= dm.core);
    out.distance = distance(at, prediction);
    out.centre = at;
    return out;
}

var<workgroup> best_distance: array<f32, WORKGROUP>;
var<workgroup> best_slot: array<u32, WORKGROUP>;
var<workgroup> found_cx: array<f32, WORKGROUP>;
var<workgroup> found_cy: array<f32, WORKGROUP>;
var<workgroup> found_module: array<f32, WORKGROUP>;
var<workgroup> found_dir: array<i32, WORKGROUP>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let n_ap_x = params[PARAM_NAPX];
    let n_ap_y = params[PARAM_NAPY];
    let diagonal = params[PARAM_DIAGONAL];
    let first_row = select(0u, diagonal - (n_ap_x - 1u), diagonal + 1u > n_ap_x);
    let i = first_row + group.x;
    if i >= n_ap_y || i > diagonal {
        return;
    }
    let j = diagonal - i;
    if j >= n_ap_x {
        return;
    }
    let index = i * n_ap_x + j;
    // The four quad corners are seeded by the host and are not searched.
    let corner = (i == 0u && j == 0u) || (i == 0u && j == n_ap_x - 1u) ||
        (i == n_ap_y - 1u && j == n_ap_x - 1u) || (i == n_ap_y - 1u && j == 0u);
    if corner {
        return;
    }

    // Extrapolate this cell's prediction from the neighbours already placed.
    var prediction = vec2<f32>(0.0, 0.0);
    var seed_module = 0.0;
    if i == 0u {
        let left = index - 1u;
        let towards = quad(1u);
        let from = vec2<f32>(cell_f32(left, CELL_CX), cell_f32(left, CELL_CY));
        seed_module = cell_f32(left, CELL_MODULE);
        let span = f32(params[PARAM_AP_POS_X + j] - params[PARAM_AP_POS_X + j - 1u]);
        let heading = normalize(towards - from);
        prediction = from + heading * (seed_module * span);
    } else if j == 0u {
        let above = (i - 1u) * n_ap_x;
        let towards = quad(3u);
        let from = vec2<f32>(cell_f32(above, CELL_CX), cell_f32(above, CELL_CY));
        seed_module = cell_f32(above, CELL_MODULE);
        let span = f32(params[PARAM_AP_POS_Y + i] - params[PARAM_AP_POS_Y + i - 1u]);
        let heading = normalize(towards - from);
        prediction = from + heading * (seed_module * span);
    } else {
        let a0 = (i - 1u) * n_ap_x + (j - 1u);
        let a1 = (i - 1u) * n_ap_x + j;
        let a3 = i * n_ap_x + (j - 1u);
        let avg01 = (cell_f32(a0, CELL_MODULE) + cell_f32(a1, CELL_MODULE)) * 0.5;
        let avg13 = (cell_f32(a1, CELL_MODULE) + cell_f32(a3, CELL_MODULE)) * 0.5;
        let p0 = vec2<f32>(cell_f32(a0, CELL_CX), cell_f32(a0, CELL_CY));
        let p1 = vec2<f32>(cell_f32(a1, CELL_CX), cell_f32(a1, CELL_CY));
        let p3 = vec2<f32>(cell_f32(a3, CELL_CX), cell_f32(a3, CELL_CY));
        prediction = (p1 - p0) / avg01 * avg13 + p3;
        seed_module = avg13;
    }

    // The module axes at this cell, each interpolated between the quad's two
    // opposite edges at the cell's own position. Long-baseline and local at
    // once: short-baseline axes taken from the nearest neighbours would turn a
    // fraction of a pixel of centre error into a visible tilt.
    let tx = f32(params[PARAM_AP_POS_X + j]) / f32(params[PARAM_SIDE_X]);
    let ty = f32(params[PARAM_AP_POS_Y + i]) / f32(params[PARAM_SIDE_Y]);
    let q0 = quad(0u);
    let q1 = quad(1u);
    let q2 = quad(2u);
    let q3 = quad(3u);
    let u_edge = (q1 - q0) + ((q2 - q3) - (q1 - q0)) * ty;
    let v_edge = (q3 - q0) + ((q2 - q1) - (q3 - q0)) * tx;
    let u_span = f32(params[PARAM_SIDE_X]) - 7.0;
    let v_span = f32(params[PARAM_SIDE_Y]) - 7.0;
    if u_span <= 0.0 || v_span <= 0.0 || seed_module <= 0.0 {
        return;
    }
    let u = step_vector(u_edge / u_span);
    let v = step_vector(v_edge / v_span);

    // The window is the widest radius the host's doubling could have reached,
    // covered in one pass. Candidates are strided across the lanes so the
    // window size is independent of the workgroup size.
    let radius = i32(ceil(4.0 * seed_module));
    let side = u32(2 * radius + 1);
    let total = side * side;
    let module_max = param_f32(PARAM_MODULE_MAX);
    let limit = radius + 1;

    var mine = Candidate(false, 0.0, 0, 0.0, vec2<f32>(0.0, 0.0));
    for (var slot = local.x; slot < total; slot += WORKGROUP) {
        let dx = i32(slot % side) - radius;
        let dy = i32(slot / side) - radius;
        let at = prediction + u * f32(dx) + v * f32(dy);
        let candidate = evaluate(at, prediction, u, v, module_max, limit);
        if candidate.ok && (!mine.ok || candidate.distance < mine.distance) {
            mine = candidate;
        }
    }

    best_distance[local.x] = select(3.4e38, mine.distance, mine.ok);
    best_slot[local.x] = local.x;
    found_cx[local.x] = mine.centre.x;
    found_cy[local.x] = mine.centre.y;
    found_module[local.x] = mine.module;
    found_dir[local.x] = mine.dir;
    workgroupBarrier();

    // Nearest accepted candidate wins, which is what the host's alternating
    // cursors and first-acceptance stop were arranging for by construction.
    for (var stride = WORKGROUP / 2u; stride > 0u; stride >>= 1u) {
        if local.x < stride {
            let other = local.x + stride;
            if best_distance[other] < best_distance[local.x] {
                best_distance[local.x] = best_distance[other];
                best_slot[local.x] = best_slot[other];
            }
        }
        workgroupBarrier();
    }

    if local.x != 0u {
        return;
    }
    let base = index * CELL_WORDS;
    if best_distance[0] >= 3.4e38 {
        // Nothing accepted: leave the prediction in place so the next diagonal
        // still has an axis to extrapolate along, and mark the cell unfound so
        // the sampler does not treat it as a measurement.
        cells[base + CELL_FOUND] = 0u;
        cells[base + CELL_CX] = bitcast<u32>(prediction.x);
        cells[base + CELL_CY] = bitcast<u32>(prediction.y);
        cells[base + CELL_MODULE] = bitcast<u32>(seed_module);
        cells[base + CELL_DIR] = bitcast<u32>(0);
        return;
    }
    let winner = best_slot[0];
    cells[base + CELL_FOUND] = 1u;
    cells[base + CELL_CX] = bitcast<u32>(found_cx[winner]);
    cells[base + CELL_CY] = bitcast<u32>(found_cy[winner]);
    cells[base + CELL_MODULE] = bitcast<u32>(found_module[winner]);
    cells[base + CELL_DIR] = bitcast<u32>(found_dir[winner]);
}
