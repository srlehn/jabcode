// Locates the alignment-pattern grid against the resident packed masks. The
// host search and this kernel accept on the same evidence, but they look for it
// in opposite shapes, at two different levels.
//
// Per candidate, the host walks outward from a prediction: a radius that
// doubles until something is found, two cursors taking turns along the module
// axis so candidates arrive ordered by distance, and the line abandoned at its
// first acceptance. Each of those avoids work on a processor that evaluates one
// candidate at a time, and none survives here. A lane owns one candidate
// offset, the whole window is evaluated whether or not an early candidate would
// have satisfied the host, and "nearest accepted candidate" falls out of a
// reduction over distances rather than out of the order candidates were visited
// in. The doubling radius goes with it: the widest radius the host could reach
// is simply the window, because covering it costs occupancy rather than time.
//
// Across cells the difference is larger. The host extrapolates each cell's
// prediction from the neighbours above and to its left, which is an accumulator
// chain: it serializes the grid into an anti-diagonal wavefront, and a wavefront
// dispatch holds at most min(nApX, nApY) cells, so a 3x3 grid would occupy three
// workgroups on a device that wants hundreds. It also accumulates error along
// the chain. Neither is necessary, because a cell's position is not actually a
// function of its neighbours: the grid sits at known module coordinates, and the
// four located finder centres give the perspective map from module space to
// image space. Every cell predicts itself in closed form from that map, so the
// whole grid searches in ONE dispatch with no dependencies at all.
//
// The neighbour information is not wasted, it is demoted to what it is good
// for. A second pass revisits only the cells that found nothing and corrects
// their prediction by the residual its located neighbours measured, which is a
// local warp correction the global quad cannot see. Two dispatches, both fully
// parallel, in place of a serial chain.
//
// The same window walk answers a second question that is not about a grid at
// all. Confirming a symbol's side version predicts where the first alignment
// pattern would sit for each candidate version and asks which prediction is
// borne out, so its candidates arrive as explicit image positions with their
// own axes rather than as module coordinates. That is the only difference, so
// it is a flag on the candidate's source and not a second kernel: the
// acceptance test is the thing worth sharing, and two copies of it is how the
// two searches would drift apart.

const PARAM_WIDTH: u32 = 0u;
const PARAM_HEIGHT: u32 = 1u;
const PARAM_NAPX: u32 = 2u;
const PARAM_NAPY: u32 = 3u;
const PARAM_MODE: u32 = 4u;
const PARAM_CORE_R: u32 = 5u;
const PARAM_CORE_G: u32 = 6u;
const PARAM_CORE_B: u32 = 7u;
const PARAM_SIDE_X: u32 = 8u;
const PARAM_SIDE_Y: u32 = 9u;
const PARAM_MODULE_MAX: u32 = 10u;
const PARAM_QUAD: u32 = 11u;
const PARAM_AP_POS_X: u32 = 19u;
const PARAM_AP_POS_Y: u32 = 28u;
const PARAM_TRANSFORM: u32 = 37u;
const PARAM_CORNER_MODULE: u32 = 46u;
// Non-zero when the candidates are an explicit list of predicted image
// positions rather than the cells of a grid.
const PARAM_EXPLICIT: u32 = 50u;
const PARAM_POSITION: u32 = 51u;

// One explicit candidate: predicted centre, seed module size, the two module
// axes as image vectors, and the run ceiling its caller measured.
const POSITION_WORDS: u32 = 8u;

// Pass 0 predicts every cell from the quad; pass 1 revisits only what pass 0
// left unfound.
const MODE_PREDICT: u32 = 0u;
const MODE_REFINE: u32 = 1u;
// Pass 2 folds the per-tile winners of the pass before it into one result per
// cell.
const MODE_REDUCE: u32 = 2u;

// A cell's candidate window is split across this many workgroups. One workgroup
// per cell leaves a device with tens of multiprocessors running a handful of
// them: a 5x2 grid has six interior cells, so the whole search occupied six
// workgroups and the memory latency of every candidate walk was exposed rather
// than hidden behind other work. Splitting the window is the only axis with
// enough parallelism in it, because the cell count is fixed by the symbol.
const TILES: u32 = 32u;

// One cell's result, and the state the refine pass corrects from.
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
@group(0) @binding(3) var<storage, read_write> tiles: array<u32>;

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
//
// centre is why a candidate is not simply reported where it was tried. A lane
// evaluates whole-step offsets from the prediction, so the accepted candidate
// nearest a prediction seeded a module away sits on the near edge of the core
// run rather than in it. The core run's own midpoint is the measurement, and it
// is what the host's cross-check returns after refining through the same walks.
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
    // How far the core run reaches against and along the step, so its midpoint
    // can be recovered once both sides are walked.
    var reach_back = 0;
    var reach_fwd = 0;
    // Outward in both directions: the first run of core colour is the pattern's
    // centre run, and the run beyond each end is its flank. A flank shorter
    // than MIN_RUN is folded back, which is how the host tolerates a single
    // mis-binarized sample inside a run.
    for (var side = 0; side < 2; side++) {
        let sign = select(1.0, -1.0, side == 0);
        var flank = 0;
        var in_core = true;
        var extent = 0;
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
                    extent = i;
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
                extent = i;
                flank = 0;
                in_core = true;
                continue;
            }
            break;
        }
        // Side zero walks against the step, so its flank is the one the caller
        // calls right and its extent is the run's backward reach.
        if side == 0 {
            runs.right = flank;
            reach_back = extent;
        } else {
            runs.left = flank;
            reach_fwd = extent;
        }
    }
    runs.centre = 0.5 * f32(reach_fwd - reach_back);
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
    // The candidate's own distance orders the reduction, so the nearest
    // accepted offset still wins; what it reports is the core run's measured
    // midpoint, which is a fraction of a module away from where it was tried.
    // Both channels measure it, so their mean is the centre along each axis.
    out.distance = distance(at, prediction);
    out.centre = at +
        u * (0.5 * (ur.centre + ub.centre)) +
        v * (0.5 * (vr.centre + vb.centre));
    return out;
}

// warp maps a point in symbol module space to image space through the
// perspective the four located finder centres define. This is what replaces the
// neighbour chain: it is exact under perspective, it costs the same for every
// cell, and no cell has to wait for another.
fn warp(p: vec2<f32>) -> vec2<f32> {
    let a11 = param_f32(PARAM_TRANSFORM);
    let a12 = param_f32(PARAM_TRANSFORM + 1u);
    let a13 = param_f32(PARAM_TRANSFORM + 2u);
    let a21 = param_f32(PARAM_TRANSFORM + 3u);
    let a22 = param_f32(PARAM_TRANSFORM + 4u);
    let a23 = param_f32(PARAM_TRANSFORM + 5u);
    let a31 = param_f32(PARAM_TRANSFORM + 6u);
    let a32 = param_f32(PARAM_TRANSFORM + 7u);
    let a33 = param_f32(PARAM_TRANSFORM + 8u);
    let denom = a13 * p.x + a23 * p.y + a33;
    return vec2<f32>(
        (a11 * p.x + a21 * p.y + a31) / denom,
        (a12 * p.x + a22 * p.y + a32) / denom,
    );
}

// seed_module interpolates the module size bilinearly across the quad. It only
// has to be good enough to size the search window and the run ceiling; what the
// cell reports is measured, not interpolated.
fn seed_module_at(tx: f32, ty: f32) -> f32 {
    let m00 = param_f32(PARAM_CORNER_MODULE);
    let m10 = param_f32(PARAM_CORNER_MODULE + 1u);
    let m11 = param_f32(PARAM_CORNER_MODULE + 2u);
    let m01 = param_f32(PARAM_CORNER_MODULE + 3u);
    let top = m00 + (m10 - m00) * tx;
    let bottom = m01 + (m11 - m01) * tx;
    return top + (bottom - top) * ty;
}

// neighbour_residual averages the offset between where the quad predicted a
// located neighbour and where it actually was. Over a cell the global quad
// misplaced, that residual is the local warp the quad could not represent.
fn neighbour_residual(i: u32, j: u32, n_ap_x: u32, n_ap_y: u32) -> vec2<f32> {
    var sum = vec2<f32>(0.0, 0.0);
    var count = 0.0;
    for (var di = -1; di <= 1; di++) {
        for (var dj = -1; dj <= 1; dj++) {
            if di == 0 && dj == 0 {
                continue;
            }
            let ni = i32(i) + di;
            let nj = i32(j) + dj;
            if ni < 0 || nj < 0 || u32(ni) >= n_ap_y || u32(nj) >= n_ap_x {
                continue;
            }
            let index = u32(ni) * n_ap_x + u32(nj);
            if cells[index * CELL_WORDS + CELL_FOUND] == 0u {
                continue;
            }
            let tx = f32(params[PARAM_AP_POS_X + u32(nj)]);
            let ty = f32(params[PARAM_AP_POS_Y + u32(ni)]);
            let predicted = warp(vec2<f32>(tx, ty));
            let actual = vec2<f32>(cell_f32(index, CELL_CX), cell_f32(index, CELL_CY));
            sum += actual - predicted;
            count += 1.0;
        }
    }
    if count == 0.0 {
        return vec2<f32>(0.0, 0.0);
    }
    return sum / count;
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
    let mode = params[PARAM_MODE];
    let explicit = params[PARAM_EXPLICIT] != 0u;
    let cells = n_ap_x * n_ap_y;
    // The search passes spread one cell's window across TILES workgroups; the
    // fold that follows them runs one workgroup per cell.
    var index = group.x;
    var tile = 0u;
    if mode != MODE_REDUCE {
        index = group.x / TILES;
        tile = group.x % TILES;
    }
    if index >= cells {
        return;
    }
    let i = index / n_ap_x;
    let j = index % n_ap_x;
    // The four quad corners are the finder measurements themselves, seeded by
    // the host and never searched. An explicit candidate list is not a grid and
    // so has no corners to skip.
    if !explicit {
        let corner = (i == 0u || i == n_ap_y - 1u) && (j == 0u || j == n_ap_x - 1u);
        if corner {
            return;
        }
    }

    if mode == MODE_REDUCE {
        fold_tiles(index, local.x);
        return;
    }
    // The refine pass exists for the cells the quad could not place. A cell that
    // already found its pattern keeps it.
    if !explicit && cells_found(index) {
        if local.x == 0u {
            clear_tile(index, tile);
        }
        return;
    }

    var prediction: vec2<f32>;
    var u: vec2<f32>;
    var v: vec2<f32>;
    var seed_module = 0.0;
    var module_max = param_f32(PARAM_MODULE_MAX);
    if explicit {
        let base = PARAM_POSITION + index * POSITION_WORDS;
        prediction = vec2<f32>(param_f32(base), param_f32(base + 1u));
        seed_module = param_f32(base + 2u);
        u = step_vector(vec2<f32>(param_f32(base + 3u), param_f32(base + 4u)));
        v = step_vector(vec2<f32>(param_f32(base + 5u), param_f32(base + 6u)));
        module_max = param_f32(base + 7u);
    } else {
        let module_x = f32(params[PARAM_AP_POS_X + j]);
        let module_y = f32(params[PARAM_AP_POS_Y + i]);
        prediction = warp(vec2<f32>(module_x, module_y));
        if mode == MODE_REFINE {
            prediction += neighbour_residual(i, j, n_ap_x, n_ap_y);
        }

        let tx = module_x / f32(params[PARAM_SIDE_X]);
        let ty = module_y / f32(params[PARAM_SIDE_Y]);
        seed_module = seed_module_at(tx, ty);

        // The module axes at this cell, each interpolated between the quad's two
        // opposite edges at the cell's own position. Long-baseline and local at
        // once: axes taken from the nearest neighbours would span only a few
        // modules, turning a fraction of a pixel of centre error into a visible
        // tilt.
        let u_span = f32(params[PARAM_SIDE_X]) - 7.0;
        let v_span = f32(params[PARAM_SIDE_Y]) - 7.0;
        if u_span <= 0.0 || v_span <= 0.0 {
            seed_module = 0.0;
        } else {
            let q0 = quad(0u);
            let q1 = quad(1u);
            let q2 = quad(2u);
            let q3 = quad(3u);
            let u_edge = (q1 - q0) + ((q2 - q3) - (q1 - q0)) * ty;
            let v_edge = (q3 - q0) + ((q2 - q1) - (q3 - q0)) * tx;
            u = step_vector(u_edge / u_span);
            v = step_vector(v_edge / v_span);
        }
    }
    if seed_module <= 0.0 {
        if local.x == 0u {
            clear_tile(index, tile);
        }
        return;
    }

    let radius = i32(ceil(4.0 * seed_module));
    let side = u32(2 * radius + 1);
    let total = side * side;
    let limit = radius + 1;

    // Candidates are interleaved across the tiles rather than given to each in
    // one contiguous run, so neighbouring lanes of one workgroup still walk
    // neighbouring pixels and the reads stay coalesced.
    var mine = Candidate(false, 0.0, 0, 0.0, vec2<f32>(0.0, 0.0));
    let stride = TILES * WORKGROUP;
    for (var slot = tile * WORKGROUP + local.x; slot < total; slot += stride) {
        let dx = i32(slot % side) - radius;
        let dy = i32(slot / side) - radius;
        let at = prediction + u * f32(dx) + v * f32(dy);
        let candidate = evaluate(at, prediction, u, v, module_max, limit);
        if candidate.ok && (!mine.ok || candidate.distance < mine.distance) {
            mine = candidate;
        }
    }
    let winner = reduce_lanes(mine, local.x);
    if local.x != 0u {
        return;
    }
    // The prediction is carried in the tile record so the fold can fall back to
    // it when no tile accepted anything, without recomputing the warp.
    write_tile(index, tile, winner, prediction, seed_module);
}

// reduce_lanes folds one workgroup's candidates down to the accepted one
// nearest the prediction, which is what the host's alternating cursors and
// first-acceptance stop were arranging for by construction.
fn reduce_lanes(mine: Candidate, lane: u32) -> Candidate {
    best_distance[lane] = select(3.4e38, mine.distance, mine.ok);
    best_slot[lane] = lane;
    found_cx[lane] = mine.centre.x;
    found_cy[lane] = mine.centre.y;
    found_module[lane] = mine.module;
    found_dir[lane] = mine.dir;
    workgroupBarrier();
    for (var stride = WORKGROUP / 2u; stride > 0u; stride >>= 1u) {
        if lane < stride {
            let other = lane + stride;
            if best_distance[other] < best_distance[lane] {
                best_distance[lane] = best_distance[other];
                best_slot[lane] = best_slot[other];
            }
        }
        workgroupBarrier();
    }
    if best_distance[0] >= 3.4e38 {
        return Candidate(false, 0.0, 0, 0.0, vec2<f32>(0.0, 0.0));
    }
    let at = best_slot[0];
    return Candidate(
        true, found_module[at], found_dir[at], best_distance[at],
        vec2<f32>(found_cx[at], found_cy[at]),
    );
}

fn cells_found(index: u32) -> bool {
    return cells[index * CELL_WORDS + CELL_FOUND] != 0u;
}

fn tile_base(index: u32, tile: u32) -> u32 {
    return (index * TILES + tile) * CELL_WORDS;
}

fn clear_tile(index: u32, tile: u32) {
    let base = tile_base(index, tile);
    tiles[base + CELL_FOUND] = 0u;
    tiles[base + CELL_MODULE] = 0u;
}

fn write_tile(index: u32, tile: u32, winner: Candidate, prediction: vec2<f32>, seed: f32) {
    let base = tile_base(index, tile);
    tiles[base + CELL_FOUND] = select(0u, 1u, winner.ok);
    tiles[base + CELL_CX] = bitcast<u32>(select(prediction.x, winner.centre.x, winner.ok));
    tiles[base + CELL_CY] = bitcast<u32>(select(prediction.y, winner.centre.y, winner.ok));
    tiles[base + CELL_MODULE] = bitcast<u32>(select(seed, winner.module, winner.ok));
    tiles[base + CELL_DIR] = bitcast<u32>(select(0, winner.dir, winner.ok));
    tiles[base + 5u] = bitcast<u32>(winner.distance);
}

// fold_tiles picks the tile whose accepted candidate sits nearest the
// prediction. A cell no tile accepted keeps the prediction any tile recorded,
// so the refine pass and the sampler still have an axis to work from.
fn fold_tiles(index: u32, lane: u32) {
    var mine = Candidate(false, 0.0, 0, 0.0, vec2<f32>(0.0, 0.0));
    var fallback = vec2<f32>(0.0, 0.0);
    var fallback_module = 0.0;
    for (var tile = lane; tile < TILES; tile += WORKGROUP) {
        let base = tile_base(index, tile);
        let module = bitcast<f32>(tiles[base + CELL_MODULE]);
        if module > 0.0 && fallback_module == 0.0 {
            fallback = vec2<f32>(
                bitcast<f32>(tiles[base + CELL_CX]),
                bitcast<f32>(tiles[base + CELL_CY]),
            );
            fallback_module = module;
        }
        if tiles[base + CELL_FOUND] == 0u {
            continue;
        }
        let candidate = Candidate(
            true, module, bitcast<i32>(tiles[base + CELL_DIR]),
            bitcast<f32>(tiles[base + 5u]),
            vec2<f32>(bitcast<f32>(tiles[base + CELL_CX]), bitcast<f32>(tiles[base + CELL_CY])),
        );
        if !mine.ok || candidate.distance < mine.distance {
            mine = candidate;
        }
    }
    // The fallback prediction is identical in every tile of a cell, so lane
    // zero's copy is the cell's.
    let winner = reduce_lanes(mine, lane);
    if lane != 0u {
        return;
    }
    let base = index * CELL_WORDS;
    if !winner.ok {
        if fallback_module > 0.0 {
            cells[base + CELL_FOUND] = 0u;
            cells[base + CELL_CX] = bitcast<u32>(fallback.x);
            cells[base + CELL_CY] = bitcast<u32>(fallback.y);
            cells[base + CELL_MODULE] = bitcast<u32>(fallback_module);
            cells[base + CELL_DIR] = bitcast<u32>(0);
        }
        return;
    }
    cells[base + CELL_FOUND] = 1u;
    cells[base + CELL_CX] = bitcast<u32>(winner.centre.x);
    cells[base + CELL_CY] = bitcast<u32>(winner.centre.y);
    cells[base + CELL_MODULE] = bitcast<u32>(winner.module);
    cells[base + CELL_DIR] = bitcast<u32>(winner.dir);
}
