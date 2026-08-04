// Counts the modules along each of the four finder-to-finder edges by walking
// the connecting line one module at a time, re-centring every step on the most
// homogeneous window. This is the last stage that needed the whole balanced
// frame on the host, for a walk that reads a few hundred small windows of it.
//
// The walk is serial by construction: each step starts where the last one
// re-centred. What is parallel is everything inside a step - the candidate
// offsets a step chooses between, and the window each candidate has to measure.
// So an edge gets a workgroup rather than a lane: the lanes evaluate the
// candidates together, one lane picks the winner, and the workgroup carries the
// position forward. Four workgroups then walk all four edges at once, which the
// host could not do at all.
//
// The winner scan stays on one lane deliberately. It is a few dozen
// comparisons against thousands of pixel reads, and doing it in scan order is
// what reproduces the host's tie-breaking exactly: the unshifted position holds
// unless a candidate is strictly better, and among equals the most negative
// offset wins.

const WORKGROUP_SIZE: u32 = 64u;
const MAX_MODULE_STEPS: u32 = 142u;
const COUNT_DIVERGED: i32 = -1;

const PARAM_WIDTH: u32 = 0u;
const PARAM_HEIGHT: u32 = 1u;
// Each edge contributes ax, ay, module size a, bx, by, module size b.
const PARAM_EDGES: u32 = 2u;
const PARAM_EDGE_STRIDE: u32 = 6u;

@group(0) @binding(0) var<storage, read> pixels: array<u32>;
@group(0) @binding(1) var<storage, read_write> counts: array<i32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

// scores holds each candidate's homogeneity, lower being more homogeneous, and
// usable marks the candidates whose window met the image at all.
var<workgroup> scores: array<f32, WORKGROUP_SIZE>;
var<workgroup> usable: array<u32, WORKGROUP_SIZE>;
var<workgroup> walk: vec2<f32>;
var<workgroup> step_unit: vec2<f32>;
var<workgroup> step_module: f32;
var<workgroup> step_count: i32;
var<workgroup> walking: bool;
var<workgroup> best_score: f32;
var<workgroup> best_offset: i32;
var<workgroup> centre_usable: u32;

fn param_f32(index: u32) -> f32 {
    return bitcast<f32>(params[index]);
}

fn channels_of(x: i32, y: i32, width: i32) -> vec3<i32> {
    let word = pixels[u32(y * width + x)];
    return vec3<i32>(
        i32(word & 0xffu),
        i32((word >> 8u) & 0xffu),
        i32((word >> 16u) & 0xffu),
    );
}

// window_score returns the summed per-channel variance of the (2w+1)-square
// pixel window centred on the point, and whether the window met the image at
// all. Comparing that sum orders candidates exactly as comparing the host's
// root mean variance does, without the square root.
//
// Every sample is taken relative to a pixel from inside the window. That is
// what makes the f32 arithmetic trustworthy here: the raw sum of squares over a
// wide window overflows an exact integer accumulator, and computed directly it
// is a difference of two large nearly equal numbers, which is precisely the
// uniform window the walk most needs to recognise. Shifted by a member of the
// window, a uniform window gives exactly zero instead, and every partially
// uniform one keeps its significant digits.
fn window_score(centre: vec2<f32>, w: i32, width: i32, height: i32) -> vec2<f32> {
    let cx = i32(centre.x + 0.5);
    let cy = i32(centre.y + 0.5);
    let x0 = max(cx - w, 0);
    let x1 = min(cx + w, width - 1);
    let y0 = max(cy - w, 0);
    let y1 = min(cy + w, height - 1);
    if x0 > x1 || y0 > y1 {
        return vec2<f32>(0.0, 0.0);
    }
    let reference = channels_of(clamp(cx, x0, x1), clamp(cy, y0, y1), width);
    var sum = vec3<f32>(0.0);
    var square = vec3<f32>(0.0);
    for (var y = y0; y <= y1; y += 1) {
        for (var x = x0; x <= x1; x += 1) {
            let offset = vec3<f32>(channels_of(x, y, width) - reference);
            sum += offset;
            square += offset * offset;
        }
    }
    let n = f32((x1 - x0 + 1) * (y1 - y0 + 1));
    let variance = (square - sum * sum / n) / n;
    return vec2<f32>(variance.x + variance.y + variance.z, 1.0);
}

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let width = i32(params[PARAM_WIDTH]);
    let height = i32(params[PARAM_HEIGHT]);
    let edge = PARAM_EDGES + group.x * PARAM_EDGE_STRIDE;
    let a = vec2<f32>(param_f32(edge + 0u), param_f32(edge + 1u));
    let module_a = param_f32(edge + 2u);
    let b = vec2<f32>(param_f32(edge + 3u), param_f32(edge + 4u));
    let module_b = param_f32(edge + 5u);
    let span = b - a;
    let total = length(span);

    if lane == 0u {
        walk = a;
        step_count = 0;
        walking = total > 0.0 && module_a > 0.0 && module_b > 0.0;
        if !walking {
            counts[group.x] = COUNT_DIVERGED;
        }
    }
    workgroupBarrier();
    if !walking {
        return;
    }

    loop {
        if lane == 0u {
            let towards = b - walk;
            let remaining = length(towards);
            // The interpolation runs on the projection of the walk onto the
            // edge, so re-centring jitter cannot push it outside the segment.
            let travelled = dot(walk - a, span) / (total * total);
            let t = clamp(travelled, 0.0, 1.0);
            let lms = module_a * (1.0 - t) + module_b * t;
            step_module = lms;
            if remaining < lms * 0.5 {
                counts[group.x] = step_count;
                walking = false;
            } else if u32(step_count) >= MAX_MODULE_STEPS {
                counts[group.x] = COUNT_DIVERGED;
                walking = false;
            } else {
                step_unit = towards / remaining;
                walk += step_unit * lms;
            }
        }
        workgroupBarrier();
        if !walking {
            return;
        }

        // A shift is bounded by a quarter module so a re-centred point can
        // never cross into the neighbouring module.
        let reach = max(i32(step_module * 0.25 + 0.5), 1);
        let candidates = u32(2 * reach + 1);
        let origin = walk;

        // The unshifted candidate sets the bar first, and when its own window
        // misses the image the position holds regardless of the others - the
        // host's early return. Everything else has to beat it strictly.
        if lane == 0u {
            let centre = window_score(origin, reach, width, height);
            best_score = centre.x;
            best_offset = 0;
            centre_usable = u32(centre.y);
        }
        workgroupBarrier();

        if centre_usable != 0u {
            // A module wide enough to offer more candidates than there are
            // lanes is folded in chunks, still in offset order, so the tie
            // between equally homogeneous candidates falls the same way as a
            // single pass over all of them.
            for (var base = 0u; base < candidates; base += WORKGROUP_SIZE) {
                let slot = base + lane;
                if slot < candidates {
                    let offset = f32(i32(slot) - reach);
                    let score = window_score(origin + step_unit * offset, reach, width, height);
                    scores[lane] = score.x;
                    usable[lane] = u32(score.y);
                } else {
                    usable[lane] = 0u;
                }
                workgroupBarrier();
                if lane == 0u {
                    for (var i = 0u; i < WORKGROUP_SIZE; i += 1u) {
                        let offset = i32(base + i) - reach;
                        if usable[i] == 0u || offset == 0 {
                            continue;
                        }
                        if scores[i] < best_score {
                            best_score = scores[i];
                            best_offset = offset;
                        }
                    }
                }
                workgroupBarrier();
            }
            if lane == 0u {
                walk = origin + step_unit * f32(best_offset);
            }
        }
        if lane == 0u {
            step_count += 1;
        }
        workgroupBarrier();
    }
}
