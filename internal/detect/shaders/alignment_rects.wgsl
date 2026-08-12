// Converts the located AP grid into one tight sampling rectangle per AP cell.
// Each lane owns one cell and performs its bounded expansion independently;
// the module sampler later chooses the smallest covering record, which is the
// final effect of the host path's widest-first overwrite order.

const PARAM_NAPX: u32 = 2u;
const PARAM_NAPY: u32 = 3u;
const PARAM_AP_POS_X: u32 = 19u;
const PARAM_AP_POS_Y: u32 = 28u;

const CELL_WORDS: u32 = 6u;
const CELL_FOUND: u32 = 0u;
const CELL_CX: u32 = 1u;
const CELL_CY: u32 = 2u;

const RECT_WORDS: u32 = 18u;
const RECT_VALID: u32 = 0u;
const RECT_AREA: u32 = 1u;
const RECT_ORIGIN_X: u32 = 2u;
const RECT_ORIGIN_Y: u32 = 3u;
const RECT_SIZE_X: u32 = 4u;
const RECT_SIZE_Y: u32 = 5u;
const RECT_TRANSFORM: u32 = 6u;
const RECT_REGIME: u32 = 15u;
const RECT_KX: u32 = 16u;
const RECT_KY: u32 = 17u;

const REGIME_CENTRE: u32 = 0u;
const REGIME_FOOTPRINT: u32 = 1u;
const SAMPLE_COVERAGE: f32 = 0.7;
const LEGACY_SAMPLE_BELOW: f32 = 9.0;
const MAX_SAMPLES_PER_AXIS: i32 = 32;

@group(0) @binding(0) var<storage, read> cells: array<u32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;
@group(0) @binding(2) var<storage, read_write> rects: array<u32>;

fn cell_found(x: u32, y: u32, n_ap_x: u32) -> bool {
    return cells[(y * n_ap_x + x) * CELL_WORDS + CELL_FOUND] != 0u;
}

fn cell_point(x: u32, y: u32, n_ap_x: u32) -> vec2<f32> {
    let base = (y * n_ap_x + x) * CELL_WORDS;
    return vec2<f32>(bitcast<f32>(cells[base + CELL_CX]), bitcast<f32>(cells[base + CELL_CY]));
}

fn finite(value: f32) -> bool {
    return value == value && abs(value) <= 3.402823e38;
}

fn square_to_quad(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>) -> array<f32, 9> {
    let delta3 = p0 - p1 + p2 - p3;
    var out: array<f32, 9>;
    if delta3.x == 0.0 && delta3.y == 0.0 {
        out[0] = p1.x - p0.x;
        out[1] = p1.y - p0.y;
        out[2] = 0.0;
        out[3] = p3.x - p0.x;
        out[4] = p3.y - p0.y;
        out[5] = 0.0;
        out[6] = p0.x;
        out[7] = p0.y;
        out[8] = 1.0;
        return out;
    }
    let delta1 = p1 - p2;
    let delta2 = p3 - p2;
    let denominator = delta1.x * delta2.y - delta2.x * delta1.y;
    let a13 = (delta3.x * delta2.y - delta2.x * delta3.y) / denominator;
    let a23 = (delta1.x * delta3.y - delta3.x * delta1.y) / denominator;
    out[0] = p1.x - p0.x + a13 * p1.x;
    out[1] = p1.y - p0.y + a13 * p1.y;
    out[2] = a13;
    out[3] = p3.x - p0.x + a23 * p3.x;
    out[4] = p3.y - p0.y + a23 * p3.y;
    out[5] = a23;
    out[6] = p0.x;
    out[7] = p0.y;
    out[8] = 1.0;
    return out;
}

fn source_transform(quad: array<f32, 9>, x0: f32, y0: f32, x1: f32, y1: f32) -> array<f32, 9> {
    let dx = x1 - x0;
    let dy = y1 - y0;
    var out: array<f32, 9>;
    out[0] = quad[0] / dx;
    out[1] = quad[1] / dx;
    out[2] = quad[2] / dx;
    out[3] = quad[3] / dy;
    out[4] = quad[4] / dy;
    out[5] = quad[5] / dy;
    out[6] = quad[6] - x0 * out[0] - y0 * out[3];
    out[7] = quad[7] - x0 * out[1] - y0 * out[4];
    out[8] = quad[8] - x0 * out[2] - y0 * out[5];
    return out;
}

fn warp(transform: array<f32, 9>, x: f32, y: f32) -> vec2<f32> {
    let denominator = transform[2] * x + transform[5] * y + transform[8];
    return vec2<f32>(
        (transform[0] * x + transform[3] * y + transform[6]) / denominator,
        (transform[1] * x + transform[4] * y + transform[7]) / denominator,
    );
}

fn sample_count(module: f32) -> u32 {
    return u32(clamp(i32(module * SAMPLE_COVERAGE + 0.5), 3, MAX_SAMPLES_PER_AXIS));
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
    let index = local.x;
    let base = index * RECT_WORDS;
    for (var word = 0u; word < RECT_WORDS; word += 1u) {
        rects[base + word] = 0u;
    }
    let n_ap_x = params[PARAM_NAPX];
    let n_ap_y = params[PARAM_NAPY];
    if n_ap_x < 2u || n_ap_y < 2u || index >= (n_ap_x - 1u) * (n_ap_y - 1u) {
        return;
    }
    let i = index / (n_ap_x - 1u);
    let j = index % (n_ap_x - 1u);
    var tl_x = j;
    var tl_y = i;
    var br_x = j + 1u;
    var br_y = i + 1u;
    var selected = false;
    let max_delta = (n_ap_x - 2u) + (n_ap_y - 2u);
    for (var delta = 0u; delta <= max_delta && !selected; delta += 1u) {
        for (var dy = 0u; dy <= min(delta, n_ap_y - 2u) && !selected; dy += 1u) {
            let dx = min(delta - dy, n_ap_x - 2u);
            for (var dy1 = 0u; dy1 <= dy && !selected; dy1 += 1u) {
                let dy2 = dy - dy1;
                for (var dx1 = 0u; dx1 <= dx && !selected; dx1 += 1u) {
                    let dx2 = dx - dx1;
                    tl_x = select(0u, j - dx1, j >= dx1);
                    tl_y = select(0u, i - dy1, i >= dy1);
                    br_x = min(j + 1u + dx2, n_ap_x - 1u);
                    br_y = min(i + 1u + dy2, n_ap_y - 1u);
                    selected = cell_found(tl_x, tl_y, n_ap_x) &&
                        cell_found(br_x, tl_y, n_ap_x) &&
                        cell_found(br_x, br_y, n_ap_x) &&
                        cell_found(tl_x, br_y, n_ap_x);
                }
            }
        }
    }
    if !selected {
        return;
    }

    let pos_tl_x = params[PARAM_AP_POS_X + tl_x];
    let pos_tl_y = params[PARAM_AP_POS_Y + tl_y];
    let pos_br_x = params[PARAM_AP_POS_X + br_x];
    let pos_br_y = params[PARAM_AP_POS_Y + br_y];
    var size_x = pos_br_x - pos_tl_x + 1u;
    var size_y = pos_br_y - pos_tl_y + 1u;
    var p0 = vec2<f32>(0.5, 0.5);
    var p1 = vec2<f32>(f32(size_x) - 0.5, 0.5);
    var p2 = vec2<f32>(f32(size_x) - 0.5, f32(size_y) - 0.5);
    var p3 = vec2<f32>(0.5, f32(size_y) - 0.5);
    if tl_y == 0u {
        size_y += 3u;
        p0.y = 3.5;
        p1.y = 3.5;
        p2.y = f32(size_y) - 0.5;
        p3.y = f32(size_y) - 0.5;
    }
    if br_y == n_ap_y - 1u {
        size_y += 3u;
        p2.y = f32(size_y) - 3.5;
        p3.y = f32(size_y) - 3.5;
    }
    if tl_x == 0u {
        size_x += 3u;
        p0.x = 3.5;
        p3.x = 3.5;
        p1.x = f32(size_x) - 0.5;
        p2.x = f32(size_x) - 0.5;
    }
    if br_x == n_ap_x - 1u {
        size_x += 3u;
        p1.x = f32(size_x) - 3.5;
        p2.x = f32(size_x) - 3.5;
    }
    let destination = square_to_quad(
        cell_point(tl_x, tl_y, n_ap_x),
        cell_point(br_x, tl_y, n_ap_x),
        cell_point(br_x, br_y, n_ap_x),
        cell_point(tl_x, br_y, n_ap_x),
    );
    let transform = source_transform(destination, p0.x, p0.y, p1.x, p3.y);
    for (var word = 0u; word < 9u; word += 1u) {
        if !finite(transform[word]) {
            return;
        }
    }
    let q00 = warp(transform, 0.0, 0.0);
    let q10 = warp(transform, f32(size_x), 0.0);
    let q01 = warp(transform, 0.0, f32(size_y));
    let q11 = warp(transform, f32(size_x), f32(size_y));
    let module_width = (distance(q10, q00) + distance(q11, q01)) / (2.0 * f32(size_x));
    let module_height = (distance(q01, q00) + distance(q11, q10)) / (2.0 * f32(size_y));
    if !finite(module_width) || !finite(module_height) || module_width <= 0.0 || module_height <= 0.0 {
        return;
    }

    let origin_x = select(pos_tl_x - 1u, 0u, tl_x == 0u);
    let origin_y = select(pos_tl_y - 1u, 0u, tl_y == 0u);
    rects[base + RECT_AREA] = (br_x - tl_x) * (br_y - tl_y);
    rects[base + RECT_ORIGIN_X] = origin_x;
    rects[base + RECT_ORIGIN_Y] = origin_y;
    rects[base + RECT_SIZE_X] = size_x;
    rects[base + RECT_SIZE_Y] = size_y;
    for (var word = 0u; word < 9u; word += 1u) {
        rects[base + RECT_TRANSFORM + word] = bitcast<u32>(transform[word]);
    }
    if min(module_width, module_height) < LEGACY_SAMPLE_BELOW {
        rects[base + RECT_REGIME] = REGIME_CENTRE;
        rects[base + RECT_KX] = 0u;
        rects[base + RECT_KY] = 0u;
    } else {
        rects[base + RECT_REGIME] = REGIME_FOOTPRINT;
        rects[base + RECT_KX] = sample_count(module_width);
        rects[base + RECT_KY] = sample_count(module_height);
    }
    rects[base + RECT_VALID] = 1u;
}
