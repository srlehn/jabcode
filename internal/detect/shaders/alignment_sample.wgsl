// Samples an AP-corrected symbol directly from the resident balanced image.
// A module chooses the tightest rectangle covering its destination coordinate,
// so all modules fan out independently after the small AP-cell reduction.

const PARAM_WIDTH: u32 = 0u;
const PARAM_HEIGHT: u32 = 1u;
const PARAM_NAPX: u32 = 2u;
const PARAM_NAPY: u32 = 3u;
const PARAM_SIDE_X: u32 = 8u;
const PARAM_SIDE_Y: u32 = 9u;

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

const REGIME_FOOTPRINT: u32 = 1u;
const RESULT_MODULES: u32 = 1u;
const SAMPLE_COVERAGE: f32 = 0.7;
const MAX_RECTS: u32 = 64u;

@group(0) @binding(0) var<storage, read> pixels: array<u32>;
@group(0) @binding(1) var<storage, read_write> result: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@group(0) @binding(3) var<storage, read> rects: array<u32>;

fn rect_f32(rect: u32, field: u32) -> f32 {
    return bitcast<f32>(rects[rect * RECT_WORDS + field]);
}

fn rect_transform(rect: u32) -> array<f32, 9> {
    var transform: array<f32, 9>;
    for (var word = 0u; word < 9u; word += 1u) {
        transform[word] = rect_f32(rect, RECT_TRANSFORM + word);
    }
    return transform;
}

fn warp(transform: array<f32, 9>, x: f32, y: f32) -> vec2<f32> {
    let denominator = transform[2] * x + transform[5] * y + transform[8];
    return vec2<f32>(
        (transform[0] * x + transform[3] * y + transform[6]) / denominator,
        (transform[1] * x + transform[4] * y + transform[7]) / denominator,
    );
}

fn sample_offset(tap: u32, count: u32) -> vec2<f32> {
    let position = (f32(tap) + 0.5) / f32(count) - 0.5;
    return vec2<f32>(SAMPLE_COVERAGE * position, 1.0 - 2.0 * abs(position));
}

fn weight_sum(count: u32) -> f32 {
    var total = 0.0;
    for (var tap = 0u; tap < count; tap += 1u) {
        total += sample_offset(tap, count).y;
    }
    return total;
}

fn texel(x: i32, y: i32, width: i32) -> vec4<f32> {
    let word = pixels[u32(y * width + x)];
    return vec4<f32>(
        f32(word & 0xffu),
        f32((word >> 8u) & 0xffu),
        f32((word >> 16u) & 0xffu),
        f32((word >> 24u) & 0xffu),
    );
}

fn pack(value: vec4<f32>) -> u32 {
    let quantized = vec4<u32>(value + vec4<f32>(0.5));
    return quantized.x | (quantized.y << 8u) | (quantized.z << 16u) | (quantized.w << 24u);
}

fn sample_footprint(
    transform: array<f32, 9>,
    cx: f32,
    cy: f32,
    width: i32,
    height: i32,
    kx: u32,
    ky: u32,
) -> vec4<f32> {
    var total = vec4<f32>(0.0);
    for (var yi = 0u; yi < ky; yi += 1u) {
        let tap_y = sample_offset(yi, ky);
        var row = vec4<f32>(0.0);
        for (var xi = 0u; xi < kx; xi += 1u) {
            let tap_x = sample_offset(xi, kx);
            let q = warp(transform, cx + tap_x.x, cy + tap_y.x);
            let px = clamp(i32(q.x), 0, width - 1);
            let py = clamp(i32(q.y), 0, height - 1);
            row += tap_x.y * texel(px, py, width);
        }
        total += tap_y.y * row;
    }
    return total / (weight_sum(kx) * weight_sum(ky));
}

fn sample_centre(px: i32, py: i32, width: i32, height: i32) -> vec4<f32> {
    let cx = clamp(px, 0, width - 1);
    let cy = clamp(py, 0, height - 1);
    var total = vec4<f32>(0.0);
    for (var dx = -1; dx <= 1; dx += 1) {
        for (var dy = -1; dy <= 1; dy += 1) {
            var sx = cx + dx;
            var sy = cy + dy;
            if sx < 0 || sx >= width {
                sx = cx;
            }
            if sy < 0 || sy >= height {
                sy = cy;
            }
            total += texel(sx, sy, width);
        }
    }
    return total / 9.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let side_x = params[PARAM_SIDE_X];
    let side_y = params[PARAM_SIDE_Y];
    if id.x >= side_x * side_y {
        return;
    }
    let module_x = id.x % side_x;
    let module_y = id.x / side_x;
    let rect_count = (params[PARAM_NAPX] - 1u) * (params[PARAM_NAPY] - 1u);
    var selected = MAX_RECTS;
    var best_area = MAX_RECTS + 1u;
    for (var rect = 0u; rect < rect_count; rect += 1u) {
        let base = rect * RECT_WORDS;
        if rects[base + RECT_VALID] == 0u {
            continue;
        }
        let origin_x = rects[base + RECT_ORIGIN_X];
        let origin_y = rects[base + RECT_ORIGIN_Y];
        let size_x = rects[base + RECT_SIZE_X];
        let size_y = rects[base + RECT_SIZE_Y];
        if module_x < origin_x || module_x >= origin_x + size_x ||
            module_y < origin_y || module_y >= origin_y + size_y {
            continue;
        }
        let area = rects[base + RECT_AREA];
        if area <= best_area {
            best_area = area;
            selected = rect;
        }
    }
    if selected >= MAX_RECTS {
        atomicStore(&result[0], 0u);
        return;
    }

    let base = selected * RECT_WORDS;
    let local_x = f32(module_x - rects[base + RECT_ORIGIN_X]) + 0.5;
    let local_y = f32(module_y - rects[base + RECT_ORIGIN_Y]) + 0.5;
    let transform = rect_transform(selected);
    let centre = warp(transform, local_x, local_y);
    let width = i32(params[PARAM_WIDTH]);
    let height = i32(params[PARAM_HEIGHT]);
    if centre.x != centre.x || centre.y != centre.y ||
        centre.x <= -2.0 || centre.x >= f32(width + 1) ||
        centre.y <= -2.0 || centre.y >= f32(height + 1) {
        atomicStore(&result[0], 0u);
        return;
    }
    var value: vec4<f32>;
    if rects[base + RECT_REGIME] == REGIME_FOOTPRINT {
        value = sample_footprint(
            transform,
            local_x,
            local_y,
            width,
            height,
            rects[base + RECT_KX],
            rects[base + RECT_KY],
        );
    } else {
        value = sample_centre(i32(centre.x), i32(centre.y), width, height);
    }
    atomicStore(&result[RESULT_MODULES + module_y * side_x + module_x], pack(value));
}
