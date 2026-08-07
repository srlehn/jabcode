// Samples the module grid from the resident balanced image: one invocation per
// module, against the perspective transform the host derived from the four
// finder centres.
//
// The host sampler walks modules in scan order so it can hoist a whole warped
// row and bail out the instant one module lands off the image. Neither shape
// survives the move: a lane owns exactly one module, so there is no row to
// hoist and nothing to carry between iterations, and a lane cannot abandon the
// grid its neighbours are still filling. The bail-out therefore becomes a
// device-wide reject flag that any lane may raise and the host reads once with
// the grid, which is the same accept/reject decision reached without a round
// trip.
//
// The footprint weights stay separable rather than materialized: the host
// builds the full ky*kx product grid once and reuses it for every module, but a
// lane would have to rebuild it, and factoring the row weight back out costs
// two multiplies instead. That also shortens the accumulator chain from every
// sample to one row of them, which is what keeps a f32 sum inside half a least
// significant bit of the f64 original.

const SAMPLE_COVERAGE: f32 = 0.7;
const REGIME_FOOTPRINT: u32 = 1u;

const PARAM_WIDTH: u32 = 0u;
const PARAM_HEIGHT: u32 = 1u;
const PARAM_SIDE_X: u32 = 2u;
const PARAM_SIDE_Y: u32 = 3u;
const PARAM_REGIME: u32 = 4u;
const PARAM_KX: u32 = 5u;
const PARAM_KY: u32 = 6u;
const PARAM_USE_DELTA: u32 = 7u;
const PARAM_TRANSFORM: u32 = 8u;
const PARAM_DELTA: u32 = 17u;

// Where this dispatch's block lands in the assembled grid, and the grid's own
// extent. A whole-symbol sample is the degenerate block at the origin whose
// destination extent is its own, so both paths run the same kernel.
const PARAM_DEST_X: u32 = 23u;
const PARAM_DEST_Y: u32 = 24u;
const PARAM_DEST_WIDTH: u32 = 25u;
const PARAM_DEST_HEIGHT: u32 = 26u;

// The reject flag shares the grid's buffer so the whole stage is one download.
// That makes every module write atomic too, which costs nothing here: each lane
// owns one word and no lane reads another's.
const RESULT_REJECTED: u32 = 0u;
const RESULT_MODULES: u32 = 1u;

@group(0) @binding(0) var<storage, read> pixels: array<u32>;
@group(0) @binding(1) var<storage, read_write> result: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

fn param_f32(index: u32) -> f32 {
    return bitcast<f32>(params[index]);
}

// warp applies the projective transform in the same left-to-right accumulation
// order as the host, so the two agree to f32's precision rather than drifting
// through a different association.
fn warp(x: f32, y: f32) -> vec2<f32> {
    let a11 = param_f32(PARAM_TRANSFORM + 0u);
    let a12 = param_f32(PARAM_TRANSFORM + 1u);
    let a13 = param_f32(PARAM_TRANSFORM + 2u);
    let a21 = param_f32(PARAM_TRANSFORM + 3u);
    let a22 = param_f32(PARAM_TRANSFORM + 4u);
    let a23 = param_f32(PARAM_TRANSFORM + 5u);
    let a31 = param_f32(PARAM_TRANSFORM + 6u);
    let a32 = param_f32(PARAM_TRANSFORM + 7u);
    let a33 = param_f32(PARAM_TRANSFORM + 8u);
    let denom = a13 * x + a23 * y + a33;
    return vec2<f32>(
        (a11 * x + a21 * y + a31) / denom,
        (a12 * x + a22 * y + a32) / denom,
    );
}

// channel_delta returns the sampling offset for one channel; alpha and any
// channel past the colour planes stay unshifted, matching the host.
fn channel_delta(channel: u32) -> vec2<f32> {
    if channel >= 3u {
        return vec2<f32>(0.0);
    }
    let base = PARAM_DELTA + channel * 2u;
    return vec2<f32>(param_f32(base), param_f32(base + 1u));
}

// sample_offset returns one footprint tap's module-space offset and its
// triangular weight: full at the module centre, approaching zero at the edge of
// the covered portion.
fn sample_offset(tap: u32, count: u32) -> vec2<f32> {
    let u = (f32(tap) + 0.5) / f32(count) - 0.5;
    return vec2<f32>(SAMPLE_COVERAGE * u, 1.0 - 2.0 * abs(u));
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

fn reject() {
    atomicStore(&result[RESULT_REJECTED], 1u);
}

// emit places one block-local module in the assembled grid. A block may hang
// past the grid on either axis, and those modules are dropped rather than
// wrapped: the block was still sampled, so an out-of-image module in the
// overhang rejects the whole grid exactly as it does on the host.
fn emit(block_x: u32, block_y: u32, value: vec4<f32>) {
    let dest_x = params[PARAM_DEST_X] + block_x;
    let dest_y = params[PARAM_DEST_Y] + block_y;
    let dest_width = params[PARAM_DEST_WIDTH];
    if dest_x >= dest_width || dest_y >= params[PARAM_DEST_HEIGHT] {
        return;
    }
    atomicStore(&result[RESULT_MODULES + dest_y * dest_width + dest_x], pack(value));
}

// sample_footprint averages the central covered portion of a module's warped
// footprint under the separable tent, which is the regime for modules large
// enough that a fixed kernel would ignore most of their source pixels.
fn sample_footprint(cx: f32, cy: f32, width: i32, height: i32) -> vec4<f32> {
    let kx = params[PARAM_KX];
    let ky = params[PARAM_KY];
    let use_delta = params[PARAM_USE_DELTA] != 0u;
    var total = vec4<f32>(0.0);
    for (var yi = 0u; yi < ky; yi += 1u) {
        let tap_y = sample_offset(yi, ky);
        var row = vec4<f32>(0.0);
        for (var xi = 0u; xi < kx; xi += 1u) {
            let tap_x = sample_offset(xi, kx);
            let q = warp(cx + tap_x.x, cy + tap_y.x);
            if use_delta {
                for (var channel = 0u; channel < 4u; channel += 1u) {
                    let d = channel_delta(channel);
                    let px = clamp(i32(q.x + d.x), 0, width - 1);
                    let py = clamp(i32(q.y + d.y), 0, height - 1);
                    row[channel] += tap_x.y * texel(px, py, width)[channel];
                }
                continue;
            }
            let px = clamp(i32(q.x), 0, width - 1);
            let py = clamp(i32(q.y), 0, height - 1);
            row += tap_x.y * texel(px, py, width);
        }
        total += tap_y.y * row;
    }
    return total / (weight_sum(kx) * weight_sum(ky));
}

// sample_centre is the small-module regime: a uniform 3x3 average over integer
// pixels, which measures better than a footprint once a module barely exceeds
// the kernel. Its out-of-range handling clamps only the one-pixel overhang and
// rejects anything further out, so the accept/reject decision stays the host's.
fn sample_centre(px: i32, py: i32, width: i32, height: i32) -> vec4<f32> {
    var out = vec4<f32>(0.0);
    for (var channel = 0u; channel < 4u; channel += 1u) {
        let d = channel_delta(channel);
        let cx = clamp(px + i32(round(d.x)), 0, width - 1);
        let cy = clamp(py + i32(round(d.y)), 0, height - 1);
        var sum = 0.0;
        for (var dx = -1; dx <= 1; dx += 1) {
            for (var dy = -1; dy <= 1; dy += 1) {
                var sx = cx + dx;
                var sy = cy + dy;
                if sx < 0 || sx > width - 1 {
                    sx = cx;
                }
                if sy < 0 || sy > height - 1 {
                    sy = cy;
                }
                sum += texel(sx, sy, width)[channel];
            }
        }
        out[channel] = sum / 9.0;
    }
    return out;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let side_x = params[PARAM_SIDE_X];
    let side_y = params[PARAM_SIDE_Y];
    if id.x >= side_x * side_y {
        return;
    }
    let width = i32(params[PARAM_WIDTH]);
    let height = i32(params[PARAM_HEIGHT]);
    let block_x = id.x % side_x;
    let block_y = id.x / side_x;
    let cx = f32(block_x) + 0.5;
    let cy = f32(block_y) + 0.5;
    let centre = warp(cx, cy);
    var mx = i32(centre.x);
    var my = i32(centre.y);

    if params[PARAM_REGIME] == REGIME_FOOTPRINT {
        // The footprint regime keeps the host's one-pixel tolerance on the
        // centre only; every tap is clamped into the image regardless.
        if mx < -1 || mx > width || my < -1 || my > height {
            reject();
            return;
        }
        emit(block_x, block_y, sample_footprint(cx, cy, width, height));
        return;
    }

    if mx < 0 || mx > width - 1 {
        if mx == -1 {
            mx = 0;
        } else if mx == width {
            mx = width - 1;
        } else {
            reject();
            return;
        }
    }
    if my < 0 || my > height - 1 {
        if my == -1 {
            my = 0;
        } else if my == height {
            my = height - 1;
        } else {
            reject();
            return;
        }
    }
    emit(block_x, block_y, sample_centre(mx, my, width, height));
}
