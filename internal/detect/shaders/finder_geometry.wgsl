// Converts a resident finder decision and four parallel edge walks into the
// sampler's control block. One lane resolves the small geometry decision; the
// module grid itself remains one invocation per module.

const DECISION_HAVE: u32 = 0u;
const DECISION_DECLINED: u32 = 2u;
const DECISION_CORNER_SOURCE: u32 = 4u;
const DECISION_CORNER_MISS: u32 = 5u;
const DECISION_ALTERNATIVES: u32 = 6u;
const DECISION_PATTERNS: u32 = 8u;
const DECISION_ALTERNATIVE_PATTERNS: u32 = 32u;
const DECISION_SCAN: u32 = 80u;
const DECISION_GEOMETRY: u32 = 81u;
const PAT_WORDS: u32 = 6u;
const PAT_X: u32 = 2u;
const PAT_Y: u32 = 3u;
const PAT_MODULE: u32 = 4u;
const EDGE_A: array<u32, 4> = array<u32, 4>(0u, 3u, 0u, 1u);
const EDGE_B: array<u32, 4> = array<u32, 4>(1u, 2u, 3u, 2u);

const PARAM_WIDTH: u32 = 0u;
const PARAM_HEIGHT: u32 = 1u;
const PARAM_SIDE_X: u32 = 2u;
const PARAM_SIDE_Y: u32 = 3u;
const PARAM_REGIME: u32 = 4u;
const PARAM_KX: u32 = 5u;
const PARAM_KY: u32 = 6u;
const PARAM_TRANSFORM: u32 = 8u;
const PARAM_DEST_X: u32 = 23u;
const PARAM_DEST_Y: u32 = 24u;
const PARAM_DEST_WIDTH: u32 = 25u;
const PARAM_DEST_HEIGHT: u32 = 26u;

const CONTROL_GEOMETRY: u32 = 0u;
const CONTROL_CORNER: u32 = 1u;
const CONTROL_DEGREES: u32 = 2u;
const CONTROL_VALID: u32 = 3u;
const CONTROL_PATTERNS: u32 = 4u;
const CORNER_CONTEXTUAL: u32 = 4u;

const REGIME_FOOTPRINT: u32 = 1u;
const SAMPLE_COVERAGE: f32 = 0.7;
const LEGACY_SAMPLE_BELOW: f32 = 9.0;
const MAX_SAMPLES_PER_AXIS: i32 = 32;
const MODULE_RATIO_SLACK: f32 = 1.4;

struct RoundedSide {
    size: i32,
    flag: i32,
}

struct EdgeEstimate {
    size: i32,
    flag: i32,
    distance_size: i32,
    distance_flag: i32,
    rounding: bool,
    agree: bool,
    module_ratio: f32,
}

@group(0) @binding(0) var<storage, read> decision: array<u32>;
@group(0) @binding(1) var<storage, read> counts: array<i32>;
@group(0) @binding(2) var<storage, read> frame: array<u32>;
@group(0) @binding(3) var<storage, read_write> sample: array<u32>;
@group(0) @binding(4) var<storage, read_write> result: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> indirect: array<u32>;
@group(0) @binding(6) var<storage, read_write> control: array<u32>;

fn pattern_word(slot: u32, field: u32) -> u32 {
    let geometry = decision[DECISION_GEOMETRY];
    if geometry > 0u && geometry <= decision[DECISION_ALTERNATIVES] &&
        slot == decision[DECISION_CORNER_MISS] {
        return decision[
            DECISION_ALTERNATIVE_PATTERNS + (geometry - 1u) * PAT_WORDS + field
        ];
    }
    return decision[DECISION_PATTERNS + slot * PAT_WORDS + field];
}

fn pattern(slot: u32) -> vec3<f32> {
    return vec3<f32>(
		bitcast<f32>(pattern_word(slot, PAT_X)),
		bitcast<f32>(pattern_word(slot, PAT_Y)),
		bitcast<f32>(pattern_word(slot, PAT_MODULE)),
    );
}

fn distance_modules(a: vec3<f32>, b: vec3<f32>) -> i32 {
    let delta = abs(b.xy - a.xy);
    let distance = length(delta);
    if distance <= 0.0 || a.z <= 0.0 || b.z <= 0.0 {
        return -1;
    }
    let cosine = max(delta.x, delta.y) / distance;
    let module = (a.z + b.z) * cosine * 0.5;
    if module <= 0.0 {
        return -1;
    }
    return i32(distance / module + 0.5);
}

fn round_side(raw: i32) -> RoundedSide {
    var size = raw;
    var flag = 1;
    switch size & 3 {
        case 0: { size += 1; }
        case 2: { size -= 1; }
        case 3: {
            size -= 2;
            flag = 0;
        }
        default: {}
    }
    if size < 21 || size > 145 {
        return RoundedSide(-1, -1);
    }
    return RoundedSide(size, flag);
}

fn edge_estimate(edge: u32) -> EdgeEstimate {
    let a = pattern(EDGE_A[edge]);
    let b = pattern(EDGE_B[edge]);
    let walked = counts[edge];
    let distance = distance_modules(a, b);
    let walk_side = round_side(walked + 7);
    let distance_side = round_side(distance + 7);
    var size = walk_side.size;
    var flag = walk_side.flag;
    if walked <= 0 {
        size = distance_side.size;
        flag = distance_side.flag;
    }
    let module_min = min(a.z, b.z);
    let module_max = max(a.z, b.z);
    let module_ratio = select(
        bitcast<f32>(0x7f800000u),
        module_max / module_min,
        module_min > 0.0,
    );
    return EdgeEstimate(
        size,
        flag,
        distance_side.size,
        distance_side.flag,
        distance - 1 <= walked && walked <= distance + 1,
        walk_side.size > 0 && walk_side.size == distance_side.size,
        module_ratio,
    );
}

fn choose_axis(a: EdgeEstimate, b: EdgeEstimate) -> i32 {
    if a.flag == -1 && b.flag == -1 {
        return -1;
    }
    if a.flag == -1 {
        return b.size;
    }
    if b.flag == -1 {
        return a.size;
    }
    if a.distance_size == b.distance_size &&
        a.distance_flag == 1 && b.distance_flag == 1 &&
        a.flag < 1 && b.flag < 1 && a.rounding && b.rounding {
        return a.distance_size;
    }
    if a.size == b.size {
        return a.size;
    }
    if a.flag != b.flag {
        return select(b.size, a.size, a.flag > b.flag);
    }
    if a.agree != b.agree {
        return select(b.size, a.size, a.agree);
    }
    let a_ok = a.module_ratio <= MODULE_RATIO_SLACK;
    let b_ok = b.module_ratio <= MODULE_RATIO_SLACK;
    if a_ok != b_ok {
        return select(b.size, a.size, a_ok);
    }
    return min(a.size, b.size);
}

fn square_to_quad() -> array<f32, 9> {
    let p0 = pattern(0u).xy;
    let p1 = pattern(1u).xy;
    let p2 = pattern(2u).xy;
    let p3 = pattern(3u).xy;
    let delta3 = p0 - p1 + p2 - p3;
    var out: array<f32, 9>;
    if delta3.x == 0.0 && delta3.y == 0.0 {
        out[0] = p1.x - p0.x;
        out[1] = p1.y - p0.y;
        out[2] = 0.0;
        out[3] = p2.x - p1.x;
        out[4] = p2.y - p1.y;
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

fn symbol_transform(side_x: i32, side_y: i32) -> array<f32, 9> {
    let quad = square_to_quad();
    let source = 3.5;
    let span_x = f32(side_x - 7);
    let span_y = f32(side_y - 7);
    var out: array<f32, 9>;
    out[0] = quad[0] / span_x;
    out[1] = quad[1] / span_x;
    out[2] = quad[2] / span_x;
    out[3] = quad[3] / span_y;
    out[4] = quad[4] / span_y;
    out[5] = quad[5] / span_y;
    out[6] = quad[6] - source * out[0] - source * out[3];
    out[7] = quad[7] - source * out[1] - source * out[4];
    out[8] = quad[8] - source * out[2] - source * out[5];
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

fn finite(value: f32) -> bool {
    return value == value && abs(value) <= 3.402823e38;
}

@compute @workgroup_size(1)
fn main() {
    if decision[DECISION_HAVE] == 0u || decision[DECISION_DECLINED] != 0u {
        return;
    }
	let geometry = decision[DECISION_GEOMETRY];
	if geometry > decision[DECISION_ALTERNATIVES] {
		return;
	}
    let side_x = choose_axis(edge_estimate(0u), edge_estimate(1u));
    let side_y = choose_axis(edge_estimate(2u), edge_estimate(3u));
    if side_x < 21 || side_y < 21 || side_x > 145 || side_y > 145 {
        return;
    }
    let width = frame[0];
    let height = frame[1];
    if width == 0u || height == 0u {
        return;
    }

    let transform = symbol_transform(side_x, side_y);
    for (var word = 0u; word < 9u; word += 1u) {
        if !finite(transform[word]) {
            return;
        }
    }
    sample[PARAM_WIDTH] = width;
    sample[PARAM_HEIGHT] = height;
    sample[PARAM_SIDE_X] = u32(side_x);
    sample[PARAM_SIDE_Y] = u32(side_y);
    sample[PARAM_DEST_X] = 0u;
    sample[PARAM_DEST_Y] = 0u;
    sample[PARAM_DEST_WIDTH] = u32(side_x);
    sample[PARAM_DEST_HEIGHT] = u32(side_y);
    for (var word = 0u; word < 9u; word += 1u) {
        sample[PARAM_TRANSFORM + word] = bitcast<u32>(transform[word]);
    }

    let q00 = warp(transform, 0.0, 0.0);
    let q10 = warp(transform, f32(side_x), 0.0);
    let q01 = warp(transform, 0.0, f32(side_y));
    let q11 = warp(transform, f32(side_x), f32(side_y));
    let module_width = (distance(q10, q00) + distance(q11, q01)) / (2.0 * f32(side_x));
    let module_height = (distance(q01, q00) + distance(q11, q10)) / (2.0 * f32(side_y));
    if !finite(module_width) || !finite(module_height) ||
        module_width <= 0.0 || module_height <= 0.0 {
        return;
    }
    if min(module_width, module_height) >= LEGACY_SAMPLE_BELOW {
        sample[PARAM_REGIME] = REGIME_FOOTPRINT;
        sample[PARAM_KX] = sample_count(module_width);
        sample[PARAM_KY] = sample_count(module_height);
    }

    control[CONTROL_GEOMETRY] = geometry;
    control[CONTROL_CORNER] = select(
        decision[DECISION_CORNER_SOURCE], CORNER_CONTEXTUAL, geometry > 0u,
    );
    control[CONTROL_DEGREES] = decision[DECISION_SCAN];
    control[CONTROL_VALID] = 1u;
    for (var slot = 0u; slot < 4u; slot += 1u) {
        for (var word = 0u; word < PAT_WORDS; word += 1u) {
            control[CONTROL_PATTERNS + slot * PAT_WORDS + word] =
                pattern_word(slot, word);
        }
    }

    atomicStore(&result[0], 1u);
    indirect[0] = (u32(side_x * side_y) + 63u) / 64u;
    indirect[1] = 1u;
    indirect[2] = 1u;
    indirect[3] = 1u;
    indirect[4] = 1u;
    indirect[5] = 1u;
}
