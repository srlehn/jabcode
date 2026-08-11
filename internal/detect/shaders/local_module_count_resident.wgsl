// Supplies finder edges from the resident batch decision. The frame extent
// comes from the binarizer control that already described the uploaded image.

struct EdgeInput {
    a: vec2<f32>,
    module_a: f32,
    b: vec2<f32>,
    module_b: f32,
}

const DECISION_HAVE: u32 = 0u;
const DECISION_CORNER_MISS: u32 = 5u;
const DECISION_ALTERNATIVES: u32 = 6u;
const DECISION_PATTERNS: u32 = 8u;
const DECISION_ALTERNATIVE_PATTERNS: u32 = 32u;
const DECISION_GEOMETRY: u32 = 81u;
const PAT_WORDS: u32 = 6u;
const PAT_X: u32 = 2u;
const PAT_Y: u32 = 3u;
const PAT_MODULE: u32 = 4u;
const EDGE_A: array<u32, 4> = array<u32, 4>(0u, 3u, 0u, 1u);
const EDGE_B: array<u32, 4> = array<u32, 4>(1u, 2u, 3u, 2u);

@group(0) @binding(2) var<storage, read> decision: array<u32>;
@group(0) @binding(3) var<storage, read> frame: array<u32>;

fn source_width() -> i32 {
    return i32(frame[0]);
}

fn source_height() -> i32 {
    return i32(frame[1]);
}

fn source_pattern_word(slot: u32, field: u32) -> u32 {
    let geometry = decision[DECISION_GEOMETRY];
    if geometry > 0u && geometry <= decision[DECISION_ALTERNATIVES] &&
        slot == decision[DECISION_CORNER_MISS] {
        return decision[
            DECISION_ALTERNATIVE_PATTERNS + (geometry - 1u) * PAT_WORDS + field
        ];
    }
    return decision[DECISION_PATTERNS + slot * PAT_WORDS + field];
}

fn source_pattern(slot: u32) -> vec3<f32> {
    return vec3<f32>(
        bitcast<f32>(source_pattern_word(slot, PAT_X)),
        bitcast<f32>(source_pattern_word(slot, PAT_Y)),
        bitcast<f32>(source_pattern_word(slot, PAT_MODULE)),
    );
}

fn source_edge(group: u32) -> EdgeInput {
    if decision[DECISION_HAVE] == 0u ||
        decision[DECISION_GEOMETRY] > decision[DECISION_ALTERNATIVES] {
        return EdgeInput(vec2<f32>(0.0), 0.0, vec2<f32>(0.0), 0.0);
    }
    let a = source_pattern(EDGE_A[group]);
    let b = source_pattern(EDGE_B[group]);
    return EdgeInput(a.xy, a.z, b.xy, b.z);
}
