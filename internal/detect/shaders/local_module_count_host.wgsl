// Supplies finder edges from a host-written parameter block for the
// compatibility and parity entry point.

struct EdgeInput {
    a: vec2<f32>,
    module_a: f32,
    b: vec2<f32>,
    module_b: f32,
}

const PARAM_WIDTH: u32 = 0u;
const PARAM_HEIGHT: u32 = 1u;
const PARAM_EDGES: u32 = 2u;
const PARAM_EDGE_STRIDE: u32 = 6u;

@group(0) @binding(2) var<storage, read> params: array<u32>;

fn source_width() -> i32 {
    return i32(params[PARAM_WIDTH]);
}

fn source_height() -> i32 {
    return i32(params[PARAM_HEIGHT]);
}

fn source_edge(group: u32) -> EdgeInput {
    let edge = PARAM_EDGES + group * PARAM_EDGE_STRIDE;
    return EdgeInput(
        vec2<f32>(bitcast<f32>(params[edge]), bitcast<f32>(params[edge + 1u])),
        bitcast<f32>(params[edge + 2u]),
        vec2<f32>(bitcast<f32>(params[edge + 3u]), bitcast<f32>(params[edge + 4u])),
        bitcast<f32>(params[edge + 5u]),
    );
}
