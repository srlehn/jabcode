struct Params {
    width: u32,
    height: u32,
    block_size: u32,
    blocks_x: u32,
    blocks_y: u32,
    flags: u32,
    fixed_r: f32,
    fixed_g: f32,
    fixed_b: f32,
}

@group(0) @binding(0) var<storage, read> input_masks: array<u32>;
@group(0) @binding(1) var<storage, read_write> packed_masks: array<u32>;
@group(0) @binding(2) var<storage, read> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let packed_index = id.x;
    let pixel_count = params.width * params.height;
    let packed_count = (pixel_count + 7u) / 8u;
    if packed_index >= packed_count {
        return;
    }

    var packed = 0u;
    let first = packed_index * 8u;
    for (var lane = 0u; lane < 8u; lane++) {
        let pixel = first + lane;
        if pixel < pixel_count {
            packed = packed | ((input_masks[pixel] & 7u) << (lane * 3u));
        }
    }
    packed_masks[packed_index] = packed;
}
