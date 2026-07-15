struct Params {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<storage, read> pixels: array<u32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let pixel = pixels[id.y * params.width + id.x];
    atomicAdd(&histogram[pixel & 0xffu], 1u);
    atomicAdd(&histogram[256u + ((pixel >> 8u) & 0xffu)], 1u);
    atomicAdd(&histogram[512u + ((pixel >> 16u) & 0xffu)], 1u);
}
