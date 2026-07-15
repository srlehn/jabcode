@group(0) @binding(0) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> bounds: array<u32>;

@compute @workgroup_size(3)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let channel = id.x;
    if channel >= 3u {
        return;
    }
    let base = channel * 256u;
    var minimum = 0u;
    for (var value = 0u; value < 256u; value++) {
        if atomicLoad(&histogram[base + value]) > 20u {
            minimum = value;
            break;
        }
    }
    var maximum = 255u;
    for (var value = 255i; value >= 0i; value--) {
        if atomicLoad(&histogram[base + u32(value)]) > 20u {
            maximum = u32(value);
            break;
        }
    }
    bounds[channel * 2u] = minimum;
    bounds[channel * 2u + 1u] = maximum;
}
