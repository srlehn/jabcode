struct Params {
    width: u32,
    height: u32,
    radius_x: u32,
    radius_y: u32,
}

@group(0) @binding(0) var<storage, read> pixels: array<u32>;
@group(0) @binding(1) var<storage, read_write> linear_pixels: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> params: Params;

fn srgb_to_linear(value: u32) -> f32 {
    let component = f32(value) / 255.0;
    if component <= 0.04045 {
        return component / 12.92;
    }
    return pow((component + 0.055) / 1.055, 2.4);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    var sum = vec3<f32>(0.0);
    let radius = i32(params.radius_x);
    for (var offset = -radius; offset <= radius; offset += 1) {
        let x = u32(clamp(i32(id.x) + offset, 0, i32(params.width) - 1));
        let pixel = pixels[id.y * params.width + x];
        sum += vec3<f32>(
            srgb_to_linear(pixel & 0xffu),
            srgb_to_linear((pixel >> 8u) & 0xffu),
            srgb_to_linear((pixel >> 16u) & 0xffu),
        );
    }
    let window = f32(params.radius_x * 2u + 1u);
    linear_pixels[id.y * params.width + id.x] = vec4<f32>(sum / window, 0.0);
}
