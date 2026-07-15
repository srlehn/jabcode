struct Params {
    width: u32,
    height: u32,
    radius_x: u32,
    radius_y: u32,
}

@group(0) @binding(0) var<storage, read> linear_pixels: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> filtered_pixels: array<u32>;
@group(0) @binding(2) var<storage, read> original_pixels: array<u32>;
@group(0) @binding(3) var<storage, read> params: Params;

fn linear_to_srgb(component: f32) -> u32 {
    var srgb: f32;
    if component <= 0.0031308 {
        srgb = component * 12.92;
    } else {
        srgb = 1.055 * pow(component, 1.0 / 2.4) - 0.055;
    }
    return u32(clamp(srgb * 255.0 + 0.5, 0.0, 255.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    var sum = vec3<f32>(0.0);
    let radius = i32(params.radius_y);
    for (var offset = -radius; offset <= radius; offset += 1) {
        let y = u32(clamp(i32(id.y) + offset, 0, i32(params.height) - 1));
        sum += linear_pixels[y * params.width + id.x].xyz;
    }
    let window = f32(params.radius_y * 2u + 1u);
    let value = sum / window;
    let index = id.y * params.width + id.x;
    let red = linear_to_srgb(value.x);
    let green = linear_to_srgb(value.y);
    let blue = linear_to_srgb(value.z);
    filtered_pixels[index] =
        red |
        (green << 8u) |
        (blue << 16u) |
        (original_pixels[index] & 0xff000000u);
}
