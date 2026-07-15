struct Params {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<storage, read> source_pixels: array<u32>;
@group(0) @binding(1) var<storage, read_write> balanced_pixels: array<u32>;
@group(0) @binding(2) var<storage, read> bounds: array<u32>;
@group(0) @binding(3) var<storage, read> params: Params;

fn balance(value: u32, channel: u32) -> u32 {
    let minimum = bounds[channel * 2u];
    let maximum = bounds[channel * 2u + 1u];
    if value < minimum {
        return 0u;
    }
    if value > maximum {
        return 255u;
    }
    if maximum <= minimum {
        return 0u;
    }
    return (value - minimum) * 255u / (maximum - minimum);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let index = id.y * params.width + id.x;
    let pixel = source_pixels[index];
    let red = balance(pixel & 0xffu, 0u);
    let green = balance((pixel >> 8u) & 0xffu, 1u);
    let blue = balance((pixel >> 16u) & 0xffu, 2u);
    balanced_pixels[index] =
        red | (green << 8u) | (blue << 16u) | (pixel & 0xff000000u);
}
