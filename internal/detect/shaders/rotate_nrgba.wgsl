struct Params {
    source_width: u32,
    source_height: u32,
    crop_x: u32,
    crop_y: u32,
    crop_width: u32,
    crop_height: u32,
    destination_width: u32,
    destination_height: u32,
    cosine: f32,
    sine: f32,
}

@group(0) @binding(0) var<storage, read> source_pixels: array<u32>;
@group(0) @binding(1) var<storage, read_write> destination_pixels: array<u32>;
@group(0) @binding(2) var<storage, read> params: Params;

fn component(pixel: u32, shift: u32) -> f32 {
    return f32((pixel >> shift) & 0xffu);
}

fn bilinear_component(
    top_left: u32,
    top_right: u32,
    bottom_left: u32,
    bottom_right: u32,
    shift: u32,
    fraction_x: f32,
    fraction_y: f32,
) -> u32 {
    return u32(floor(
        component(top_left, shift) * (1.0 - fraction_x) * (1.0 - fraction_y) +
        component(top_right, shift) * fraction_x * (1.0 - fraction_y) +
        component(bottom_left, shift) * (1.0 - fraction_x) * fraction_y +
        component(bottom_right, shift) * fraction_x * fraction_y +
        0.5,
    ));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let output_x = id.x;
    let output_y = id.y;
    if output_x >= params.destination_width || output_y >= params.destination_height {
        return;
    }

    let source_center_x = f32(params.crop_width) / 2.0;
    let source_center_y = f32(params.crop_height) / 2.0;
    let destination_center_x = f32(params.destination_width) / 2.0;
    let destination_center_y = f32(params.destination_height) / 2.0;
    let delta_x = f32(output_x) - destination_center_x;
    let delta_y = f32(output_y) - destination_center_y;
    let source_x = params.cosine * delta_x + params.sine * delta_y + source_center_x;
    let source_y = -params.sine * delta_x + params.cosine * delta_y + source_center_y;
    let destination_index = output_y * params.destination_width + output_x;
    if source_x < 0.0 || source_y < 0.0 ||
       source_x > f32(params.crop_width - 1u) ||
       source_y > f32(params.crop_height - 1u) {
        destination_pixels[destination_index] = 0xffffffffu;
        return;
    }

    let x0 = u32(source_x);
    let y0 = u32(source_y);
    let x1 = min(x0 + 1u, params.crop_width - 1u);
    let y1 = min(y0 + 1u, params.crop_height - 1u);
    let fraction_x = source_x - f32(x0);
    let fraction_y = source_y - f32(y0);
    let absolute_x0 = params.crop_x + x0;
    let absolute_x1 = params.crop_x + x1;
    let absolute_y0 = params.crop_y + y0;
    let absolute_y1 = params.crop_y + y1;
    let top_left = source_pixels[absolute_y0 * params.source_width + absolute_x0];
    let top_right = source_pixels[absolute_y0 * params.source_width + absolute_x1];
    let bottom_left = source_pixels[absolute_y1 * params.source_width + absolute_x0];
    let bottom_right = source_pixels[absolute_y1 * params.source_width + absolute_x1];
    let red = bilinear_component(
        top_left, top_right, bottom_left, bottom_right,
        0u, fraction_x, fraction_y,
    );
    let green = bilinear_component(
        top_left, top_right, bottom_left, bottom_right,
        8u, fraction_x, fraction_y,
    );
    let blue = bilinear_component(
        top_left, top_right, bottom_left, bottom_right,
        16u, fraction_x, fraction_y,
    );
    destination_pixels[destination_index] =
        red | (green << 8u) | (blue << 16u) | 0xff000000u;
}
