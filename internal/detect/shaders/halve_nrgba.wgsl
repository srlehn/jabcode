struct Params {
    source_width: u32,
    source_height: u32,
    destination_width: u32,
    destination_height: u32,
}

@group(0) @binding(0) var<storage, read> source_pixels: array<u32>;
@group(0) @binding(1) var<storage, read_write> destination_pixels: array<u32>;
@group(0) @binding(2) var<storage, read> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let output_x = id.x;
    let output_y = id.y;
    if output_x >= params.destination_width || output_y >= params.destination_height {
        return;
    }

    let source_x0 = output_x * params.source_width / params.destination_width;
    let source_y0 = output_y * params.source_height / params.destination_height;
    let source_x1 = max(
        (output_x + 1u) * params.source_width / params.destination_width,
        source_x0 + 1u,
    );
    let source_y1 = max(
        (output_y + 1u) * params.source_height / params.destination_height,
        source_y0 + 1u,
    );

    var red = 0u;
    var green = 0u;
    var blue = 0u;
    var count = 0u;
    for (var y = source_y0; y < source_y1; y++) {
        for (var x = source_x0; x < source_x1; x++) {
            let pixel = source_pixels[y * params.source_width + x];
            red = red + (pixel & 0xffu);
            green = green + ((pixel >> 8u) & 0xffu);
            blue = blue + ((pixel >> 16u) & 0xffu);
            count = count + 1u;
        }
    }
    destination_pixels[output_y * params.destination_width + output_x] =
        (red / count) |
        ((green / count) << 8u) |
        ((blue / count) << 16u) |
        0xff000000u;
}
