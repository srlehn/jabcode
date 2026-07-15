struct Params {
    values: array<u32, 18>,
}

@group(0) @binding(0) var<storage, read> pixels: array<u32>;
@group(0) @binding(1) var<storage, read_write> partials: array<u32>;
@group(0) @binding(2) var<storage, read> params: Params;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let rectangle = 2u + group.x * 4u;
    let start_x = params.values[rectangle];
    let start_y = params.values[rectangle + 1u];
    let end_x = params.values[rectangle + 2u];
    let end_y = params.values[rectangle + 3u];
    let rectangle_width = end_x - start_x;
    let rectangle_height = end_y - start_y;
    let area = rectangle_width * rectangle_height;
    var sum = vec3<u32>(0u);
    var count = 0u;
    for (var offset = lane; offset < area; offset += 64u) {
        let x = start_x + offset % rectangle_width;
        let y = start_y + offset / rectangle_width;
        let pixel = pixels[y * params.values[0] + x];
        sum += vec3<u32>(
            pixel & 0xffu,
            (pixel >> 8u) & 0xffu,
            (pixel >> 16u) & 0xffu,
        );
        count += 1u;
    }
    let output = (group.x * 64u + lane) * 4u;
    partials[output] = sum.x;
    partials[output + 1u] = sum.y;
    partials[output + 2u] = sum.z;
    partials[output + 3u] = count;
}
