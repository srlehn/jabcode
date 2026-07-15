struct Params {
    width: u32,
    height: u32,
    row_count: u32,
    column_count: u32,
}

@group(0) @binding(0) var<storage, read> pixels: array<u32>;
@group(0) @binding(1) var<storage, read_write> samples: array<u32>;
@group(0) @binding(2) var<storage, read> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row_samples = params.row_count * params.width;
    let column_samples = params.column_count * params.height;
    let sample_count = row_samples + column_samples;
    if id.x >= sample_count {
        return;
    }
    var x: u32;
    var y: u32;
    if id.x < row_samples {
        let row = id.x / params.width;
        x = id.x % params.width;
        y = row * params.height / params.row_count;
    } else {
        let offset = id.x - row_samples;
        let column = offset / params.height;
        x = column * params.width / params.column_count;
        y = offset % params.height;
    }
    let pixel = pixels[y * params.width + x];
    samples[id.x] =
        (pixel & 0xffu) +
        ((pixel >> 8u) & 0xffu) +
        ((pixel >> 16u) & 0xffu);
}
