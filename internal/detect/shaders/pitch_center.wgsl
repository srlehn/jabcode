// Subtracts each pitch line's mean from its converted luma samples,
// producing the centered values the autocorrelation kernel multiplies. The
// line sums stay resident from the reduction and are normalized here.

struct PitchLagParams {
    width: u32,
    height: u32,
    row_count: u32,
    column_count: u32,
    max_lag: u32,
    inv_row_hi: u32,
    inv_row_lo: u32,
    inv_col_hi: u32,
    inv_col_lo: u32,
}

@group(0) @binding(0) var<storage, read> samples: array<u32>;
@group(0) @binding(1) var<storage, read> sums: array<f32>;
@group(0) @binding(2) var<storage, read_write> centered: array<f32>;
@group(0) @binding(3) var<storage, read> params: PitchLagParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row_samples = params.row_count * params.width;
    let column_samples = params.column_count * params.height;
    if id.x >= row_samples + column_samples {
        return;
    }
    var line: u32;
    var length: u32;
    if id.x < row_samples {
        line = id.x / params.width;
        length = params.width;
    } else {
        line = params.row_count + (id.x - row_samples) / params.height;
        length = params.height;
    }
    let value = (f32(samples[id.x]) / f32(3u));
    centered[id.x] = (value - sums[line] / f32(length));
}
