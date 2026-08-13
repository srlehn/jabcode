// Subtracts each pitch line's mean from its converted luma samples,
// producing the centered values the autocorrelation kernel multiplies. The
// line sums stay resident from the reduction and are normalized here.

struct PitchLagParams {
    width: u32,
    height: u32,
}

const PITCH_SAMPLE_LINES: u32 = 32u;

@group(0) @binding(0) var<storage, read> samples: array<u32>;
@group(0) @binding(1) var<storage, read> sums: array<f32>;
@group(0) @binding(2) var<storage, read_write> centered: array<f32>;
@group(0) @binding(3) var<storage, read> params: PitchLagParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row_count = min(PITCH_SAMPLE_LINES, params.height);
    let column_count = min(PITCH_SAMPLE_LINES, params.width);
    let row_samples = row_count * params.width;
    let column_samples = column_count * params.height;
    if id.x >= row_samples + column_samples {
        return;
    }
    var line: u32;
    var length: u32;
    if id.x < row_samples {
        line = id.x / params.width;
        length = params.width;
    } else {
        line = row_count + (id.x - row_samples) / params.height;
        length = params.height;
    }
    let value = (f32(samples[id.x]) / f32(3u));
    centered[id.x] = (value - sums[line] / f32(length));
}
