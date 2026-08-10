// One invocation per sampled pitch line: folds the line's luma samples into
// a sum in the representation the remaining device stages consume. Centering
// divides it by the line length where it already lies; no host normalization
// boundary exists in the resident chain.

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
@group(0) @binding(1) var<storage, read_write> sums: array<f32>;
@group(0) @binding(2) var<storage, read> params: PitchLagParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let line = id.x;
    if line >= params.row_count + params.column_count {
        return;
    }
    var base: u32;
    var length: u32;
    if line < params.row_count {
        base = line * params.width;
        length = params.width;
    } else {
        base = params.row_count * params.width + (line - params.row_count) * params.height;
        length = params.height;
    }
    var sum = 0.0;
    for (var x = 0u; x < length; x++) {
        sum = (sum + (f32(samples[base + x]) / f32(3u)));
    }
    sums[line] = sum;
}
