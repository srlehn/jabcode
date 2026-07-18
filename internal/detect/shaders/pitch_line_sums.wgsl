// One invocation per sampled pitch line: folds the line's luma samples into
// an exact float64 sum with the same element order and rounding as the CPU
// pitch estimator (each sample converted to float64 and divided by three
// before the fold). The host divides the sums by the line lengths itself:
// correctly rounded division by an arbitrary line length needs more
// quotient bits than mant_div_small carries, and the few native divisions
// cost nothing there.

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
@group(0) @binding(1) var<storage, read_write> sums: array<F64>;
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
    var sum = F64(0u, 0u);
    for (var x = 0u; x < length; x++) {
        sum = sf_add(sum, sf_div_small(sf_from_u32(samples[base + x]), 3u));
    }
    sums[line] = sum;
}
