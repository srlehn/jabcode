// One invocation per axis and lag: the biased autocorrelation of the
// centered pitch samples, folded over the axis's lines in line order with
// the CPU estimator's exact accumulation order (sequential per-line inner
// product, then the line's contribution scaled by the host-computed
// reciprocal of the line length). Lags past a line's last sample contribute
// an exact zero, matching the CPU estimator skipping them.

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

@group(0) @binding(0) var<storage, read> centered: array<F64>;
@group(0) @binding(1) var<storage, read_write> acf: array<F64>;
@group(0) @binding(2) var<storage, read> params: PitchLagParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let lags = params.max_lag + 1u;
    if id.x >= 2u * lags {
        return;
    }
    let axis = id.x / lags;
    let lag = id.x % lags;
    var lines: u32;
    var length: u32;
    var base: u32;
    var inv: F64;
    if axis == 0u {
        lines = params.row_count;
        length = params.width;
        base = 0u;
        inv = F64(params.inv_row_hi, params.inv_row_lo);
    } else {
        lines = params.column_count;
        length = params.height;
        base = params.row_count * params.width;
        inv = F64(params.inv_col_hi, params.inv_col_lo);
    }
    var total = F64(0u, 0u);
    for (var line = 0u; line < lines; line++) {
        let start = base + line * length;
        var count = 0u;
        if lag < length {
            count = length - lag;
        }
        var sum = F64(0u, 0u);
        for (var x = 0u; x < count; x++) {
            sum = sf_add(sum, sf_mul(centered[start + x], centered[start + x + lag]));
        }
        total = sf_add(total, sf_mul(sum, inv));
    }
    acf[id.x] = total;
}
