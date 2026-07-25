// One invocation per axis and lag: the biased autocorrelation of the
// centered pitch samples, folded over the axis's lines in line order with
// the CPU estimator's exact accumulation order. Each per-line inner product
// runs over four independent partial sums combined left to right, then a
// scalar tail, then the line's contribution scaled by the host-computed
// reciprocal of the line length. That shape is not incidental: it matches
// acfAccumulate operation for operation, because either this kernel or its
// CPU twin may run depending on whether background compilation finished, and
// detection must not depend on which. Lags past a line's last sample
// contribute an exact zero, matching the CPU estimator skipping them.

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
        var s0 = F64(0u, 0u);
        var s1 = F64(0u, 0u);
        var s2 = F64(0u, 0u);
        var s3 = F64(0u, 0u);
        var x = 0u;
        for (; x + 4u <= count; x += 4u) {
            s0 = sf_add(s0, sf_mul(centered[start + x], centered[start + x + lag]));
            s1 = sf_add(s1, sf_mul(centered[start + x + 1u], centered[start + x + 1u + lag]));
            s2 = sf_add(s2, sf_mul(centered[start + x + 2u], centered[start + x + 2u + lag]));
            s3 = sf_add(s3, sf_mul(centered[start + x + 3u], centered[start + x + 3u + lag]));
        }
        var sum = sf_add(sf_add(sf_add(s0, s1), s2), s3);
        for (; x < count; x++) {
            sum = sf_add(sum, sf_mul(centered[start + x], centered[start + x + lag]));
        }
        total = sf_add(total, sf_mul(sum, inv));
    }
    acf[id.x] = total;
}
