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
}

const PITCH_SAMPLE_LINES: u32 = 32u;

@group(0) @binding(0) var<storage, read> centered: array<f32>;
@group(0) @binding(1) var<storage, read_write> acf: array<f32>;
@group(0) @binding(2) var<storage, read> params: PitchLagParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row_count = min(PITCH_SAMPLE_LINES, params.height);
    let column_count = min(PITCH_SAMPLE_LINES, params.width);
    let max_lag = max(2u, min(params.width, params.height) / 8u);
    let lags = max_lag + 1u;
    if id.x >= 2u * lags {
        return;
    }
    let axis = id.x / lags;
    let lag = id.x % lags;
    var lines: u32;
    var length: u32;
    var base: u32;
    var inv: f32;
    if axis == 0u {
        lines = row_count;
        length = params.width;
        base = 0u;
        inv = 1.0 / f32(params.width);
    } else {
        lines = column_count;
        length = params.height;
        base = row_count * params.width;
        inv = 1.0 / f32(params.height);
    }
    var total = 0.0;
    for (var line = 0u; line < lines; line++) {
        let start = base + line * length;
        var count = 0u;
        if lag < length {
            count = length - lag;
        }
        var s0 = 0.0;
        var s1 = 0.0;
        var s2 = 0.0;
        var s3 = 0.0;
        var x = 0u;
        for (; x + 4u <= count; x += 4u) {
            s0 = (s0 + (centered[start + x] * centered[start + x + lag]));
            s1 = (s1 + (centered[start + x + 1u] * centered[start + x + 1u + lag]));
            s2 = (s2 + (centered[start + x + 2u] * centered[start + x + 2u + lag]));
            s3 = (s3 + (centered[start + x + 3u] * centered[start + x + 3u + lag]));
        }
        var sum = (((s0 + s1) + s2) + s3);
        for (; x < count; x++) {
            sum = (sum + (centered[start + x] * centered[start + x + lag]));
        }
        total = (total + (sum * inv));
    }
    acf[id.x] = total;
}
