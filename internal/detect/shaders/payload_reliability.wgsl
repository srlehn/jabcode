// Builds max-log per-bit reliability only after hard LDPC reports a failed
// sub-block. One workgroup owns one module and distributes its palette entries
// across lanes, replacing the host's serial colors-by-bits loop with parallel
// minimum reductions. The sampled grid and deinterleaving table stay resident.

const NOT_DATA: u32 = 0xffffffffu;
const GRID_MODULES: u32 = 1u;
const WORKGROUP: u32 = 64u;
const MAX_BITS: u32 = 8u;

const RELIABILITY_SCALE: f32 = 32767.0;
const NORMALIZED_DISTANCE_MAX: f32 = 3.0;
const ABSOLUTE_DISTANCE_MAX: f32 = 3.0 * 255.0 * 255.0;

const PARAM_SIDE_X: u32 = 0u;
const PARAM_SIDE_Y: u32 = 1u;
const PARAM_COLOR_NUMBER: u32 = 3u;
const PARAM_BITS_PER_MODULE: u32 = 4u;
const PARAM_GROSS_BITS: u32 = 8u;
const PARAM_PALETTE_COPIES: u32 = 10u;
const PARAM_NORMALIZED_PALETTE: u32 = 50u;
const PARAM_PALETTE_BYTES: u32 = 178u;

@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> grid: array<u32>;
@group(0) @binding(2) var<storage, read> map: array<u32>;
@group(0) @binding(3) var<storage, read> permutation: array<u32>;
@group(0) @binding(4) var<storage, read_write> reliability: array<u32>;

var<workgroup> minimum_zero: array<f32, 512>;
var<workgroup> minimum_one: array<f32, 512>;

fn param_f32(index: u32) -> f32 {
    return bitcast<f32>(params[index]);
}

fn nearest_palette(x: u32, y: u32, side_x: u32, side_y: u32) -> u32 {
    var px = array<i32, 4>(6, i32(side_x) - 7, i32(side_x) - 7, 6);
    var py = array<i32, 4>(3, 3, i32(side_y) - 4, i32(side_y) - 4);
    var best = sqrt(f32(side_x) * f32(side_x) + f32(side_y) * f32(side_y));
    var chosen = 0u;
    for (var copy = 0u; copy < 4u; copy += 1u) {
        let dx = f32(i32(x) - px[copy]);
        let dy = f32(i32(y) - py[copy]);
        let distance = sqrt(dx * dx + dy * dy);
        if distance < best {
            best = distance;
            chosen = copy;
        }
    }
    return chosen;
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let module = group.x;
    let side_x = params[PARAM_SIDE_X];
    let side_y = params[PARAM_SIDE_Y];
    if module >= side_x * side_y || map[module] == NOT_DATA {
        return;
    }

    let x = module / side_y;
    let y = module % side_y;
    let word = grid[GRID_MODULES + y * side_x + x];
    let value = vec3<f32>(
        f32(word & 0xffu),
        f32((word >> 8u) & 0xffu),
        f32((word >> 16u) & 0xffu),
    );
    let colors = params[PARAM_COLOR_NUMBER];
    let bits = params[PARAM_BITS_PER_MODULE];
    let copies = params[PARAM_PALETTE_COPIES];
    let corner = nearest_palette(x, y, side_x, side_y);
    let absolute = colors > 8u;
    let metric_max = select(NORMALIZED_DISTANCE_MAX, ABSOLUTE_DISTANCE_MAX, absolute);
    let normalized = value / max(1.0, max(max(value.x, value.y), value.z));

    var lane_zero: array<f32, 8>;
    var lane_one: array<f32, 8>;
    for (var bit = 0u; bit < MAX_BITS; bit += 1u) {
        lane_zero[bit] = metric_max + 1.0;
        lane_one[bit] = metric_max + 1.0;
    }
    for (var entry = local.x; entry < colors; entry += WORKGROUP) {
        var distance = 0.0;
        if absolute {
            let copy = corner % copies;
            let at = PARAM_PALETTE_BYTES + colors * 3u * copy + entry * 3u;
            let delta = vec3<f32>(
                f32(params[at]), f32(params[at + 1u]), f32(params[at + 2u]),
            ) - value;
            distance = dot(delta, delta);
        } else {
            let at = PARAM_NORMALIZED_PALETTE + colors * 4u * corner + entry * 4u;
            let delta = vec3<f32>(
                param_f32(at), param_f32(at + 1u), param_f32(at + 2u),
            ) - normalized;
            distance = dot(delta, delta);
        }
        for (var bit = 0u; bit < bits; bit += 1u) {
            let shift = bits - 1u - bit;
            if ((entry >> shift) & 1u) == 0u {
                lane_zero[bit] = min(lane_zero[bit], distance);
            } else {
                lane_one[bit] = min(lane_one[bit], distance);
            }
        }
    }
    for (var bit = 0u; bit < bits; bit += 1u) {
        let at = bit * WORKGROUP + local.x;
        minimum_zero[at] = lane_zero[bit];
        minimum_one[at] = lane_one[bit];
    }
    workgroupBarrier();

    for (var stride = WORKGROUP / 2u; stride > 0u; stride >>= 1u) {
        if local.x < stride {
            for (var bit = 0u; bit < bits; bit += 1u) {
                let at = bit * WORKGROUP + local.x;
                minimum_zero[at] = min(minimum_zero[at], minimum_zero[at + stride]);
                minimum_one[at] = min(minimum_one[at], minimum_one[at + stride]);
            }
        }
        workgroupBarrier();
    }

    if local.x == 0u {
        let position = map[module];
        let gross = params[PARAM_GROSS_BITS];
        for (var bit = 0u; bit < bits; bit += 1u) {
            let source = position * bits + bit;
            if source < gross {
                let gap = abs(minimum_one[bit * WORKGROUP] - minimum_zero[bit * WORKGROUP]);
                reliability[permutation[source]] = u32(round(clamp(
                    gap * RELIABILITY_SCALE / metric_max,
                    0.0,
                    RELIABILITY_SCALE,
                )));
            }
        }
    }
}
