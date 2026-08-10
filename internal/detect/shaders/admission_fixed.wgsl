// Gates payload correction on the format-fixed modules while the sampled grid
// is still resident. Palette coherence has already been checked on the host;
// this stage answers only the part that needs module pixels.
//
// The finder layers and alignment modules are flattened across the workgroup,
// so the palette search for each checked module runs on a different lane where
// possible. The result stays in the two parameter blocks: acceptance clears
// their pending flags, while rejection marks every LDPC block failed so the
// existing compact correction result carries the answer back.

const GRID_MODULES: u32 = 1u;
const WORKGROUP: u32 = 256u;
const DISTANCE_TO_BORDER: i32 = 4;
const FINDER_CHECKS: u32 = 68u;
const ALIGNMENT_CHECKS_PER_CELL: u32 = 7u;

const ADMISSION_PENDING: u32 = 1u;
const ADMISSION_REJECTED: u32 = 2u;
const LDPC_REJECTED: u32 = 2u;

const PARAM_SIDE_X: u32 = 0u;
const PARAM_SIDE_Y: u32 = 1u;
const PARAM_COLOR_NUMBER: u32 = 3u;
const PARAM_BITS_PER_MODULE: u32 = 4u;
const PARAM_AP_NUM_X: u32 = 6u;
const PARAM_AP_NUM_Y: u32 = 7u;
const PARAM_PALETTE_COPIES: u32 = 10u;
const PARAM_AP_POS_X: u32 = 12u;
const PARAM_AP_POS_Y: u32 = 21u;
const PARAM_PALETTE_THRESHOLDS: u32 = 30u;
const PARAM_PALETTE_EXTREMES: u32 = 42u;
const PARAM_NORMALIZED_PALETTE: u32 = 50u;
const PARAM_PALETTE_BYTES: u32 = 178u;
const PARAM_ADMISSION: u32 = 1714u;

const LDPC_PARAM_BLOCKS: u32 = 4u;
const LDPC_PARAM_ADMISSION: u32 = 13u;

@group(0) @binding(0) var<storage, read_write> params: array<u32>;
@group(0) @binding(1) var<storage, read> grid: array<u32>;
@group(0) @binding(2) var<storage, read_write> ldpc_params: array<u32>;
@group(0) @binding(3) var<storage, read_write> net: array<atomic<u32>>;

var<workgroup> agreements: array<u32, 256>;
var<workgroup> checks: array<u32, 256>;

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

fn classify_absolute(rgb: vec3<u32>, corner: u32, colors: u32, copies: u32) -> u32 {
    let copy = corner % copies;
    let value = vec3<f32>(f32(rgb.x), f32(rgb.y), f32(rgb.z));
    var closest = 3.0 * 255.0 * 255.0 + 1.0;
    var index = 0u;
    let base = PARAM_PALETTE_BYTES + colors * 3u * copy;
    for (var entry = 0u; entry < colors; entry += 1u) {
        let at = base + entry * 3u;
        let delta = vec3<f32>(
            f32(params[at]), f32(params[at + 1u]), f32(params[at + 2u]),
        ) - value;
        let distance = dot(delta, delta);
        if distance < closest {
            closest = distance;
            index = entry;
        }
    }
    return index;
}

fn classify(rgb: vec3<u32>, x: u32, y: u32, side_x: u32, side_y: u32, colors: u32, copies: u32) -> u32 {
    let copy = nearest_palette(x, y, side_x, side_y);
    if colors > 8u {
        return classify_absolute(rgb, copy, colors, copies);
    }
    let value = vec3<f32>(f32(rgb.x), f32(rgb.y), f32(rgb.z));
    let threshold = vec3<f32>(
        param_f32(PARAM_PALETTE_THRESHOLDS + copy * 3u),
        param_f32(PARAM_PALETTE_THRESHOLDS + copy * 3u + 1u),
        param_f32(PARAM_PALETTE_THRESHOLDS + copy * 3u + 2u),
    );
    if value.x < threshold.x && value.y < threshold.y && value.z < threshold.z {
        return 0u;
    }

    let normalized = value / max(max(value.x, value.y), value.z);
    var closest = 255.0 * 255.0 * 3.0;
    var index = 0u;
    let base = PARAM_NORMALIZED_PALETTE + colors * 4u * copy;
    for (var entry = 0u; entry < colors; entry += 1u) {
        let at = base + entry * 4u;
        let delta = vec3<f32>(param_f32(at), param_f32(at + 1u), param_f32(at + 2u)) - normalized;
        let distance = dot(delta, delta);
        if distance < closest {
            closest = distance;
            index = entry;
        }
    }

    if colors == 8u && (index == 0u || index == 7u) {
        let sum = rgb.x + rgb.y + rgb.z;
        let black = params[PARAM_PALETTE_EXTREMES + copy * 2u];
        let white = params[PARAM_PALETTE_EXTREMES + copy * 2u + 1u];
        if sum < (black + white) / 2u {
            index = 0u;
        } else {
            index = 7u;
        }
    }
    return index;
}

fn checked_module(x: i32, y: i32, want: u32) -> vec2<u32> {
    let side_x = i32(params[PARAM_SIDE_X]);
    let side_y = i32(params[PARAM_SIDE_Y]);
    if x < 0 || y < 0 || x >= side_x || y >= side_y {
        return vec2<u32>(0u, 0u);
    }
    let word = grid[GRID_MODULES + u32(y) * u32(side_x) + u32(x)];
    let rgb = vec3<u32>(word & 0xffu, (word >> 8u) & 0xffu, (word >> 16u) & 0xffu);
    let got = classify(
        rgb, u32(x), u32(y), u32(side_x), u32(side_y),
        params[PARAM_COLOR_NUMBER], params[PARAM_PALETTE_COPIES],
    );
    return vec2<u32>(select(0u, 1u, got == want), 1u);
}

fn fixed_color(high: bool, nc: u32) -> u32 {
    if high {
        var values = array<u32, 8>(0u, 2u, 6u, 14u, 30u, 60u, 124u, 252u);
        return values[nc];
    }
    var values = array<u32, 8>(0u, 3u, 3u, 3u, 7u, 15u, 15u, 31u);
    return values[nc];
}

fn finder_color(finder: u32, nc: u32) -> u32 {
    if finder < 2u {
        return 0u;
    }
    return fixed_color(finder == 2u, nc);
}

fn finder_module(finder: u32, i: i32, j: i32, second_arc: bool) -> vec2<i32> {
    let width = i32(params[PARAM_SIDE_X]);
    let height = i32(params[PARAM_SIDE_Y]);
    if !second_arc {
        switch finder {
            case 0u: { return vec2<i32>(DISTANCE_TO_BORDER - j - 1, DISTANCE_TO_BORDER - i - 1); }
            case 1u: { return vec2<i32>(width - DISTANCE_TO_BORDER - j, DISTANCE_TO_BORDER - i - 1); }
            case 2u: { return vec2<i32>(width - DISTANCE_TO_BORDER - j, height - DISTANCE_TO_BORDER + i); }
            default: { return vec2<i32>(DISTANCE_TO_BORDER - j - 1, height - DISTANCE_TO_BORDER + i); }
        }
    }
    switch finder {
        case 0u: { return vec2<i32>(DISTANCE_TO_BORDER + j - 1, DISTANCE_TO_BORDER + i - 1); }
        case 1u: { return vec2<i32>(width - DISTANCE_TO_BORDER + j, DISTANCE_TO_BORDER + i - 1); }
        case 2u: { return vec2<i32>(width - DISTANCE_TO_BORDER + j, height - DISTANCE_TO_BORDER - i); }
        default: { return vec2<i32>(DISTANCE_TO_BORDER + j - 1, height - DISTANCE_TO_BORDER - i); }
    }
}

fn check_finder(index: u32) -> vec2<u32> {
    let finder = index / 17u;
    let within = index % 17u;
    var layer = 0u;
    var boundary = 0u;
    var second_arc = false;
    if within > 0u && within < 7u {
        layer = 1u;
        let at = within - 1u;
        boundary = at / 2u;
        second_arc = (at % 2u) != 0u;
    } else if within >= 7u {
        layer = 2u;
        let at = within - 7u;
        boundary = at / 2u;
        second_arc = (at % 2u) != 0u;
    }

    var i = i32(layer);
    var j = i32(boundary) - i32(layer);
    if boundary < layer {
        i = i32(boundary);
        j = i32(layer);
    }
    let nc = params[PARAM_BITS_PER_MODULE] - 1u;
    var want_finder = finder;
    if (layer % 2u) != 0u {
        want_finder = 3u - finder;
    }
    let want = finder_color(want_finder, nc);
    let position = finder_module(finder, i, j, second_arc);
    return checked_module(position.x, position.y, want);
}

fn check_alignment(index: u32) -> vec2<u32> {
    let cell = index / ALIGNMENT_CHECKS_PER_CELL;
    let module = index % ALIGNMENT_CHECKS_PER_CELL;
    let ap_x = params[PARAM_AP_NUM_X];
    let ap_y = params[PARAM_AP_NUM_Y];
    let x_index = cell / ap_y;
    let y_index = cell % ap_y;
    let corner = (x_index == 0u || x_index == ap_x - 1u) &&
        (y_index == 0u || y_index == ap_y - 1u);
    if corner {
        return vec2<u32>(0u, 0u);
    }

    let x = i32(params[PARAM_AP_POS_X + x_index]) - 1;
    let y = i32(params[PARAM_AP_POS_Y + y_index]) - 1;
    let nc = params[PARAM_BITS_PER_MODULE] - 1u;
    let core = fixed_color(true, nc);
    let periphery = fixed_color(false, nc);
    let left = ((x_index + y_index) % 2u) == 0u;
    let dx = select(1, -1, left);

    switch module {
        case 0u: { return checked_module(x, y, core); }
        case 1u: { return checked_module(x + dx, y - 1, periphery); }
        case 2u: { return checked_module(x, y - 1, periphery); }
        case 3u: { return checked_module(x - 1, y, periphery); }
        case 4u: { return checked_module(x + 1, y, periphery); }
        case 5u: { return checked_module(x, y + 1, periphery); }
        default: { return checked_module(x - dx, y + 1, periphery); }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
    let lane = local.x;
    if params[PARAM_ADMISSION] == ADMISSION_REJECTED {
        for (var block = lane; block < ldpc_params[LDPC_PARAM_BLOCKS]; block += WORKGROUP) {
            atomicStore(&net[block], LDPC_REJECTED);
        }
        return;
    }
    if params[PARAM_ADMISSION] != ADMISSION_PENDING {
        return;
    }

    var result = vec2<u32>(0u, 0u);
    for (var index = lane; index < FINDER_CHECKS; index += WORKGROUP) {
        result += check_finder(index);
    }
    let alignment_checks = params[PARAM_AP_NUM_X] * params[PARAM_AP_NUM_Y] * ALIGNMENT_CHECKS_PER_CELL;
    for (var index = lane; index < alignment_checks; index += WORKGROUP) {
        result += check_alignment(index);
    }
    agreements[lane] = result.x;
    checks[lane] = result.y;
    workgroupBarrier();

    for (var stride = WORKGROUP / 2u; stride > 0u; stride >>= 1u) {
        if lane < stride {
            agreements[lane] += agreements[lane + stride];
            checks[lane] += checks[lane + stride];
        }
        workgroupBarrier();
    }

    if lane == 0u {
        let agree = agreements[0];
        let checked = checks[0];
        let admitted = checked >= 20u && agree * 5u >= checked * 2u;
        if admitted {
            params[PARAM_ADMISSION] = 0u;
            ldpc_params[LDPC_PARAM_ADMISSION] = 0u;
        } else {
            params[PARAM_ADMISSION] = ADMISSION_REJECTED;
            ldpc_params[LDPC_PARAM_ADMISSION] = ADMISSION_REJECTED;
            for (var block = 0u; block < ldpc_params[LDPC_PARAM_BLOCKS]; block += 1u) {
                atomicStore(&net[block], LDPC_REJECTED);
            }
        }
    }
}
