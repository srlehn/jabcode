// Reads Part I of the primary metadata off the resident module grid and emits
// its six bits for the hard corrector, so the grid never has to come back for
// the host to interpret it.
//
// Part I is four modules and the walk between them is a fixed sequence of
// reflections about the symbol's axes, so this is serial by construction and one
// lane runs the whole thing. Parallelism is not why it moves: it moves because
// it was the last stage that made the host read module pixels, and a stage that
// cannot answer where the data already sits has not been ported.
//
// The classification is deliberately not the palette classifier. Part I is read
// before the palette exists, so it decides from absolute channel values and from
// the ratios between them, and it recognizes only the three colours the encoder
// is allowed to place here.

const STATUS_OK: u32 = 0u;
// Part I did not resolve to a legal colour pair even after the reference retry,
// which is how a genuinely default-mode symbol presents. The host's answer is to
// load default metadata and skip Part II, so the status says which happened
// rather than failing the read.
const STATUS_DEFAULT: u32 = 1u;

const NC_BLACK: u32 = 0u;
const NC_CYAN: u32 = 3u;
const NC_YELLOW: u32 = 6u;
const NC_INVALID: u32 = 8u;

const THS_BLACK: f32 = 80.0;
const THS_STD: f32 = 0.08;

// The sampler writes its reject flag in the first word of the grid buffer.
const GRID_MODULES: u32 = 1u;

const PARAM_SIDE_X: u32 = 0u;
const PARAM_SIDE_Y: u32 = 1u;

const RECORD_STATUS: u32 = 0u;
const RECORD_MODULES: u32 = 1u;
// The walk is serial, so each stage hands the next its position rather than
// replaying the steps before it.
const RECORD_WALK_X: u32 = 2u;
const RECORD_WALK_Y: u32 = 3u;

@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> grid: array<u32>;
@group(0) @binding(2) var<storage, read_write> bits: array<u32>;
@group(0) @binding(3) var<storage, read_write> record: array<u32>;

fn module_rgb(x: i32, y: i32) -> vec3<f32> {
    let side_x = i32(params[PARAM_SIDE_X]);
    let word = grid[GRID_MODULES + u32(y * side_x + x)];
    return vec3<f32>(
        f32(word & 0xffu),
        f32((word >> 8u) & 0xffu),
        f32((word >> 16u) & 0xffu),
    );
}

// decode_module_nc maps a Part I module to one of the eight canonical colour
// indices: black by absolute darkness, white by how little its channels vary,
// and otherwise a bit per channel with the middle channel decided by which of
// its two ratios is the larger.
//
// The three comparisons that order the channels are the host's own, in its
// order. A tie decides which channel carries the middle bit, so which index
// comes out of an equal pair is part of the contract rather than an
// implementation detail.
fn decode_module_nc(rgb: vec3<f32>) -> u32 {
    if rgb.x < THS_BLACK && rgb.y < THS_BLACK && rgb.z < THS_BLACK {
        return 0u;
    }
    var channel = array<f32, 3>(rgb.x, rgb.y, rgb.z);
    let mean = (rgb.x + rgb.y + rgb.z) / 3.0;
    let spread = rgb - vec3<f32>(mean);
    let variance = dot(spread, spread) / 3.0;

    var lo = 0u;
    var mid = 1u;
    var hi = 2u;
    if channel[lo] > channel[hi] {
        let swap = lo;
        lo = hi;
        hi = swap;
    }
    if channel[lo] > channel[mid] {
        let swap = lo;
        lo = mid;
        mid = swap;
    }
    if channel[mid] > channel[hi] {
        let swap = mid;
        mid = hi;
        hi = swap;
    }

    if sqrt(variance) / channel[hi] <= THS_STD {
        return 7u;
    }
    var value = array<u32, 3>(0u, 0u, 0u);
    value[hi] = 1u;
    // Both ratios can divide by zero, and the host's arithmetic produces the
    // same infinities and NaN, so the comparison agrees on the degenerate
    // modules rather than needing a guard the host does not have.
    if channel[mid] / channel[lo] > channel[hi] / channel[mid] {
        value[mid] = 1u;
    }
    return (value[0] << 2u) + (value[1] << 1u) + value[2];
}

// nc_index maps one of the three colours the encoder may place in Part I to its
// position in the pair encoding, or 3 for anything else.
fn nc_index(color: u32) -> u32 {
    switch color {
        case NC_BLACK: { return 0u; }
        case NC_CYAN: { return 1u; }
        case NC_YELLOW: { return 2u; }
        default: { return 3u; }
    }
}

// nc_pair_value is the table lookup written as arithmetic: the eight encoded
// values are the pairs of three colours in order, minus the ninth pair, which
// has no encoding and reports invalid the same way an unrecognized colour does.
fn nc_pair_value(first: u32, second: u32) -> u32 {
    let a = nc_index(first);
    let b = nc_index(second);
    if a > 2u || b > 2u {
        return NC_INVALID;
    }
    let value = a * 3u + b;
    if value > 7u {
        return NC_INVALID;
    }
    return value;
}

// reference_colors builds eight colour references from the symbol's own finder
// cores. The plain classification decides from absolute channel values, which a
// display cast defeats - a screen's black is bright enough in blue to fail the
// black test - so an invalid Part I is retried against references the same
// capture provides. The gains must all be positive, or the cores are not
// carrying the colours this assumes and the retry is not attempted.
fn reference_colors(refs: ptr<function, array<vec3<f32>, 8>>) -> bool {
    let side_x = i32(params[PARAM_SIDE_X]);
    let side_y = i32(params[PARAM_SIDE_Y]);
    let black = (module_rgb(3, 3) + module_rgb(side_x - 4, 3)) * 0.5;
    let yellow = module_rgb(side_x - 4, side_y - 4);
    let cyan = module_rgb(3, side_y - 4);
    let gain = vec3<f32>(
        yellow.x - black.x,
        ((yellow.y - black.y) + (cyan.y - black.y)) * 0.5,
        cyan.z - black.z,
    );
    if gain.x <= 0.0 || gain.y <= 0.0 || gain.z <= 0.0 {
        return false;
    }
    for (var c = 0u; c < 8u; c += 1u) {
        (*refs)[c] = black + vec3<f32>(
            f32((c >> 2u) & 1u) * gain.x,
            f32((c >> 1u) & 1u) * gain.y,
            f32(c & 1u) * gain.z,
        );
    }
    return true;
}

fn nearest_reference(rgb: vec3<f32>, refs: ptr<function, array<vec3<f32>, 8>>) -> u32 {
    var best = 0u;
    var closest = -1.0;
    for (var c = 0u; c < 8u; c += 1u) {
        let delta = rgb - (*refs)[c];
        let distance = dot(delta, delta);
        if closest < 0.0 || distance < closest {
            closest = distance;
            best = c;
        }
    }
    return best;
}

// advance_metadata_module steps to the next module of the primary metadata walk.
// It is the same sequence payload_map.wgsl replays to reserve these modules, and
// the two must stay identical: one decides what Part I reads and the other
// decides which modules the payload treats as data.
fn advance_metadata_module(position: ptr<function, vec2<i32>>, count: i32) {
    let width = i32(params[PARAM_SIDE_X]);
    let height = i32(params[PARAM_SIDE_Y]);
    let phase = count % 4;
    if phase == 0 || phase == 2 {
        (*position).y = height - 1 - (*position).y;
    }
    if phase == 1 || phase == 3 {
        (*position).x = width - 1 - (*position).x;
    }
    if phase == 0 {
        if count <= 20 || (count >= 44 && count <= 68) ||
            (count >= 96 && count <= 124) || (count >= 156 && count <= 172) {
            (*position).y += 1;
        } else if (count > 20 && count < 44) || (count > 68 && count < 96) ||
            (count > 124 && count < 156) {
            (*position).x -= 1;
        }
    }
    if count == 44 || count == 96 || count == 156 {
        let swap = (*position).x;
        (*position).x = (*position).y;
        (*position).y = swap;
    }
}

// Where the primary metadata strip begins, which is also where the default
// ladder restarts its palette read.
const METADATA_START = vec2<i32>(6, 1);

@compute @workgroup_size(1)
fn main() {
    // Constructed rather than assigned straight from METADATA_START: this
    // variable is mutated through a pointer below, and initializing it from a
    // module-scope composite const made the walk read the wrong modules on the
    // target driver.
    var position = vec2<i32>(METADATA_START.x, METADATA_START.y);
    var colors = array<u32, 4>();
    var samples = array<vec3<f32>, 4>();
    for (var taken = 0u; taken < 4u; taken += 1u) {
        let rgb = module_rgb(position.x, position.y);
        samples[taken] = rgb;
        colors[taken] = decode_module_nc(rgb);
        advance_metadata_module(&position, i32(taken) + 1);
    }

    var first = nc_pair_value(colors[0], colors[1]);
    var second = nc_pair_value(colors[2], colors[3]);
    if first == NC_INVALID || second == NC_INVALID {
        var refs = array<vec3<f32>, 8>();
        if reference_colors(&refs) {
            for (var i = 0u; i < 4u; i += 1u) {
                colors[i] = nearest_reference(samples[i], &refs);
            }
            first = nc_pair_value(colors[0], colors[1]);
            second = nc_pair_value(colors[2], colors[3]);
        }
    }
    if first == NC_INVALID || second == NC_INVALID {
        // The symbol carries no explicit colour mode, so the walk starts over
        // with default metadata: the palette occupies the modules Part I was
        // read from, and the host's ladder resets its position and count for
        // exactly that reason.
        record[RECORD_MODULES] = 0u;
        record[RECORD_WALK_X] = u32(METADATA_START.x);
        record[RECORD_WALK_Y] = u32(METADATA_START.y);
        record[RECORD_STATUS] = STATUS_DEFAULT;
        return;
    }
    record[RECORD_MODULES] = 4u;
    record[RECORD_WALK_X] = u32(position.x);
    record[RECORD_WALK_Y] = u32(position.y);

    for (var i = 0u; i < 3u; i += 1u) {
        bits[i] = (first >> (2u - i)) & 1u;
        bits[3u + i] = (second >> (2u - i)) & 1u;
    }
    record[RECORD_STATUS] = STATUS_OK;
}
