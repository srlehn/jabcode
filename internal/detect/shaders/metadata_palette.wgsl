// Reads the symbol's embedded colour palette off the resident module grid and
// derives everything the classifier needs from it: the normalized entries and
// the per-copy black thresholds.
//
// It runs after Part I's correction and reads the colour mode straight out of
// the corrector's output, so the host never learns it and never has to say how
// many palette modules there are. Four copies are embedded, one per corner, and
// a module is classified later against whichever copy is nearest, which is what
// corrects a capture's local colour cast where the module actually sits.
//
// Continuing the walk rather than restarting it is deliberate: the palette
// modules follow Part I's on the same serial path, and Part I leaves its
// position in the record for exactly this reason.

const STATUS_OK: u32 = 0u;
const STATUS_DEFAULT: u32 = 1u;
// The colour mode is one the device chain does not classify, or the walk left
// the symbol. Both are declines rather than failed reads: the host answers them
// from its own walk.
const STATUS_UNSUPPORTED: u32 = 2u;
const STATUS_FAILED: u32 = 3u;

const PALETTE_COPIES: u32 = 4u;
const PALETTE_FINDER_COLORS: u32 = 2u;
const DISTANCE_TO_BORDER: i32 = 4;

const GRID_MODULES: u32 = 1u;

const PARAM_SIDE_X: u32 = 0u;
const PARAM_SIDE_Y: u32 = 1u;
// Which palette entry copy c carries at slot i, already reduced modulo the
// colour count and already resolved for the wire variant, because that choice
// is the host's and does not belong in a kernel.
const PARAM_PLACEMENT_4: u32 = 8u;
const PARAM_PLACEMENT_8: u32 = 24u;

const RECORD_STATUS: u32 = 0u;
const RECORD_MODULES: u32 = 1u;
const RECORD_WALK_X: u32 = 2u;
const RECORD_WALK_Y: u32 = 3u;
const RECORD_NC: u32 = 4u;
const RECORD_COLORS: u32 = 5u;
// The corrector's output buffer is reused by Part II, so Part I's parity
// verdict is copied here while it is still there.
const RECORD_PART1_SYNDROME: u32 = 12u;
const RECORD_PALETTE: u32 = 16u;
const RECORD_NORMALIZED: u32 = 112u;
const RECORD_THRESHOLDS: u32 = 240u;

@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> grid: array<u32>;
@group(0) @binding(2) var<storage, read> net: array<u32>;
@group(0) @binding(3) var<storage, read_write> record: array<u32>;

fn module_rgb(x: i32, y: i32) -> vec3<u32> {
    let side_x = i32(params[PARAM_SIDE_X]);
    let word = grid[GRID_MODULES + u32(y * side_x + x)];
    return vec3<u32>(word & 0xffu, (word >> 8u) & 0xffu, (word >> 16u) & 0xffu);
}

fn placement(copy: u32, slot: u32, colors: u32) -> u32 {
    if colors == 4u {
        return params[PARAM_PLACEMENT_4 + copy * 4u + slot];
    }
    return params[PARAM_PLACEMENT_8 + copy * 8u + slot];
}

fn write_palette(copy: u32, entry: u32, colors: u32, rgb: vec3<u32>) {
    let at = RECORD_PALETTE + (copy * colors + entry) * 3u;
    record[at] = rgb.x;
    record[at + 1u] = rgb.y;
    record[at + 2u] = rgb.z;
}

fn palette_entry(copy: u32, entry: u32, colors: u32) -> vec3<u32> {
    let at = RECORD_PALETTE + (copy * colors + entry) * 3u;
    return vec3<u32>(record[at], record[at + 1u], record[at + 2u]);
}

fn put_f32(at: u32, value: f32) {
    record[at] = bitcast<u32>(value);
}

// finder_palette_positions returns the two modules of one corner's finder
// pattern that carry palette colours 0 and 1. The 4- and 8-colour modes read
// them from the pattern rather than from the metadata strip; the higher modes
// embed every colour instead, and this kernel declines those.
fn finder_palette_positions(copy: u32) -> array<vec2<i32>, 2> {
    let width = i32(params[PARAM_SIDE_X]);
    let height = i32(params[PARAM_SIDE_Y]);
    let near = DISTANCE_TO_BORDER - 1;
    switch copy {
        case 0u: {
            return array<vec2<i32>, 2>(vec2<i32>(near, near), vec2<i32>(near + 1, near));
        }
        case 1u: {
            let x = width - DISTANCE_TO_BORDER;
            return array<vec2<i32>, 2>(vec2<i32>(x, near), vec2<i32>(x - 1, near));
        }
        case 2u: {
            let x = width - DISTANCE_TO_BORDER;
            let y = height - DISTANCE_TO_BORDER;
            return array<vec2<i32>, 2>(vec2<i32>(x, y), vec2<i32>(x - 1, y));
        }
        default: {
            let y = height - DISTANCE_TO_BORDER;
            return array<vec2<i32>, 2>(vec2<i32>(near, y), vec2<i32>(near + 1, y));
        }
    }
}

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

// normalize_palette precomputes each entry's unit-max direction and its mean
// intensity, which is what the classifier compares a module against. An exact
// black entry normalizes to the zero vector rather than dividing by zero: a NaN
// entry would never win a distance comparison, which drops black out of every
// ranking and leaves it reachable only through the threshold shortcut below.
fn normalize_palette(colors: u32) {
    for (var i = 0u; i < colors * PALETTE_COPIES; i += 1u) {
        let at = RECORD_PALETTE + i * 3u;
        let rgb = vec3<u32>(record[at], record[at + 1u], record[at + 2u]);
        let out = RECORD_NORMALIZED + i * 4u;
        let peak = max(max(rgb.x, rgb.y), rgb.z);
        if peak == 0u {
            put_f32(out, 0.0);
            put_f32(out + 1u, 0.0);
            put_f32(out + 2u, 0.0);
            put_f32(out + 3u, 0.0);
            continue;
        }
        let scale = f32(peak);
        put_f32(out, f32(rgb.x) / scale);
        put_f32(out + 1u, f32(rgb.y) / scale);
        put_f32(out + 2u, f32(rgb.z) / scale);
        put_f32(out + 3u, (f32(rgb.x + rgb.y + rgb.z) / 3.0) / 255.0);
    }
}

// palette_thresholds puts each channel's black threshold midway between the
// darkest entry that is bright in that channel and the brightest that is dark
// in it, per copy. Which entries those are is a property of the palette's fixed
// layout, so the two colour modes name them outright.
fn palette_thresholds(colors: u32) {
    for (var copy = 0u; copy < PALETTE_COPIES; copy += 1u) {
        var low = vec3<u32>(0u);
        var high = vec3<u32>(255u);
        if colors == 4u {
            let c0 = palette_entry(copy, 0u, colors);
            let c1 = palette_entry(copy, 1u, colors);
            let c2 = palette_entry(copy, 2u, colors);
            let c3 = palette_entry(copy, 3u, colors);
            low = vec3<u32>(max(c0.x, c1.x), max(c0.y, c2.y), max(c2.z, c3.z));
            high = vec3<u32>(min(c2.x, c3.x), min(c1.y, c3.y), min(c0.z, c1.z));
        } else {
            let c0 = palette_entry(copy, 0u, colors);
            let c1 = palette_entry(copy, 1u, colors);
            let c2 = palette_entry(copy, 2u, colors);
            let c3 = palette_entry(copy, 3u, colors);
            let c4 = palette_entry(copy, 4u, colors);
            let c5 = palette_entry(copy, 5u, colors);
            let c6 = palette_entry(copy, 6u, colors);
            let c7 = palette_entry(copy, 7u, colors);
            low = vec3<u32>(
                max(max(c0.x, c1.x), max(c2.x, c3.x)),
                max(max(c0.y, c1.y), max(c4.y, c5.y)),
                max(max(c0.z, c2.z), max(c4.z, c6.z)),
            );
            high = vec3<u32>(
                min(min(c4.x, c5.x), min(c6.x, c7.x)),
                min(min(c2.y, c3.y), min(c6.y, c7.y)),
                min(min(c1.z, c3.z), min(c5.z, c7.z)),
            );
        }
        let out = RECORD_THRESHOLDS + copy * 3u;
        put_f32(out, f32(low.x + high.x) / 2.0);
        put_f32(out + 1u, f32(low.y + high.y) / 2.0);
        put_f32(out + 2u, f32(low.z + high.z) / 2.0);
    }
}

@compute @workgroup_size(1)
fn main() {
    if record[RECORD_STATUS] != STATUS_OK {
        return;
    }
    // The corrector writes one status word per block ahead of the message, and
    // Part I is a single block, so its three bits start at index one.
    let nc = (net[1] << 2u) + (net[2] << 1u) + net[3];
    let colors = 1u << (nc + 1u);
    record[RECORD_NC] = nc;
    record[RECORD_COLORS] = colors;
    record[RECORD_PART1_SYNDROME] = net[0];
    if colors != 4u && colors != 8u {
        record[RECORD_STATUS] = STATUS_UNSUPPORTED;
        return;
    }

    for (var copy = 0u; copy < PALETTE_COPIES; copy += 1u) {
        let positions = finder_palette_positions(copy);
        write_palette(copy, placement(copy, 0u, colors), colors,
            module_rgb(positions[0].x, positions[0].y));
        write_palette(copy, placement(copy, 1u, colors), colors,
            module_rgb(positions[1].x, positions[1].y));
    }

    let width = i32(params[PARAM_SIDE_X]);
    let height = i32(params[PARAM_SIDE_Y]);
    var position = vec2<i32>(i32(record[RECORD_WALK_X]), i32(record[RECORD_WALK_Y]));
    var taken = record[RECORD_MODULES];
    for (var slot = PALETTE_FINDER_COLORS; slot < colors; slot += 1u) {
        for (var copy = 0u; copy < PALETTE_COPIES; copy += 1u) {
            // A matrix too small for the declared colour mode walks the strip
            // off the symbol; that is a decline, not an out-of-range read.
            if position.x < 0 || position.y < 0 || position.x >= width || position.y >= height {
                record[RECORD_STATUS] = STATUS_FAILED;
                return;
            }
            write_palette(copy, placement(copy, slot, colors), colors,
                module_rgb(position.x, position.y));
            taken += 1u;
            advance_metadata_module(&position, i32(taken));
        }
    }

    normalize_palette(colors);
    palette_thresholds(colors);
    record[RECORD_MODULES] = taken;
    record[RECORD_WALK_X] = u32(position.x);
    record[RECORD_WALK_Y] = u32(position.y);
}
