// Reads Part II of the primary metadata off the resident module grid and emits
// its bits for the hard corrector.
//
// Unlike Part I this is the ordinary palette classification, because by now the
// palette exists: the stage before this one read it, normalized it and derived
// its thresholds, and all of that is in the record where this stage finds it.
// Part II is therefore the first metadata read that costs the same work a data
// module costs, which is why it could not have run any earlier.
//
// A module carries several bits and the part does not end on a module boundary,
// so the last module's low bits are dropped. That is the host's own truncation,
// not an approximation of it.

const STATUS_OK: u32 = 0u;

// How many spatial corners a module is ranked against. It is four whatever the
// mode, because that is what the host ranks; a mode embedding fewer palettes
// folds the winning corner onto one it has.
const PALETTE_CORNERS: u32 = 4u;
const PART2_LENGTH: u32 = 38u;

const GRID_MODULES: u32 = 1u;

const PARAM_SIDE_X: u32 = 0u;
const PARAM_SIDE_Y: u32 = 1u;

// The per-mode description, as metadata_palette.wgsl writes and reads it.
const PARAM_MODE: u32 = 8u;
const MODE_WORDS: u32 = 5u;
const MODE_COPIES: u32 = 0u;

const RECORD_STATUS: u32 = 0u;
const RECORD_MODULES: u32 = 1u;
const RECORD_WALK_X: u32 = 2u;
const RECORD_WALK_Y: u32 = 3u;
const RECORD_NC: u32 = 4u;
const RECORD_COLORS: u32 = 5u;
const RECORD_PALETTE: u32 = 16u;
const RECORD_NORMALIZED: u32 = 400u;
const RECORD_THRESHOLDS: u32 = 528u;

@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> grid: array<u32>;
@group(0) @binding(2) var<storage, read_write> bits: array<u32>;
@group(0) @binding(3) var<storage, read_write> record: array<u32>;

fn mode_word(nc: u32, field: u32) -> u32 {
    return params[PARAM_MODE + nc * MODE_WORDS + field];
}

fn record_f32(at: u32) -> f32 {
    return bitcast<f32>(record[at]);
}

fn module_rgb(x: i32, y: i32) -> vec3<u32> {
    let side_x = i32(params[PARAM_SIDE_X]);
    let word = grid[GRID_MODULES + u32(y * side_x + x)];
    return vec3<u32>(word & 0xffu, (word >> 8u) & 0xffu, (word >> 16u) & 0xffu);
}

// nearest_palette picks the embedded palette copy closest to a module, which is
// what corrects a capture's local colour cast where the module actually sits.
fn nearest_palette(x: u32, y: u32) -> u32 {
    let side_x = i32(params[PARAM_SIDE_X]);
    let side_y = i32(params[PARAM_SIDE_Y]);
    var px = array<i32, 4>(6, side_x - 7, side_x - 7, 6);
    var py = array<i32, 4>(3, 3, side_y - 4, side_y - 4);
    var best = sqrt(f32(side_x) * f32(side_x) + f32(side_y) * f32(side_y));
    var chosen = 0u;
    for (var copy = 0u; copy < PALETTE_CORNERS; copy += 1u) {
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

fn palette_sum(copy: u32, entry: u32, colors: u32) -> u32 {
    let at = RECORD_PALETTE + (copy * colors + entry) * 3u;
    return record[at] + record[at + 1u] + record[at + 2u];
}

// classify_absolute ranks raw palette distance, which is what the host does
// above eight colours: those modes normalize nothing and have no black
// threshold, so the whole shortcut-and-direction machinery below does not apply
// to them.
//
// The copy index is folded rather than ranked: nearest_palette ranks the four
// spatial corners whatever the mode, while a high-colour symbol embeds only two
// palettes, and the host folds the corner onto a copy with a remainder. Ranking
// only the embedded corners instead would pick a different copy from the host
// across a whole quadrant of the symbol.
fn classify_absolute(rgb: vec3<u32>, corner: u32, colors: u32, copies: u32) -> u32 {
    let copy = corner % copies;
    let value = vec3<f32>(f32(rgb.x), f32(rgb.y), f32(rgb.z));
    var closest = 3.0 * 255.0 * 255.0 + 1.0;
    var index = 0u;
    for (var entry = 0u; entry < colors; entry += 1u) {
        let at = RECORD_PALETTE + (copy * colors + entry) * 3u;
        let delta = vec3<f32>(
            f32(record[at]), f32(record[at + 1u]), f32(record[at + 2u]),
        ) - value;
        let distance = dot(delta, delta);
        if distance < closest {
            closest = distance;
            index = entry;
        }
    }
    return index;
}

// classify maps a module's sampled colour to a palette index: black first, then
// the nearest normalized palette entry, then the black/white tie-break the
// eight-colour palette needs because its two achromatic entries normalize alike.
fn classify(rgb: vec3<u32>, x: u32, y: u32, colors: u32, copies: u32) -> u32 {
    let copy = nearest_palette(x, y);
    if colors > 8u {
        return classify_absolute(rgb, copy, colors, copies);
    }
    let value = vec3<f32>(f32(rgb.x), f32(rgb.y), f32(rgb.z));
    let threshold = vec3<f32>(
        record_f32(RECORD_THRESHOLDS + copy * 3u),
        record_f32(RECORD_THRESHOLDS + copy * 3u + 1u),
        record_f32(RECORD_THRESHOLDS + copy * 3u + 2u),
    );
    if value.x < threshold.x && value.y < threshold.y && value.z < threshold.z {
        return 0u;
    }

    let normalized = value / max(max(value.x, value.y), value.z);
    var closest = 255.0 * 255.0 * 3.0;
    var index = 0u;
    let base = RECORD_NORMALIZED + colors * 4u * copy;
    for (var entry = 0u; entry < colors; entry += 1u) {
        let at = base + entry * 4u;
        let delta = vec3<f32>(
            record_f32(at), record_f32(at + 1u), record_f32(at + 2u),
        ) - normalized;
        let distance = dot(delta, delta);
        if distance < closest {
            closest = distance;
            index = entry;
        }
    }

    if colors == 8u && (index == 0u || index == 7u) {
        let sum = rgb.x + rgb.y + rgb.z;
        let black = palette_sum(copy, 0u, colors);
        let white = palette_sum(copy, 7u, colors);
        if sum < (black + white) / 2u {
            index = 0u;
        } else {
            index = 7u;
        }
    }
    return index;
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

// bits_per_module is the base-two logarithm of the colour count, which is how
// many bits one classified module carries.
fn bits_per_module(colors: u32) -> u32 {
    var count = 0u;
    var value = colors;
    while value > 1u {
        value >>= 1u;
        count += 1u;
    }
    return count;
}

@compute @workgroup_size(1)
fn main() {
    if record[RECORD_STATUS] != STATUS_OK {
        return;
    }
    let colors = record[RECORD_COLORS];
    let copies = mode_word(record[RECORD_NC], MODE_COPIES);
    let per_module = bits_per_module(colors);
    let width = i32(params[PARAM_SIDE_X]);
    let height = i32(params[PARAM_SIDE_Y]);

    var position = vec2<i32>(i32(record[RECORD_WALK_X]), i32(record[RECORD_WALK_Y]));
    var taken = record[RECORD_MODULES];
    var emitted = 0u;
    while emitted < PART2_LENGTH {
        if position.x < 0 || position.y < 0 || position.x >= width || position.y >= height {
            record[RECORD_STATUS] = 3u;
            return;
        }
        let value = classify(
            module_rgb(position.x, position.y),
            u32(position.x), u32(position.y), colors, copies,
        );
        for (var i = 0u; i < per_module && emitted < PART2_LENGTH; i += 1u) {
            bits[emitted] = (value >> (per_module - 1u - i)) & 1u;
            emitted += 1u;
        }
        taken += 1u;
        advance_metadata_module(&position, i32(taken));
    }
    record[RECORD_MODULES] = taken;
    record[RECORD_WALK_X] = u32(position.x);
    record[RECORD_WALK_Y] = u32(position.y);
}
