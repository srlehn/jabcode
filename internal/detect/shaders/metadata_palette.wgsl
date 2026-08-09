// Reads the symbol's embedded colour palette off the resident module grid and
// derives everything the classifier needs from it: the normalized entries and
// the per-copy black thresholds.
//
// It runs after Part I's correction and reads the colour mode straight out of
// the corrector's output, so the host never learns it and never has to say how
// many palette modules there are. A symbol embeds several copies of its palette,
// four at or below eight colours and two above, and a module is classified later
// against whichever copy is nearest, which is what corrects a capture's local
// colour cast where the module actually sits.
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

const DISTANCE_TO_BORDER: i32 = 4;

const GRID_MODULES: u32 = 1u;

const PARAM_SIDE_X: u32 = 0u;
const PARAM_SIDE_Y: u32 = 1u;
// The colour mode a symbol carrying no explicit metadata is read in.
const PARAM_DEFAULT_NC: u32 = 2u;

// One description per colour mode, indexed by NC. Every field is a property of
// the format that the host already states somewhere, so the kernel reads the
// shape of the mode the corrector reported rather than branching on colour
// counts of its own. A mode the host did not describe has zero copies and is
// declined, which is what makes the covered set a host table.
const PARAM_MODE: u32 = 8u;
const MODE_WORDS: u32 = 5u;
const MODE_COPIES: u32 = 0u;
const MODE_SLOTS: u32 = 1u;
const MODE_FINDER: u32 = 2u;
const MODE_BASE: u32 = 3u;
const MODE_THRESHOLD: u32 = 4u;

// Which palette entry copy c carries at slot i, already reduced modulo the
// colour count and already resolved for the wire variant, because that choice
// is the host's and does not belong in a kernel. Each mode's table starts at its
// own MODE_BASE.
const PARAM_PLACEMENT: u32 = 48u;

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
const RECORD_NORMALIZED: u32 = 1552u;
const RECORD_THRESHOLDS: u32 = 1680u;

// How many entries the normalized region holds. Only the modes at or below
// eight colours are classified against it, so it is sized for those rather than
// for the whole colour range.
const NORMALIZED_ENTRIES: u32 = 32u;

// How many spatial corners the classifier ranks a module against, which is four
// whatever the mode: a mode embedding fewer palettes folds the winner onto one
// it has, exactly as the host does.
const PALETTE_CORNERS: u32 = 4u;

@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> grid: array<u32>;
@group(0) @binding(2) var<storage, read> net: array<u32>;
@group(0) @binding(3) var<storage, read_write> record: array<u32>;

fn module_rgb(x: i32, y: i32) -> vec3<u32> {
    let side_x = i32(params[PARAM_SIDE_X]);
    let word = grid[GRID_MODULES + u32(y * side_x + x)];
    return vec3<u32>(word & 0xffu, (word >> 8u) & 0xffu, (word >> 16u) & 0xffu);
}

fn mode_word(nc: u32, field: u32) -> u32 {
    return params[PARAM_MODE + nc * MODE_WORDS + field];
}

// The colour mode a symbol without explicit metadata is read in. It is a
// constant of the format, so the host states it rather than the kernel naming a
// number of its own.
fn mode_default_nc() -> u32 {
    return params[PARAM_DEFAULT_NC];
}

fn placement(nc: u32, copy: u32, slot: u32) -> u32 {
    let slots = mode_word(nc, MODE_SLOTS);
    return params[PARAM_PLACEMENT + mode_word(nc, MODE_BASE) + copy * slots + slot];
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
// embed every colour instead, and their mode description says so by carrying no
// finder colours, so this is never reached for them.
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

fn palette_byte(at: u32) -> u32 {
    return record[RECORD_PALETTE + at];
}

fn put_palette_byte(at: u32, value: u32) {
    record[RECORD_PALETTE + at] = value;
}

fn lerp_byte(a: u32, b: u32, wa: u32, wb: u32) -> u32 {
    return (a * wa + b * wb) / (wa + wb);
}

fn copy_palette(dst: u32, src: u32, count: u32) {
    for (var j = 0u; j < count; j += 1u) {
        put_palette_byte(dst + j, palette_byte(src + j));
    }
}

// interpolate_block fills the four gap blocks of a 128- or 256-colour copy from
// the four blocks the symbol actually carries, span bytes each.
fn interpolate_block(offset: u32, span: u32) {
    for (var j = 0u; j < span; j += 1u) {
        put_palette_byte(offset + span + j, lerp_byte(
            palette_byte(offset + j), palette_byte(offset + 2u * span + j), 1u, 1u));
    }
    for (var j = 0u; j < span; j += 1u) {
        let lo = palette_byte(offset + 2u * span + j);
        let hi = palette_byte(offset + 5u * span + j);
        put_palette_byte(offset + 3u * span + j, lerp_byte(lo, hi, 2u, 1u));
        put_palette_byte(offset + 4u * span + j, lerp_byte(lo, hi, 1u, 2u));
    }
    for (var j = 0u; j < span; j += 1u) {
        put_palette_byte(offset + 6u * span + j, lerp_byte(
            palette_byte(offset + 5u * span + j),
            palette_byte(offset + 7u * span + j), 1u, 1u));
    }
}

// expand_subblock spreads a sixteen-colour sub-block of the carried palette over
// the thirty-two-colour block at dst, interpolating the four gap quarters.
fn expand_subblock(dst: u32, src: u32) {
    // Highest destination first, so a sub-block expanded onto itself still reads
    // sources the expansion has not yet overwritten.
    copy_palette(dst + 84u, src + 36u, 12u);
    copy_palette(dst + 60u, src + 24u, 12u);
    copy_palette(dst + 24u, src + 12u, 12u);
    copy_palette(dst, src, 12u);
    for (var j = 0u; j < 12u; j += 1u) {
        put_palette_byte(dst + 12u + j, lerp_byte(
            palette_byte(dst + j), palette_byte(dst + 24u + j), 1u, 1u));
    }
    for (var j = 0u; j < 12u; j += 1u) {
        put_palette_byte(dst + 36u + j, lerp_byte(
            palette_byte(dst + 24u + j), palette_byte(dst + 60u + j), 2u, 1u));
        put_palette_byte(dst + 48u + j, lerp_byte(
            palette_byte(dst + j), palette_byte(dst + 60u + j), 1u, 2u));
    }
    for (var j = 0u; j < 12u; j += 1u) {
        put_palette_byte(dst + 72u + j, lerp_byte(
            palette_byte(dst + 60u + j), palette_byte(dst + 84u + j), 1u, 1u));
    }
}

// interpolate_palette reconstructs a 128- or 256-colour palette from the 64
// representatives the symbol embeds.
//
// It runs here rather than on the host because Part II is classified on this
// side, against the whole palette: a walk that handed back the representatives
// and left the reconstruction to the caller would classify Part II's modules
// against a palette three quarters of which is still zero.
fn interpolate_palette(colors: u32, copies: u32) {
    if colors != 128u && colors != 256u {
        return;
    }
    for (var copy = 0u; copy < copies; copy += 1u) {
        let offset = colors * 3u * copy;
        if colors == 128u {
            // Sixteen colours per block. Block one is already in place.
            copy_palette(offset + 336u, offset + 144u, 48u);
            copy_palette(offset + 240u, offset + 96u, 48u);
            copy_palette(offset + 96u, offset + 48u, 48u);
            interpolate_block(offset, 48u);
        } else {
            expand_subblock(offset + 672u, offset + 144u);
            expand_subblock(offset + 480u, offset + 96u);
            expand_subblock(offset + 192u, offset + 48u);
            expand_subblock(offset, offset);
            interpolate_block(offset, 96u);
        }
    }
}

// normalize_palette precomputes each entry's unit-max direction and its mean
// intensity, which is what the classifier compares a module against. An exact
// black entry normalizes to the zero vector rather than dividing by zero: a NaN
// entry would never win a distance comparison, which drops black out of every
// ranking and leaves it reachable only through the threshold shortcut below.
fn normalize_palette(colors: u32, copies: u32) {
    // Above eight colours the classifier ranks absolute distance against the
    // palette bytes and never reads a normalized entry, so there is nothing to
    // derive. The condition is the classifier's own, not the region's capacity:
    // sixteen colours in two copies is exactly NORMALIZED_ENTRIES, so a capacity
    // test would fill entries no stage would ever read.
    if colors > 8u {
        return;
    }
    for (var i = 0u; i < colors * copies; i += 1u) {
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
// layout, so the two colour modes that have a rule name them outright.
//
// A mode with no rule leaves every threshold zero, and that is the host's
// behaviour rather than an omission: getPaletteThreshold covers four and eight
// colours and nothing else, so the black shortcut simply never fires above
// eight. Inventing a rule here would classify modules black that the host
// classifies by distance, and hard LDPC has nothing underneath it to report the
// difference.
// Every corner is written, not just the copies this mode embeds. The classifier
// indexes thresholds by the nearest of four spatial corners whatever the mode,
// and the record outlives one walk, so a corner left alone would be answered
// from whichever symbol was walked before this one.
fn palette_thresholds(colors: u32, copies: u32, rule: u32) {
    for (var copy = 0u; copy < PALETTE_CORNERS; copy += 1u) {
        var low = vec3<u32>(0u);
        var high = vec3<u32>(0u);
        if copy < copies && rule == 4u {
            let c0 = palette_entry(copy, 0u, colors);
            let c1 = palette_entry(copy, 1u, colors);
            let c2 = palette_entry(copy, 2u, colors);
            let c3 = palette_entry(copy, 3u, colors);
            low = vec3<u32>(max(c0.x, c1.x), max(c0.y, c2.y), max(c2.z, c3.z));
            high = vec3<u32>(min(c2.x, c3.x), min(c1.y, c3.y), min(c0.z, c1.z));
        } else if copy < copies && rule == 8u {
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
    let status = record[RECORD_STATUS];
    if status != STATUS_OK && status != STATUS_DEFAULT {
        return;
    }
    // A symbol carrying no explicit colour mode is read in the default one,
    // whose palette occupies the modules Part I was read from. Part I left the
    // walk at the start for that, and Part II is skipped, so the only thing this
    // stage does differently is take the mode from the host's default rather
    // than from the corrector's output.
    var nc = mode_default_nc();
    if status == STATUS_OK {
        // The corrector writes one status word per block ahead of the message,
        // and Part I is a single block, so its three bits start at index one.
        nc = (net[1] << 2u) + (net[2] << 1u) + net[3];
        record[RECORD_PART1_SYNDROME] = net[0];
    }
    let colors = 1u << (nc + 1u);
    record[RECORD_NC] = nc;
    record[RECORD_COLORS] = colors;
    let copies = mode_word(nc, MODE_COPIES);
    if copies == 0u {
        record[RECORD_STATUS] = STATUS_UNSUPPORTED;
        return;
    }
    let finder_colors = mode_word(nc, MODE_FINDER);
    let slots = mode_word(nc, MODE_SLOTS);

    // A mode whose finder carries no palette colour skips this entirely and
    // reads every colour from the strip below.
    for (var copy = 0u; copy < copies; copy += 1u) {
        let positions = finder_palette_positions(copy);
        for (var slot = 0u; slot < finder_colors; slot += 1u) {
            let at = positions[slot];
            write_palette(copy, placement(nc, copy, slot), colors,
                module_rgb(at.x, at.y));
        }
    }

    let width = i32(params[PARAM_SIDE_X]);
    let height = i32(params[PARAM_SIDE_Y]);
    var position = vec2<i32>(i32(record[RECORD_WALK_X]), i32(record[RECORD_WALK_Y]));
    var taken = record[RECORD_MODULES];
    for (var slot = finder_colors; slot < slots; slot += 1u) {
        for (var copy = 0u; copy < copies; copy += 1u) {
            // A matrix too small for the declared colour mode walks the strip
            // off the symbol; that is a decline, not an out-of-range read.
            if position.x < 0 || position.y < 0 || position.x >= width || position.y >= height {
                record[RECORD_STATUS] = STATUS_FAILED;
                return;
            }
            write_palette(copy, placement(nc, copy, slot), colors,
                module_rgb(position.x, position.y));
            taken += 1u;
            advance_metadata_module(&position, i32(taken));
        }
    }

    interpolate_palette(colors, copies);
    normalize_palette(colors, copies);
    palette_thresholds(colors, copies, mode_word(nc, MODE_THRESHOLD));
    record[RECORD_MODULES] = taken;
    record[RECORD_WALK_X] = u32(position.x);
    record[RECORD_WALK_Y] = u32(position.y);
}
