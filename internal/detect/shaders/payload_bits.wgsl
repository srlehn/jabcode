// Turns the resident module grid into the deinterleaved codeword the hard LDPC
// corrector reads, with no host stage in between: classification against the
// embedded palette, the position mask, bit expansion and the deinterleaving
// permutation all happen where the grid already is.
//
// The host runs these as four passes over three intermediate slices, because
// each is a different loop shape. A lane instead owns one module for the whole
// chain: it reads its own pixel, resolves its own colour, unmasks it, and
// scatters its own bits to their final codeword positions. Nothing is carried
// between modules, so the four passes cost one.

const NOT_DATA: u32 = 0xffffffffu;
// The sampler writes its reject flag in the first word of the grid buffer.
const GRID_MODULES: u32 = 1u;

const PARAM_SIDE_X: u32 = 0u;
const PARAM_SIDE_Y: u32 = 1u;
const PARAM_COLOR_NUMBER: u32 = 3u;
const PARAM_BITS_PER_MODULE: u32 = 4u;
const PARAM_MASK_TYPE: u32 = 5u;
const PARAM_GROSS_BITS: u32 = 8u;
const PARAM_PALETTE_COPIES: u32 = 10u;
const PARAM_PALETTE_THRESHOLDS: u32 = 30u;
const PARAM_PALETTE_EXTREMES: u32 = 42u;
const PARAM_NORMALIZED_PALETTE: u32 = 50u;
// The palette bytes themselves, which only the modes above eight colours read.
// They and the normalized entries never both apply, so the two regions could
// have shared one; keeping them apart costs a few hundred words and means the
// classifier that reads one never has to know how the other was packed.
const PARAM_PALETTE_BYTES: u32 = 178u;

@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> grid: array<u32>;
@group(0) @binding(2) var<storage, read> map: array<u32>;
@group(0) @binding(3) var<storage, read> permutation: array<u32>;
@group(0) @binding(4) var<storage, read_write> codeword: array<u32>;

fn param_f32(index: u32) -> f32 {
    return bitcast<f32>(params[index]);
}

// nearest_palette picks the embedded palette copy closest to a module, which is
// what corrects a capture's local colour cast where the module actually sits.
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

// classify_absolute ranks raw palette distance, which is what the host does
// above eight colours: those modes normalize nothing and have no black
// threshold, so neither the shortcut nor the direction ranking below applies.
//
// The corner is folded onto a copy rather than ranked among the embedded ones.
// nearest_palette ranks four spatial corners whatever the mode, and a
// high-colour symbol embeds two palettes, so the host takes the remainder; a
// device that ranked only two corners would disagree with it over a whole
// quadrant of the symbol.
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

// classify maps a module's sampled colour to a palette index: black first, then
// the nearest normalized palette entry, then the black/white tie-break the
// eight-colour palette needs because its two achromatic entries normalize alike.
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

fn mask_value(pattern: u32, x: u32, y: u32) -> u32 {
    switch pattern {
        case 0u: { return x + y; }
        case 1u: { return x; }
        case 2u: { return y; }
        case 3u: { return x / 2u + y / 3u; }
        case 4u: { return x / 3u + y / 2u; }
        case 5u: { return (x + y) / 2u + (x + y) / 3u; }
        case 6u: { return (x * x * y) % 7u + (2u * x * x + 2u * y) % 19u; }
        case 7u: { return (x * y * y) % 5u + (2u * x + y * y) % 13u; }
        default: { return 0u; }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let side_x = params[PARAM_SIDE_X];
    let side_y = params[PARAM_SIDE_Y];
    if id.x >= side_x * side_y {
        return;
    }
    let position = map[id.x];
    if position == NOT_DATA {
        return;
    }
    // The map is numbered down each column in turn; the grid is written across
    // each row, so the two indexings differ and both are derived here.
    let x = id.x / side_y;
    let y = id.x % side_y;
    let word = grid[GRID_MODULES + y * side_x + x];
    let rgb = vec3<u32>(word & 0xffu, (word >> 8u) & 0xffu, (word >> 16u) & 0xffu);

    let colors = params[PARAM_COLOR_NUMBER];
    let copies = params[PARAM_PALETTE_COPIES];
    let unmasked = classify(rgb, x, y, side_x, side_y, colors, copies) ^
        (mask_value(params[PARAM_MASK_TYPE], x, y) % colors);

    let bits = params[PARAM_BITS_PER_MODULE];
    let gross = params[PARAM_GROSS_BITS];
    for (var bit = 0u; bit < bits; bit += 1u) {
        let at = position * bits + bit;
        // Bits past the gross length are the padding the host drops before
        // deinterleaving, and the permutation is not defined for them.
        if at < gross {
            codeword[permutation[at]] = (unmasked >> (bits - 1u - bit)) & 1u;
        }
    }
}
