// Builds one fixed directional sweep from resident image state and the prior
// finder decision. The command stream records every possible direction, but a
// zero decision dimension writes a zero-width indirect scan. This preserves the
// early-stop traversal without asking the host whether the previous fold won.

const BIN_WIDTH: u32 = 0u;
const BIN_HEIGHT: u32 = 1u;
const BIN_FLAGS: u32 = 5u;
const BIN_ROW_STRIDE: u32 = 9u;

const DIRECTION_COUNT: u32 = 6u;
const SCAN_CAPACITY: u32 = 262144u;
const COMPACT_CAPACITY: u32 = 33268u;
const SCAN_SKIP_CROSS_CHECK: u32 = 2u;

const CHAIN_CLASSIFY_CURRENT: u32 = 3264u;
const CHAIN_CLASSIFY_BSI: u32 = 1876u;
const CHAIN_CROSS_COLOR_BITS: u32 = 0u;
const CHAIN_COLOR_SOURCE: u32 = 2u;

@group(0) @binding(0) var<storage, read> binarizer: array<u32>;
@group(0) @binding(1) var<storage, read> decision: array<u32>;
@group(0) @binding(2) var<storage, read_write> cursor: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> scan_params: array<u32>;
@group(0) @binding(4) var<storage, read_write> chain_params: array<u32>;
@group(0) @binding(5) var<storage, read_write> scan_args: array<u32>;

fn direction_degrees(slot: u32) -> f32 {
    switch slot {
    case 0u: { return 90.0; }
    case 1u: { return 15.0; }
    case 2u: { return 30.0; }
    case 3u: { return 45.0; }
    case 4u: { return 60.0; }
    default: { return 75.0; }
    }
}

fn direction(angle: f32) -> vec3<f32> {
    let radians = angle * 0.017453292519943295;
    let c = cos(radians);
    let s = sin(radians);
    let major = max(abs(c), abs(s));
    return vec3<f32>(c / major, s / major, 1.0 / major);
}

fn put_direction(word: u32, value: vec3<f32>) {
    chain_params[word + 0u] = bitcast<u32>(value.x);
    chain_params[word + 1u] = bitcast<u32>(value.y);
    chain_params[word + 2u] = bitcast<u32>(value.z);
}

@compute @workgroup_size(1)
fn main() {
    let slot = atomicAdd(&cursor[0], 1u);
    if slot >= DIRECTION_COUNT {
        scan_args[0] = 0u;
        scan_args[1] = 1u;
        scan_args[2] = 1u;
        scan_args[3] = 0u;
        return;
    }

    let width = binarizer[BIN_WIDTH];
    let height = binarizer[BIN_HEIGHT];
    let step = max(binarizer[BIN_ROW_STRIDE], 1u);
    let degrees = direction_degrees(slot);
    let base = direction(degrees);
    let radians = degrees * 0.017453292519943295;
    let nx = -sin(radians);
    let ny = cos(radians);
    let x_extent = f32(max(width, 1u) - 1u);
    let y_extent = f32(max(height, 1u) - 1u);
    let q_lo = min(0.0, x_extent * nx) + min(0.0, y_extent * ny);
    let q_hi = max(0.0, x_extent * nx) + max(0.0, y_extent * ny);
    let line_count = u32(floor((q_hi - q_lo) / f32(step))) + 1u;

    scan_params[0] = width;
    scan_params[1] = height;
    scan_params[2] = 2u;
    scan_params[3] = width + height;
    scan_params[4] = bitcast<u32>(base.x);
    scan_params[5] = bitcast<u32>(base.y);
    scan_params[6] = bitcast<u32>(nx);
    scan_params[7] = bitcast<u32>(ny);
    scan_params[8] = bitcast<u32>(q_lo);
    scan_params[9] = bitcast<u32>(f32(step));
    scan_params[10] = line_count;
    scan_params[11] = 0u;
    scan_params[12] = 0u;
    scan_params[13] = SCAN_SKIP_CROSS_CHECK;

    chain_params[0] = width;
    chain_params[1] = height;
    chain_params[2] = SCAN_CAPACITY;
    chain_params[3] = ((binarizer[BIN_FLAGS] & 2u) >> 1u) | CHAIN_COLOR_SOURCE;
    chain_params[4] = CHAIN_CLASSIFY_CURRENT;
    chain_params[5] = CHAIN_CLASSIFY_BSI;
    chain_params[6] = CHAIN_CROSS_COLOR_BITS;
    chain_params[7] = COMPACT_CAPACITY;
    chain_params[8] = bitcast<u32>(base.x);
    chain_params[9] = bitcast<u32>(base.y);
    chain_params[10] = bitcast<u32>(nx);
    chain_params[11] = bitcast<u32>(ny);
    chain_params[12] = bitcast<u32>(q_lo);
    chain_params[13] = bitcast<u32>(f32(step));
    put_direction(14u, base);
    put_direction(17u, direction(degrees + 90.0));
    put_direction(20u, direction(degrees + 45.0));
    put_direction(23u, direction(degrees - 45.0));

    let active = select(0u, line_count, decision[0] != 0u);
    scan_args[0] = active;
    scan_args[1] = 3u;
    scan_args[2] = 1u;
    scan_args[3] = 0u;
}
