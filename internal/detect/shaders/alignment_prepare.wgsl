// Starts a paired alignment candidate from one fixed direct-result slot. The
// direct result is the authority for the interpreted version, while resident
// finder control supplies the physical quad. No host-visible geometry sits
// between those two pieces of evidence.

const RESULT_MAGIC: u32 = 0x4A414252u;
const RESULT_VERSION: u32 = 3u;
const RESULT_WORDS: u32 = 5705u;
const RESULT_SLOTS: u32 = 36u;
const RESULT_META_STATUS: u32 = 2u;
const RESULT_VERSION_X: u32 = 7u;
const RESULT_VERSION_Y: u32 = 8u;
const RESULT_SIDE_X: u32 = 31u;
const RESULT_SIDE_Y: u32 = 32u;
const RESULT_PROFILE: u32 = 33u;
const RESULT_SLOT: u32 = 34u;

const STATUS_OK: u32 = 0u;
const STATUS_DEFAULT: u32 = 1u;
const STATUS_SIZE_MISMATCH: u32 = 4u;
const STATUS_ECC_ORDER: u32 = 5u;

const SAMPLE_WIDTH: u32 = 0u;
const SAMPLE_HEIGHT: u32 = 1u;
const SAMPLE_SIDE_X: u32 = 2u;
const SAMPLE_SIDE_Y: u32 = 3u;
const SAMPLE_DEST_X: u32 = 23u;
const SAMPLE_DEST_Y: u32 = 24u;
const SAMPLE_DEST_WIDTH: u32 = 25u;
const SAMPLE_DEST_HEIGHT: u32 = 26u;
const SAMPLE_METADATA_CONTROL: u32 = 27u;
const METADATA_PROFILE_MASK: u32 = 0xffu;
const METADATA_SLOT_SHIFT: u32 = 8u;

const CONTROL_VALID: u32 = 3u;
const CONTROL_PATTERNS: u32 = 4u;
const PATTERN_WORDS: u32 = 6u;
const PATTERN_DIRECTION: u32 = 1u;
const PATTERN_X: u32 = 2u;
const PATTERN_Y: u32 = 3u;
const PATTERN_MODULE: u32 = 4u;

const PARAM_WIDTH: u32 = 0u;
const PARAM_HEIGHT: u32 = 1u;
const PARAM_NAPX: u32 = 2u;
const PARAM_NAPY: u32 = 3u;
const PARAM_MODE: u32 = 4u;
const PARAM_CORE_R: u32 = 5u;
const PARAM_CORE_G: u32 = 6u;
const PARAM_CORE_B: u32 = 7u;
const PARAM_SIDE_X: u32 = 8u;
const PARAM_SIDE_Y: u32 = 9u;
const PARAM_MODULE_MAX: u32 = 10u;
const PARAM_QUAD: u32 = 11u;
const PARAM_AP_POS_X: u32 = 19u;
const PARAM_AP_POS_Y: u32 = 28u;
const PARAM_TRANSFORM: u32 = 37u;
const PARAM_CORNER_MODULE: u32 = 46u;
const PARAM_EXPLICIT: u32 = 50u;

const MODE_PREDICT: u32 = 0u;
const TILES: u32 = 32u;
const CELL_WORDS: u32 = 6u;
const CELL_FOUND: u32 = 0u;
const CELL_CX: u32 = 1u;
const CELL_CY: u32 = 2u;
const CELL_MODULE: u32 = 3u;
const CELL_DIRECTION: u32 = 4u;
const MAX_CELLS: u32 = 81u;

const INDIRECT_SAMPLE: u32 = 0u;
const INDIRECT_ATTEMPT: u32 = 3u;
const INDIRECT_PREDICT: u32 = 6u;
const INDIRECT_REDUCE_1: u32 = 9u;
const INDIRECT_REFINE: u32 = 12u;
const INDIRECT_REDUCE_2: u32 = 15u;
const INDIRECT_RECTS: u32 = 18u;
const INDIRECT_WORDS: u32 = 21u;

const AP_NUM: array<u32, 32> = array<u32, 32>(
    2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u,
    3u, 4u, 4u, 4u, 4u, 5u, 5u, 5u,
    5u, 6u, 6u, 6u, 6u, 7u, 7u, 7u,
    7u, 8u, 8u, 8u, 8u, 9u, 9u, 9u,
);

const AP_POS: array<u32, 288> = array<u32, 288>(
    4u,18u,0u,0u,0u,0u,0u,0u,0u,
    4u,22u,0u,0u,0u,0u,0u,0u,0u,
    4u,26u,0u,0u,0u,0u,0u,0u,0u,
    4u,30u,0u,0u,0u,0u,0u,0u,0u,
    4u,34u,0u,0u,0u,0u,0u,0u,0u,
    4u,17u,38u,0u,0u,0u,0u,0u,0u,
    4u,20u,42u,0u,0u,0u,0u,0u,0u,
    4u,23u,46u,0u,0u,0u,0u,0u,0u,
    4u,26u,50u,0u,0u,0u,0u,0u,0u,
    4u,14u,32u,54u,0u,0u,0u,0u,0u,
    4u,17u,39u,58u,0u,0u,0u,0u,0u,
    4u,20u,46u,62u,0u,0u,0u,0u,0u,
    4u,23u,44u,66u,0u,0u,0u,0u,0u,
    4u,26u,37u,51u,70u,0u,0u,0u,0u,
    4u,14u,36u,58u,74u,0u,0u,0u,0u,
    4u,17u,39u,56u,78u,0u,0u,0u,0u,
    4u,20u,42u,63u,82u,0u,0u,0u,0u,
    4u,23u,38u,54u,70u,86u,0u,0u,0u,
    4u,26u,38u,56u,77u,90u,0u,0u,0u,
    4u,14u,33u,53u,72u,94u,0u,0u,0u,
    4u,17u,38u,59u,79u,98u,0u,0u,0u,
    4u,20u,36u,53u,70u,86u,102u,0u,0u,
    4u,23u,36u,55u,74u,93u,106u,0u,0u,
    4u,26u,36u,58u,79u,100u,110u,0u,0u,
    4u,14u,36u,58u,80u,92u,114u,0u,0u,
    4u,17u,34u,52u,70u,88u,99u,118u,0u,
    4u,20u,37u,54u,72u,89u,106u,122u,0u,
    4u,23u,38u,56u,74u,92u,113u,126u,0u,
    4u,26u,36u,58u,78u,98u,120u,130u,0u,
    4u,14u,32u,49u,67u,84u,102u,112u,134u,
    4u,17u,35u,53u,71u,89u,107u,119u,138u,
    4u,20u,38u,55u,73u,91u,108u,126u,142u,
);

@group(0) @binding(0) var<storage, read> result: array<u32>;
@group(0) @binding(1) var<storage, read_write> sample: array<u32>;
@group(0) @binding(2) var<storage, read> control: array<u32>;
@group(0) @binding(3) var<storage, read_write> params: array<u32>;
@group(0) @binding(4) var<storage, read_write> cells: array<u32>;
@group(0) @binding(5) var<storage, read_write> indirect: array<u32>;
@group(0) @binding(6) var<storage, read_write> sampled: array<atomic<u32>>;

fn finite(value: f32) -> bool {
    return value == value && abs(value) <= 3.402823e38;
}

fn pattern_word(slot: u32, field: u32) -> u32 {
    return control[CONTROL_PATTERNS + slot * PATTERN_WORDS + field];
}

fn square_to_quad() -> array<f32, 9> {
    let p0 = vec2<f32>(bitcast<f32>(pattern_word(0u, PATTERN_X)), bitcast<f32>(pattern_word(0u, PATTERN_Y)));
    let p1 = vec2<f32>(bitcast<f32>(pattern_word(1u, PATTERN_X)), bitcast<f32>(pattern_word(1u, PATTERN_Y)));
    let p2 = vec2<f32>(bitcast<f32>(pattern_word(2u, PATTERN_X)), bitcast<f32>(pattern_word(2u, PATTERN_Y)));
    let p3 = vec2<f32>(bitcast<f32>(pattern_word(3u, PATTERN_X)), bitcast<f32>(pattern_word(3u, PATTERN_Y)));
    let delta3 = p0 - p1 + p2 - p3;
    var out: array<f32, 9>;
    if delta3.x == 0.0 && delta3.y == 0.0 {
        out[0] = p1.x - p0.x;
        out[1] = p1.y - p0.y;
        out[2] = 0.0;
        out[3] = p3.x - p0.x;
        out[4] = p3.y - p0.y;
        out[5] = 0.0;
        out[6] = p0.x;
        out[7] = p0.y;
        out[8] = 1.0;
        return out;
    }
    let delta1 = p1 - p2;
    let delta2 = p3 - p2;
    let denominator = delta1.x * delta2.y - delta2.x * delta1.y;
    let a13 = (delta3.x * delta2.y - delta2.x * delta3.y) / denominator;
    let a23 = (delta1.x * delta3.y - delta3.x * delta1.y) / denominator;
    out[0] = p1.x - p0.x + a13 * p1.x;
    out[1] = p1.y - p0.y + a13 * p1.y;
    out[2] = a13;
    out[3] = p3.x - p0.x + a23 * p3.x;
    out[4] = p3.y - p0.y + a23 * p3.y;
    out[5] = a23;
    out[6] = p0.x;
    out[7] = p0.y;
    out[8] = 1.0;
    return out;
}

fn source_transform(quad: array<f32, 9>, x0: f32, y0: f32, x1: f32, y1: f32) -> array<f32, 9> {
    let dx = x1 - x0;
    let dy = y1 - y0;
    var out: array<f32, 9>;
    out[0] = quad[0] / dx;
    out[1] = quad[1] / dx;
    out[2] = quad[2] / dx;
    out[3] = quad[3] / dy;
    out[4] = quad[4] / dy;
    out[5] = quad[5] / dy;
    out[6] = quad[6] - x0 * out[0] - y0 * out[3];
    out[7] = quad[7] - x0 * out[1] - y0 * out[4];
    out[8] = quad[8] - x0 * out[2] - y0 * out[5];
    return out;
}

fn command(at: u32, x: u32) {
    indirect[at] = x;
    indirect[at + 1u] = 1u;
    indirect[at + 2u] = 1u;
}

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
    let lane = local.x;
    if lane < MAX_CELLS {
        for (var word = 0u; word < CELL_WORDS; word += 1u) {
            cells[lane * CELL_WORDS + word] = 0u;
        }
    }
    if lane < INDIRECT_WORDS {
        indirect[lane] = 0u;
    }
    if lane == 0u {
        atomicStore(&sampled[0], 0u);
    }
    storageBarrier();
    workgroupBarrier();
    if lane != 0u || control[CONTROL_VALID] == 0u {
        return;
    }

    let metadata_control = sample[SAMPLE_METADATA_CONTROL];
    let slot = (metadata_control >> METADATA_SLOT_SHIFT) & 0xffu;
    if slot >= RESULT_SLOTS {
        return;
    }
    let base = slot * RESULT_WORDS;
    if result[base] != RESULT_MAGIC || result[base + 1u] != RESULT_VERSION ||
        result[base + RESULT_SLOT] != slot ||
        result[base + RESULT_PROFILE] != (metadata_control & METADATA_PROFILE_MASK) ||
        result[base + RESULT_SIDE_X] == 0u || result[base + RESULT_SIDE_Y] == 0u {
        return;
    }
    let status = result[base + RESULT_META_STATUS];
    // Default mode needs physical side confirmation before AP coordinates are
    // authoritative. It stays on the dedicated resident confirmation task.
    if status == STATUS_DEFAULT ||
        (status != STATUS_OK && status != STATUS_SIZE_MISMATCH && status != STATUS_ECC_ORDER) {
        return;
    }
    let version_x = result[base + RESULT_VERSION_X];
    let version_y = result[base + RESULT_VERSION_Y];
    if version_x < 1u || version_x > 32u || version_y < 1u || version_y > 32u ||
        (version_x < 6u && version_y < 6u) {
        return;
    }
    let n_ap_x = AP_NUM[version_x - 1u];
    let n_ap_y = AP_NUM[version_y - 1u];
    let cell_count = n_ap_x * n_ap_y;
    let side_x = 17u + 4u * version_x;
    let side_y = 17u + 4u * version_y;
    let width = sample[SAMPLE_WIDTH];
    let height = sample[SAMPLE_HEIGHT];
    if cell_count > MAX_CELLS || width == 0u || height == 0u {
        return;
    }

    let first_x = f32(AP_POS[(version_x - 1u) * 9u]);
    let first_y = f32(AP_POS[(version_y - 1u) * 9u]);
    let last_x = f32(AP_POS[(version_x - 1u) * 9u + n_ap_x - 1u]);
    let last_y = f32(AP_POS[(version_y - 1u) * 9u + n_ap_y - 1u]);
    let transform = source_transform(square_to_quad(), first_x, first_y, last_x, last_y);
    for (var word = 0u; word < 9u; word += 1u) {
        if !finite(transform[word]) {
            return;
        }
    }

    params[PARAM_WIDTH] = width;
    params[PARAM_HEIGHT] = height;
    params[PARAM_NAPX] = n_ap_x;
    params[PARAM_NAPY] = n_ap_y;
    params[PARAM_MODE] = MODE_PREDICT;
    // APX has the default palette's yellow core.
    params[PARAM_CORE_R] = 1u;
    params[PARAM_CORE_G] = 1u;
    params[PARAM_CORE_B] = 0u;
    params[PARAM_SIDE_X] = side_x;
    params[PARAM_SIDE_Y] = side_y;
    params[PARAM_EXPLICIT] = 0u;
    var module_sum = 0.0;
    for (var corner = 0u; corner < 4u; corner += 1u) {
        let cx = bitcast<f32>(pattern_word(corner, PATTERN_X));
        let cy = bitcast<f32>(pattern_word(corner, PATTERN_Y));
        let module = bitcast<f32>(pattern_word(corner, PATTERN_MODULE));
        if !finite(cx) || !finite(cy) || !finite(module) || module <= 0.0 {
            return;
        }
        params[PARAM_QUAD + corner * 2u] = bitcast<u32>(cx);
        params[PARAM_QUAD + corner * 2u + 1u] = bitcast<u32>(cy);
        params[PARAM_CORNER_MODULE + corner] = bitcast<u32>(module);
        module_sum += module;
    }
    params[PARAM_MODULE_MAX] = bitcast<u32>(0.75 * module_sum);
    for (var at = 0u; at < 9u; at += 1u) {
        params[PARAM_AP_POS_X + at] = AP_POS[(version_x - 1u) * 9u + at];
        params[PARAM_AP_POS_Y + at] = AP_POS[(version_y - 1u) * 9u + at];
        params[PARAM_TRANSFORM + at] = bitcast<u32>(transform[at]);
    }

    let corners = array<u32, 4>(
        0u,
        n_ap_x - 1u,
        cell_count - 1u,
        (n_ap_y - 1u) * n_ap_x,
    );
    for (var corner = 0u; corner < 4u; corner += 1u) {
        let cell = corners[corner] * CELL_WORDS;
        cells[cell + CELL_FOUND] = 1u;
        cells[cell + CELL_CX] = pattern_word(corner, PATTERN_X);
        cells[cell + CELL_CY] = pattern_word(corner, PATTERN_Y);
        cells[cell + CELL_MODULE] = pattern_word(corner, PATTERN_MODULE);
        cells[cell + CELL_DIRECTION] = pattern_word(corner, PATTERN_DIRECTION);
    }

    sample[SAMPLE_SIDE_X] = side_x;
    sample[SAMPLE_SIDE_Y] = side_y;
    sample[SAMPLE_DEST_X] = 0u;
    sample[SAMPLE_DEST_Y] = 0u;
    sample[SAMPLE_DEST_WIDTH] = side_x;
    sample[SAMPLE_DEST_HEIGHT] = side_y;
    atomicStore(&sampled[0], 1u);

    command(INDIRECT_SAMPLE, (side_x * side_y + 63u) / 64u);
    command(INDIRECT_ATTEMPT, 1u);
    command(INDIRECT_PREDICT, cell_count * TILES);
    command(INDIRECT_REDUCE_1, cell_count);
    command(INDIRECT_REFINE, cell_count * TILES);
    command(INDIRECT_REDUCE_2, cell_count);
    command(INDIRECT_RECTS, 1u);
}
