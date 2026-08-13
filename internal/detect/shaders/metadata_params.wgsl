// Publishes one metadata parameter block from tables retained when the route
// context is created. Palette placement and format defaults are immutable;
// rebuilding them in every candidate dispatch wasted work and made a tiny
// control step depend on compiler handling of large shader constant arrays.
// Only the sampled side and its alignment positions vary per attempt.

const PROFILE_COUNT: u32 = 3u;
const SAMPLE_SIDE_X: u32 = 25u;
const SAMPLE_SIDE_Y: u32 = 26u;
const SAMPLE_METADATA_PROFILE: u32 = 27u;
const SAMPLE_METADATA_PROFILE_MASK: u32 = 0xffu;

const PARAM_SIDE_X: u32 = 0u;
const PARAM_SIDE_Y: u32 = 1u;
const PARAM_AP_NUM_X: u32 = 576u;
const PARAM_AP_NUM_Y: u32 = 577u;
const PARAM_AP_POS_X: u32 = 578u;
const PARAM_AP_POS_Y: u32 = 587u;
const PARAM_WORDS: u32 = 596u;

const STATIC_AP_NUM_BASE: u32 = 1788u;
const STATIC_AP_POS_BASE: u32 = 1820u;

@group(0) @binding(0) var<storage, read> sample: array<u32>;
@group(0) @binding(1) var<storage, read_write> grid: array<u32>;
@group(0) @binding(2) var<storage, read_write> params: array<u32>;
@group(0) @binding(3) var<storage, read> tables: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
    let lane = local.x;
    let profile = sample[SAMPLE_METADATA_PROFILE] & SAMPLE_METADATA_PROFILE_MASK;
    let side_x = sample[SAMPLE_SIDE_X];
    let side_y = sample[SAMPLE_SIDE_Y];
    let valid_x = side_x >= 21u && side_x <= 145u && (side_x - 21u) % 4u == 0u;
    let valid_y = side_y >= 21u && side_y <= 145u && (side_y - 21u) % 4u == 0u;
    if profile >= PROFILE_COUNT || !valid_x || !valid_y {
        if lane == 0u {
            grid[0] = 0u;
        }
        return;
    }

    let template = profile * PARAM_WORDS;
    for (var word = lane; word < PARAM_WORDS; word += 64u) {
        params[word] = tables[template + word];
    }
    storageBarrier();
    workgroupBarrier();

    if lane != 0u {
        return;
    }
    let version_x = (side_x - 21u) / 4u;
    let version_y = (side_y - 21u) / 4u;
    params[PARAM_SIDE_X] = side_x;
    params[PARAM_SIDE_Y] = side_y;
    params[PARAM_AP_NUM_X] = tables[STATIC_AP_NUM_BASE + version_x];
    params[PARAM_AP_NUM_Y] = tables[STATIC_AP_NUM_BASE + version_y];
    for (var at = 0u; at < 9u; at += 1u) {
        params[PARAM_AP_POS_X + at] = tables[STATIC_AP_POS_BASE + version_x * 9u + at];
        params[PARAM_AP_POS_Y + at] = tables[STATIC_AP_POS_BASE + version_y * 9u + at];
    }
}
