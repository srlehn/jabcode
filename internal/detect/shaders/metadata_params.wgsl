// Builds the metadata walk's format table from the resident sampled shape.
// Only the wire profile is selected by command control. Symbol dimensions come
// from the sampler, and every other value is an immutable format rule, so no
// per-frame metadata block needs to cross the device boundary.

const PROFILE_ISO: u32 = 0u;
const PROFILE_ISO_HIGH_COLOR: u32 = 1u;
const PROFILE_CURRENT_C: u32 = 2u;

// Alignment sampling leaves the last block's local size in words 2 and 3.
// Metadata addresses the assembled grid, whose full shape is the sampler's
// destination size even when the last block covers only one alignment cell.
const SAMPLE_SIDE_X: u32 = 25u;
const SAMPLE_SIDE_Y: u32 = 26u;
const SAMPLE_METADATA_PROFILE: u32 = 27u;

const PARAM_SIDE_X: u32 = 0u;
const PARAM_SIDE_Y: u32 = 1u;
const PARAM_DEFAULT_NC: u32 = 2u;
const PARAM_GENERATOR: u32 = 3u;
const PARAM_DEFAULT_WC: u32 = 4u;
const PARAM_DEFAULT_WR: u32 = 5u;
const PARAM_DEFAULT_MASK: u32 = 6u;
const PARAM_MODE: u32 = 8u;
const MODE_WORDS: u32 = 5u;
const MODE_COPIES: u32 = 0u;
const MODE_SLOTS: u32 = 1u;
const MODE_FINDER: u32 = 2u;
const MODE_BASE: u32 = 3u;
const MODE_THRESHOLD: u32 = 4u;
const PARAM_PLACEMENT: u32 = 48u;
const PARAM_AP_NUM_X: u32 = 576u;
const PARAM_AP_NUM_Y: u32 = 577u;
const PARAM_AP_POS_X: u32 = 578u;
const PARAM_AP_POS_Y: u32 = 587u;
const PARAM_WORDS: u32 = 596u;

const MODE_BASES = array<u32, 7>(0u, 16u, 48u, 80u, 144u, 272u, 400u);
const AP_NUM = array<u32, 32>(
    2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u,
    3u, 4u, 4u, 4u, 4u, 5u, 5u, 5u,
    5u, 6u, 6u, 6u, 6u, 7u, 7u, 7u,
    7u, 8u, 8u, 8u, 8u, 9u, 9u, 9u,
);
const AP_POS = array<u32, 288>(
    4u, 18u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    4u, 22u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    4u, 26u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    4u, 30u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    4u, 34u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    4u, 17u, 38u, 0u, 0u, 0u, 0u, 0u, 0u,
    4u, 20u, 42u, 0u, 0u, 0u, 0u, 0u, 0u,
    4u, 23u, 46u, 0u, 0u, 0u, 0u, 0u, 0u,
    4u, 26u, 50u, 0u, 0u, 0u, 0u, 0u, 0u,
    4u, 14u, 32u, 54u, 0u, 0u, 0u, 0u, 0u,
    4u, 17u, 39u, 58u, 0u, 0u, 0u, 0u, 0u,
    4u, 20u, 46u, 62u, 0u, 0u, 0u, 0u, 0u,
    4u, 23u, 44u, 66u, 0u, 0u, 0u, 0u, 0u,
    4u, 26u, 37u, 51u, 70u, 0u, 0u, 0u, 0u,
    4u, 14u, 36u, 58u, 74u, 0u, 0u, 0u, 0u,
    4u, 17u, 39u, 56u, 78u, 0u, 0u, 0u, 0u,
    4u, 20u, 42u, 63u, 82u, 0u, 0u, 0u, 0u,
    4u, 23u, 38u, 54u, 70u, 86u, 0u, 0u, 0u,
    4u, 26u, 38u, 56u, 77u, 90u, 0u, 0u, 0u,
    4u, 14u, 33u, 53u, 72u, 94u, 0u, 0u, 0u,
    4u, 17u, 38u, 59u, 79u, 98u, 0u, 0u, 0u,
    4u, 20u, 36u, 53u, 70u, 86u, 102u, 0u, 0u,
    4u, 23u, 36u, 55u, 74u, 93u, 106u, 0u, 0u,
    4u, 26u, 36u, 58u, 79u, 100u, 110u, 0u, 0u,
    4u, 14u, 36u, 58u, 80u, 92u, 114u, 0u, 0u,
    4u, 17u, 34u, 52u, 70u, 88u, 99u, 118u, 0u,
    4u, 20u, 37u, 54u, 72u, 89u, 106u, 122u, 0u,
    4u, 23u, 38u, 56u, 74u, 92u, 113u, 126u, 0u,
    4u, 26u, 36u, 58u, 78u, 98u, 120u, 130u, 0u,
    4u, 14u, 32u, 49u, 67u, 84u, 102u, 112u, 134u,
    4u, 17u, 35u, 53u, 71u, 89u, 107u, 119u, 138u,
    4u, 20u, 38u, 55u, 73u, 91u, 108u, 126u, 142u,
);
const PRIMARY_PLACEMENT = array<u32, 32>(
    0u, 3u, 5u, 6u, 1u, 2u, 4u, 7u,
    0u, 6u, 5u, 3u, 1u, 2u, 4u, 7u,
    6u, 0u, 5u, 3u, 1u, 2u, 4u, 7u,
    3u, 0u, 5u, 6u, 1u, 2u, 4u, 7u,
);
const ISO_FOUR_PLACEMENT = array<u32, 16>(
    0u, 1u, 2u, 3u,
    0u, 3u, 2u, 1u,
    3u, 0u, 2u, 1u,
    1u, 0u, 2u, 3u,
);

@group(0) @binding(0) var<storage, read> sample: array<u32>;
@group(0) @binding(1) var<storage, read_write> grid: array<u32>;
@group(0) @binding(2) var<storage, read_write> params: array<u32>;

fn palette_placement(profile: u32, copy: u32, slot: u32, colors: u32) -> u32 {
    if profile != PROFILE_CURRENT_C && colors == 4u {
        return ISO_FOUR_PLACEMENT[copy * 4u + slot];
    }
    if slot < 8u {
        return PRIMARY_PLACEMENT[copy * 8u + slot];
    }
    return slot;
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
    let lane = local.x;
    for (var i = lane; i < PARAM_WORDS; i += 64u) {
        params[i] = 0u;
    }
    storageBarrier();
    workgroupBarrier();

    let profile = sample[SAMPLE_METADATA_PROFILE];
    if lane == 0u {
        let side_x = sample[SAMPLE_SIDE_X];
        let side_y = sample[SAMPLE_SIDE_Y];
        let valid_x = side_x >= 21u && side_x <= 145u && (side_x - 21u) % 4u == 0u;
        let valid_y = side_y >= 21u && side_y <= 145u && (side_y - 21u) % 4u == 0u;
        if !valid_x || !valid_y || profile > PROFILE_CURRENT_C {
            grid[0] = 0u;
            return;
        }
        params[PARAM_SIDE_X] = side_x;
        params[PARAM_SIDE_Y] = side_y;
        params[PARAM_DEFAULT_NC] = 2u;
        params[PARAM_GENERATOR] = select(0u, 1u, profile == PROFILE_CURRENT_C);
        params[PARAM_DEFAULT_WC] = 4u;
        params[PARAM_DEFAULT_WR] = 9u;
        params[PARAM_DEFAULT_MASK] = 7u;

        let version_x = (side_x - 21u) / 4u;
        let version_y = (side_y - 21u) / 4u;
        params[PARAM_AP_NUM_X] = AP_NUM[version_x];
        params[PARAM_AP_NUM_Y] = AP_NUM[version_y];
        for (var i = 0u; i < 9u; i += 1u) {
            params[PARAM_AP_POS_X + i] = AP_POS[version_x * 9u + i];
            params[PARAM_AP_POS_Y + i] = AP_POS[version_y * 9u + i];
        }
    }

    if lane < 1u || lane > 7u {
        return;
    }
    if profile == PROFILE_ISO && lane > 2u {
        return;
    }
    let colors = 1u << (lane + 1u);
    let copies = select(2u, 4u, colors <= 8u);
    let slots = min(colors, 64u);
    let base = MODE_BASES[lane - 1u];
    let mode = PARAM_MODE + lane * MODE_WORDS;
    params[mode + MODE_COPIES] = copies;
    params[mode + MODE_SLOTS] = slots;
    params[mode + MODE_FINDER] = select(0u, 2u, colors <= 8u);
    params[mode + MODE_BASE] = base;
    params[mode + MODE_THRESHOLD] = select(0u, colors, colors <= 8u);
    for (var copy = 0u; copy < copies; copy += 1u) {
        for (var slot = 0u; slot < slots; slot += 1u) {
            params[PARAM_PLACEMENT + base + copy * slots + slot] =
                palette_placement(profile, copy, slot, colors) % colors;
        }
    }
}
