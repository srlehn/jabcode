// Selects one retained metadata parity graph and publishes its fixed hard-LDPC
// shape. The current sample chooses the wire family and metadata control word 7
// chooses Part I or Part II, so neither rows nor parameters cross from the host.

const PROFILE_CURRENT_C: u32 = 2u;
const SAMPLE_METADATA_PROFILE: u32 = 27u;
const SAMPLE_METADATA_PROFILE_MASK: u32 = 0xffu;
const METADATA_CORRECTION_PART: u32 = 7u;
const ROW_SET_WORDS: u32 = 512u;

const PARAM_LENGTH: u32 = 0u;
const PARAM_HEIGHT: u32 = 1u;
const PARAM_RANK: u32 = 2u;
const PARAM_NET: u32 = 3u;
const PARAM_BLOCKS: u32 = 4u;
const PARAM_ROW_DEGREE: u32 = 5u;
const PARAM_TAIL_BLOCK: u32 = 6u;
const PARAM_TAIL_LENGTH: u32 = 7u;
const PARAM_TAIL_HEIGHT: u32 = 8u;
const PARAM_TAIL_RANK: u32 = 9u;
const PARAM_TAIL_NET: u32 = 10u;
const PARAM_TAIL_ROW_DEGREE: u32 = 11u;
const PARAM_TAIL_ROW_BASE: u32 = 12u;
const PARAM_ADMISSION: u32 = 13u;
const PARAM_ROW_BASE: u32 = 14u;
const PARAM_WORDS: u32 = 16u;

@group(0) @binding(0) var<storage, read> sample: array<u32>;
@group(0) @binding(1) var<storage, read> metadata: array<u32>;
@group(0) @binding(2) var<storage, read_write> params: array<u32>;

@compute @workgroup_size(1)
fn main() {
    for (var word = 0u; word < PARAM_WORDS; word += 1u) {
        params[word] = 0u;
    }
    let part = metadata[METADATA_CORRECTION_PART];
    let part_two = part == 1u;
    let profile = sample[SAMPLE_METADATA_PROFILE] & SAMPLE_METADATA_PROFILE_MASK;
    let current = profile == PROFILE_CURRENT_C;
    let set = select(0u, 2u, current) + select(0u, 1u, part_two);
    params[PARAM_LENGTH] = select(6u, 38u, part_two);
    params[PARAM_HEIGHT] = select(3u, 19u, part_two);
    params[PARAM_RANK] = params[PARAM_HEIGHT];
    params[PARAM_NET] = params[PARAM_HEIGHT];
    params[PARAM_BLOCKS] = 1u;
    params[PARAM_ROW_DEGREE] = select(4u, 19u, part_two);
    params[PARAM_TAIL_BLOCK] = 1u;
    params[PARAM_ROW_BASE] = set * ROW_SET_WORDS;
    if part > 1u || profile > PROFILE_CURRENT_C {
        params[PARAM_ADMISSION] = 2u;
    }
}
