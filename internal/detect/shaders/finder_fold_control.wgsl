// Publishes the fixed resident locate traversal's fold controls. The cursor is
// advanced by device work between assembly dispatches, so the host records one
// command sequence without uploading a different parameter block for each
// source.

const BIN_WIDTH: u32 = 0u;
const BIN_HEIGHT: u32 = 1u;
const BIN_FLAGS: u32 = 5u;

const ROW_CHANNEL: u32 = 1u;
const ROW_CAPACITY: u32 = 33268u;
const ROW_STRIDE: u32 = 13u;
const ROW_SUMMARY_WORDS: u32 = 7u;
const ROW_SUMMARY_COMPACTED: u32 = 0u;
const ROW_SUMMARY_OVERFLOW: u32 = 6u;

const DIRECTION_CAPACITY: u32 = 33268u;
const DIRECTION_STRIDE: u32 = 6u;
const DIRECTION_SUMMARY_WORDS: u32 = 7u;
const DIRECTION_SUMMARY_COMPACTED: u32 = 0u;
const DIRECTION_SUMMARY_REQUIRED: u32 = 6u;
const DIRECTION_RECORD_CAPACITY: u32 = 262144u;

const FAMILY_POOL_CAPACITY: u32 = 20958u;
const PATTERN_STOP: u32 = 499u;
const CONTEXTUAL_CAPACITY: u32 = 32768u;
const MIN_CONTEXTUAL_FOUND: u32 = 3u;

@group(0) @binding(0) var<storage, read> binarizer: array<u32>;
@group(0) @binding(1) var<storage, read_write> cursor: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> fold_params: array<u32>;
@group(0) @binding(3) var<storage, read_write> assembly_params: array<u32>;
@group(0) @binding(4) var<storage, read_write> family_pool_params: array<u32>;
@group(0) @binding(5) var<storage, read_write> group_params: array<u32>;
@group(0) @binding(6) var<storage, read_write> contextual_params: array<u32>;
@group(0) @binding(7) var<storage, read_write> corner_params: array<u32>;

fn write_assembly(
    base: u32,
    capacity: u32,
    stride: u32,
    append: u32,
    count_offset: u32,
    required_offset: u32,
    required_limit: u32,
) {
    assembly_params[0] = base;
    assembly_params[1] = capacity;
    assembly_params[2] = stride;
    assembly_params[3] = append;
    assembly_params[4] = 1u;
    assembly_params[5] = count_offset;
    assembly_params[6] = required_offset;
    assembly_params[7] = required_limit;
}

@compute @workgroup_size(1)
fn main() {
    let stage = atomicAdd(&cursor[0], 1u);

    // Stage two appends the column source to the row source assembled by stage
    // one. Every other stage begins a new fold and resets its running count.
    if stage != 2u {
        fold_params[0] = 0u;
        fold_params[1] = 0u;
    }
    fold_params[2] = (binarizer[BIN_FLAGS] & 2u) >> 1u;
    fold_params[3] = 0u;
    fold_params[4] = PATTERN_STOP;
    fold_params[5] = CONTEXTUAL_CAPACITY;
    fold_params[6] = 1u;
    fold_params[7] = 0u;

    if stage <= 1u {
        let summary = ROW_CHANNEL * ROW_SUMMARY_WORDS;
        write_assembly(
            ROW_CHANNEL * ROW_CAPACITY,
            ROW_CAPACITY,
            ROW_STRIDE,
            0u,
            summary + ROW_SUMMARY_COMPACTED,
            summary + ROW_SUMMARY_OVERFLOW,
            0u,
        );
    } else {
        // Stage two is the conditional column slot. Stages three through seven
        // are the five retry slots, so subtracting two gives the resident slot.
        let slot = stage - 2u;
        let summary = slot * DIRECTION_SUMMARY_WORDS;
        write_assembly(
            slot * DIRECTION_CAPACITY,
            DIRECTION_CAPACITY,
            DIRECTION_STRIDE,
            select(0u, 1u, stage == 2u),
            summary + DIRECTION_SUMMARY_COMPACTED,
            summary + DIRECTION_SUMMARY_REQUIRED,
            DIRECTION_RECORD_CAPACITY,
        );
    }

    family_pool_params[0] = FAMILY_POOL_CAPACITY;
    family_pool_params[1] = 0u;
    family_pool_params[2] = 0u;
    family_pool_params[3] = 0u;

    group_params[0] = CONTEXTUAL_CAPACITY;
    group_params[1] = 0u;
    group_params[2] = 6u;
    group_params[3] = 1u;

    contextual_params[0] = CONTEXTUAL_CAPACITY;
    contextual_params[1] = MIN_CONTEXTUAL_FOUND;
    contextual_params[2] = 0u;
    contextual_params[3] = 0u;

    corner_params[0] = max(binarizer[BIN_WIDTH], 1u);
    corner_params[1] = max(binarizer[BIN_HEIGHT], 1u);
    corner_params[2] = 0u;
    corner_params[3] = 0u;
}
