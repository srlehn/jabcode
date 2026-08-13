// Retains the same finder set the host would publish for the average-RGB retry,
// then gates that retry from the resident locate decision. The row selection is
// kept as the fallback, while a later located decision replaces it. The host
// records the fixed command graph; this control writes zero dispatches when an
// earlier pass has already located a quad.

const STAGE_CAPTURE_AVERAGE_ROW: u32 = 0u;
const STAGE_CAPTURE_SURVIVORS: u32 = 1u;
const STAGE_SELECT_AVERAGE: u32 = 2u;
const STAGE_SELECT_PITCH: u32 = 3u;
const STAGE_CAPTURE_AVERAGE_DECISION: u32 = 4u;
const STAGE_CAPTURE_PASS_RESULT: u32 = 5u;

const PITCH_SAMPLE_LINES: u32 = 32u;

const PAT_WORDS: u32 = 6u;
const PAT_X: u32 = 2u;
const PAT_Y: u32 = 3u;
const PAT_MODULE: u32 = 4u;
const PAT_FOUND: u32 = 5u;

const SELECTION_PATTERNS: u32 = 16u;
const FOLD_TOTAL: u32 = 0u;
const DECISION_HAVE: u32 = 0u;
const DECISION_DECLINED: u32 = 2u;
const DECISION_PATTERNS: u32 = 8u;
const DECISION_DIAGNOSTIC: u32 = 83u;
const DECISION_PASS_INPUT: u32 = 84u;
const DIAGNOSTIC_PASS_MASK: u32 = 0xffu;
const PREPARED_PASSES: u32 = 6u;
const DIAGNOSTIC_ADMITTED_SHIFT: u32 = 16u;
const DIAGNOSTIC_LOCATED_SHIFT: u32 = 24u;

const BIN_WIDTH: u32 = 0u;
const BIN_HEIGHT: u32 = 1u;
const BIN_BLOCK_SIZE: u32 = 2u;
const BIN_BLOCKS_X: u32 = 3u;
const BIN_BLOCKS_Y: u32 = 4u;
const BIN_FLAGS: u32 = 5u;
const BIN_ROW_STRIDE: u32 = 9u;
const BIN_SCAN_CAPACITY: u32 = 10u;
const BIN_RETRY_ACTIVE: u32 = 11u;

const SCAN_CHANNELS: u32 = 2u;
const CHAIN_CLASSIFY_CURRENT: u32 = 3264u;
const CHAIN_CLASSIFY_BSI: u32 = 1876u;
const CHAIN_CROSS_COLOR_BITS: u32 = 0u;
const CHAIN_COLOR_SOURCE: u32 = 2u;
const CHAIN_ROW_STRIDE_SHIFT: u32 = 8u;
const CHAIN_COMPACT_CAPACITY: u32 = 33268u;

const AVERAGE_WORDS: u32 = 18u;
const CONTROL_STAGE: u32 = 18u;
const CONTROL_MAX_SURVIVORS: u32 = 19u;
const INDIRECT_AVERAGE: u32 = 20u;
const INDIRECT_REDUCE: u32 = 23u;
const INDIRECT_CANVAS: u32 = 26u;
const INDIRECT_BLOCKS: u32 = 29u;
const INDIRECT_PACK: u32 = 32u;
const INDIRECT_SCAN: u32 = 35u;
const INDIRECT_CHAIN: u32 = 38u;
const INDIRECT_PITCH_SAMPLES: u32 = 41u;
const INDIRECT_PITCH_ONE: u32 = 44u;
const INDIRECT_PITCH_CENTER: u32 = 47u;
const INDIRECT_PITCH_ACF: u32 = 50u;
const INDIRECT_PITCH_SELECT: u32 = 53u;

@group(0) @binding(0) var<storage, read> selection: array<u32>;
@group(0) @binding(1) var<storage, read> fold: array<u32>;
@group(0) @binding(2) var<storage, read_write> decision: array<u32>;
@group(0) @binding(3) var<storage, read_write> binarizer: array<u32>;
@group(0) @binding(4) var<storage, read_write> scan: array<u32>;
@group(0) @binding(5) var<storage, read_write> chain: array<u32>;
@group(0) @binding(6) var<storage, read_write> average: array<u32>;
@group(0) @binding(7) var<storage, read_write> pass_indirect: array<u32>;

fn write_dispatch(offset: u32, x: u32, y: u32) {
    average[offset] = x;
    average[offset + 1u] = y;
    average[offset + 2u] = 1u;
}

fn capture_survivors() {
    if binarizer[BIN_RETRY_ACTIVE] != 0u {
        average[CONTROL_MAX_SURVIVORS] = max(
            average[CONTROL_MAX_SURVIVORS],
            fold[FOLD_TOTAL],
        );
    }
}

fn capture_pass_result() {
    let pass = decision[DECISION_PASS_INPUT] & DIAGNOSTIC_PASS_MASK;
    let admitted = pass == 0u || binarizer[BIN_RETRY_ACTIVE] != 0u;
    if !admitted || pass >= PREPARED_PASSES {
        return;
    }
    decision[DECISION_DIAGNOSTIC] |= 1u << (DIAGNOSTIC_ADMITTED_SHIFT + pass);
    let retained_pass = decision[DECISION_DIAGNOSTIC] & DIAGNOSTIC_PASS_MASK;
    if decision[DECISION_HAVE] != 0u && retained_pass == pass {
        decision[DECISION_DIAGNOSTIC] |= 1u << (DIAGNOSTIC_LOCATED_SHIFT + pass);
    }
}

fn capture_average(pattern_base: u32, from_decision: bool) {
    average[0] = binarizer[BIN_WIDTH];
    average[1] = binarizer[BIN_HEIGHT];
    for (var slot = 0u; slot < 4u; slot += 1u) {
        let output = 2u + slot * 4u;
        let pattern = pattern_base + slot * PAT_WORDS;
        average[output] = 0u;
        average[output + 1u] = 0u;
        average[output + 2u] = 0u;
        average[output + 3u] = 0u;
        let found = select(
            selection[pattern + PAT_FOUND],
            decision[pattern + PAT_FOUND],
            from_decision,
        );
        if found == 0u {
            continue;
        }
        let width = i32(binarizer[BIN_WIDTH]);
        let height = i32(binarizer[BIN_HEIGHT]);
        let x_word = select(
            selection[pattern + PAT_X],
            decision[pattern + PAT_X],
            from_decision,
        );
        let y_word = select(
            selection[pattern + PAT_Y],
            decision[pattern + PAT_Y],
            from_decision,
        );
        let module_word = select(
            selection[pattern + PAT_MODULE],
            decision[pattern + PAT_MODULE],
            from_decision,
        );
        let x = bitcast<f32>(x_word);
        let y = bitcast<f32>(y_word);
        let radius = bitcast<f32>(module_word) * 4.0;
        average[output] = u32(max(i32(x - radius), 0));
        average[output + 1u] = u32(max(i32(y - radius), 0));
        average[output + 2u] = u32(min(i32(x + radius), width - 1));
        average[output + 3u] = u32(min(i32(y + radius), height - 1));
    }
}

fn select_average() {
    let active = decision[DECISION_HAVE] == 0u &&
        decision[DECISION_DECLINED] == 0u;
    let width = binarizer[BIN_WIDTH];
    let height = binarizer[BIN_HEIGHT];
    let capacity = binarizer[BIN_SCAN_CAPACITY];
    let row_stride = max(binarizer[BIN_ROW_STRIDE], 1u);
    binarizer[BIN_RETRY_ACTIVE] = select(0u, 1u, active);
    pass_indirect[0] = select(0u, 1u, active);
    pass_indirect[1] = 1u;
    pass_indirect[2] = 1u;

    binarizer[BIN_BLOCK_SIZE] = 1u;
    binarizer[BIN_BLOCKS_X] = 1u;
    binarizer[BIN_BLOCKS_Y] = 1u;
    binarizer[BIN_FLAGS] = 1u;

    scan[0] = width;
    scan[1] = height;
    scan[2] = SCAN_CHANNELS;
    scan[3] = capacity;

    chain[0] = width;
    chain[1] = height;
    chain[2] = capacity;
    chain[3] = CHAIN_COLOR_SOURCE | (row_stride << CHAIN_ROW_STRIDE_SHIFT);
    chain[4] = CHAIN_CLASSIFY_CURRENT;
    chain[5] = CHAIN_CLASSIFY_BSI;
    chain[6] = CHAIN_CROSS_COLOR_BITS;
    chain[7] = CHAIN_COMPACT_CAPACITY;

    if !active {
        write_dispatch(INDIRECT_AVERAGE, 0u, 1u);
        write_dispatch(INDIRECT_REDUCE, 0u, 1u);
        write_dispatch(INDIRECT_CANVAS, 0u, 1u);
        write_dispatch(INDIRECT_BLOCKS, 0u, 1u);
        write_dispatch(INDIRECT_PACK, 0u, 1u);
        write_dispatch(INDIRECT_SCAN, 0u, 1u);
        write_dispatch(INDIRECT_CHAIN, 0u, 1u);
        return;
    }
    write_dispatch(INDIRECT_AVERAGE, 4u, 1u);
    write_dispatch(INDIRECT_REDUCE, 1u, 1u);
    write_dispatch(INDIRECT_CANVAS, (width + 7u) / 8u, (height + 7u) / 8u);
    write_dispatch(INDIRECT_BLOCKS, 0u, 1u);
    let packed_words = (width * height + 7u) / 8u;
    write_dispatch(INDIRECT_PACK, (packed_words + 63u) / 64u, 1u);
    write_dispatch(INDIRECT_SCAN, (height + 63u) / 64u, 1u);
    write_dispatch(INDIRECT_CHAIN, (capacity + 63u) / 64u, 1u);
}

fn select_pitch() {
    let active = decision[DECISION_HAVE] == 0u &&
        decision[DECISION_DECLINED] == 0u;
    let width = binarizer[BIN_WIDTH];
    let height = binarizer[BIN_HEIGHT];
    let row_count = min(PITCH_SAMPLE_LINES, height);
    let column_count = min(PITCH_SAMPLE_LINES, width);
    let samples = row_count * width + column_count * height;
    let max_lag = max(2u, min(width, height) / 8u);
    binarizer[BIN_RETRY_ACTIVE] = select(0u, 1u, active);
    pass_indirect[0] = select(0u, 1u, active);
    pass_indirect[1] = 1u;
    pass_indirect[2] = 1u;
    let enabled = select(0u, 1u, active);
    write_dispatch(INDIRECT_PITCH_SAMPLES, enabled * ((samples + 63u) / 64u), 1u);
    write_dispatch(INDIRECT_PITCH_ONE, enabled, 1u);
    write_dispatch(INDIRECT_PITCH_CENTER, enabled * ((samples + 63u) / 64u), 1u);
    write_dispatch(INDIRECT_PITCH_ACF, enabled * ((2u * (max_lag + 1u) + 63u) / 64u), 1u);
    write_dispatch(INDIRECT_PITCH_SELECT, enabled * ((2u * (max_lag + 1u) + 255u) / 256u), 1u);
}

@compute @workgroup_size(1)
fn main() {
    let stage = average[CONTROL_STAGE];
    if stage == STAGE_CAPTURE_AVERAGE_ROW {
        capture_average(SELECTION_PATTERNS, false);
        average[CONTROL_MAX_SURVIVORS] = fold[FOLD_TOTAL];
        return;
    }
    if stage == STAGE_CAPTURE_AVERAGE_DECISION {
        if decision[DECISION_HAVE] != 0u {
            capture_average(DECISION_PATTERNS, true);
        }
        return;
    }
    if stage == STAGE_CAPTURE_SURVIVORS {
        capture_survivors();
        return;
    }
    if stage == STAGE_CAPTURE_PASS_RESULT {
        capture_pass_result();
        return;
    }
    if stage == STAGE_SELECT_AVERAGE {
        select_average();
        return;
    }
    select_pitch();
}
