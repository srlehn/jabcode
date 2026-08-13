// Reduces resident pitch evidence into the four retry records consumed by
// descreen and print preparation. The resident control selects:
// valleys, positive peak values, winning lags, histogram/schedule reduction,
// then one selected retry record copied into the live pipeline controls.

const PITCH_STAGE_VALLEY: u32 = 0u;
const PITCH_STAGE_PEAK: u32 = 1u;
const PITCH_STAGE_LAG: u32 = 2u;
const PITCH_STAGE_SCHEDULE: u32 = 3u;
const PITCH_STAGE_PRINT: u32 = 4u;
const PITCH_STAGE_SELECT: u32 = 5u;

const PITCH_SAMPLE_LINES: u32 = 32u;
const SEED_BUCKETS: u32 = 1024u;
const SEEDS_PER_PIXEL: f32 = 4.0;
const PRINT_MIN_SEEDS: u32 = 100u;
const PRINT_BLUR_LEAD_RADIUS: u32 = 3u;

const CONTROL_ROW_VALLEY: u32 = 0u;
const CONTROL_COLUMN_VALLEY: u32 = 1u;
const CONTROL_ROW_PEAK: u32 = 2u;
const CONTROL_COLUMN_PEAK: u32 = 3u;
const CONTROL_ROW_LAG: u32 = 4u;
const CONTROL_COLUMN_LAG: u32 = 5u;
const CONTROL_MEDIAN_BUCKET: u32 = 6u;
const CONTROL_SELECTOR: u32 = 7u;
const CONTROL_STAGE: u32 = 8u;

const RETRY_RECORD_WORDS: u32 = 4u;
const RETRY_DESCREEN_FIRST: u32 = 0u;
const RETRY_DESCREEN_SECOND: u32 = 1u;
const RETRY_PRINT_FIRST: u32 = 2u;
const RETRY_PRINT_SECOND: u32 = 3u;
const RETRY_INDIRECT_CANVAS: u32 = 16u;
const RETRY_INDIRECT_BLOCKS: u32 = 19u;
const RETRY_INDIRECT_PACK: u32 = 22u;
const RETRY_INDIRECT_SCAN: u32 = 25u;
const RETRY_INDIRECT_CHAIN: u32 = 28u;

const BIN_WIDTH: u32 = 0u;
const BIN_HEIGHT: u32 = 1u;
const BIN_BLOCK_SIZE: u32 = 2u;
const BIN_BLOCKS_X: u32 = 3u;
const BIN_BLOCKS_Y: u32 = 4u;
const BIN_FLAGS: u32 = 5u;
const BIN_ROW_STRIDE: u32 = 9u;
const BIN_SCAN_CAPACITY: u32 = 10u;
const BIN_RETRY_ACTIVE: u32 = 11u;

const RETRY_MAX_SURVIVORS: u32 = 19u;
const PRINT_MAX_SURVIVORS: u32 = 2u;

const BLOCK_DIVISOR: u32 = 24u;
const BLOCK_MIN: u32 = 64u;
const BLOCK_MAX: u32 = 512u;

const SCAN_CHANNELS: u32 = 2u;
const CHAIN_CLASSIFY_CURRENT: u32 = 3264u;
const CHAIN_CLASSIFY_BSI: u32 = 1876u;
const CHAIN_CROSS_COLOR_BITS: u32 = 0u;
const CHAIN_COLOR_SOURCE: u32 = 2u;
const CHAIN_ROW_STRIDE_SHIFT: u32 = 8u;
const CHAIN_COMPACT_CAPACITY: u32 = 33268u;

@group(0) @binding(0) var<storage, read> acf: array<f32>;
@group(0) @binding(1) var<storage, read_write> seed_histogram: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> binarizer: array<u32>;
@group(0) @binding(3) var<storage, read_write> control: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> schedule: array<u32>;
@group(0) @binding(5) var<storage, read_write> descreen: array<u32>;
@group(0) @binding(6) var<storage, read_write> scan: array<u32>;
@group(0) @binding(7) var<storage, read_write> chain: array<u32>;
@group(0) @binding(8) var<storage, read> retry_control: array<u32>;
@group(0) @binding(9) var<storage, read_write> pass_indirect: array<u32>;

var<workgroup> prefix: array<u32, 1024>;

fn max_lag() -> u32 {
    return max(2u, min(binarizer[BIN_WIDTH], binarizer[BIN_HEIGHT]) / 8u);
}

fn acf_index(axis: u32, lag: u32) -> u32 {
    return axis * (max_lag() + 1u) + lag;
}

fn control_axis(base: u32, axis: u32) -> u32 {
    return base + axis;
}

fn radius_below_module(radius: u32, module_scale: f32) -> u32 {
    if radius < 1u || module_scale <= f32(2u * radius + 1u) {
        return 0u;
    }
    return radius;
}

fn cell_radius(pitch: u32) -> u32 {
    if pitch == 0u {
        return 0u;
    }
    return max(1u, pitch / 2u);
}

fn write_retry(slot: u32, active: bool, radius_x: u32, radius_y: u32, print_levels: bool) {
    let base = slot * RETRY_RECORD_WORDS;
    schedule[base] = select(0u, 1u, active);
    schedule[base + 1u] = radius_x;
    schedule[base + 2u] = radius_y;
    schedule[base + 3u] = select(0u, 1u, print_levels);
}

fn write_dispatch(offset: u32, x: u32, y: u32) {
    schedule[offset] = x;
    schedule[offset + 1u] = y;
    schedule[offset + 2u] = 1u;
}

fn reduce_schedule(local_index: u32, print_stage: bool) {
    for (var part = 0u; part < 4u; part++) {
        let bucket = local_index + part * 256u;
        prefix[bucket] = atomicLoad(&seed_histogram[bucket]);
    }
    workgroupBarrier();

    for (var offset = 1u; offset < SEED_BUCKETS; offset *= 2u) {
        var additions = array<u32, 4>(0u, 0u, 0u, 0u);
        for (var part = 0u; part < 4u; part++) {
            let bucket = local_index + part * 256u;
            if bucket >= offset {
                additions[part] = prefix[bucket - offset];
            }
        }
        workgroupBarrier();
        for (var part = 0u; part < 4u; part++) {
            let bucket = local_index + part * 256u;
            prefix[bucket] += additions[part];
        }
        workgroupBarrier();
    }

    let total = prefix[SEED_BUCKETS - 1u];
    if local_index == 0u {
        atomicStore(&control[CONTROL_MEDIAN_BUCKET], 0xffffffffu);
    }
    workgroupBarrier();
    if total > 0u {
        let target = total / 2u;
        for (var part = 0u; part < 4u; part++) {
            let bucket = local_index + part * 256u;
            if prefix[bucket] > target {
                atomicMin(&control[CONTROL_MEDIAN_BUCKET], bucket);
            }
        }
    }
    workgroupBarrier();

    // The descreen decision retains its input distribution because the host
    // print gate includes seeds from every descreen pass too. The later print
    // reduction consumes the complete distribution and begins the next epoch.
    if print_stage {
        for (var part = 0u; part < 4u; part++) {
            let bucket = local_index + part * 256u;
            atomicStore(&seed_histogram[bucket], 0u);
        }
    }

    if local_index != 0u {
        return;
    }
    let median_bucket = atomicLoad(&control[CONTROL_MEDIAN_BUCKET]);
    if total == 0u || median_bucket == 0xffffffffu {
        return;
    }
    let module_scale = (f32(median_bucket) + 0.5) / SEEDS_PER_PIXEL;
    if print_stage {
        let print_radius = max(1u, u32(floor(module_scale / 4.0 + 0.5)));
        let print_active = total >= PRINT_MIN_SEEDS;
        if print_radius < PRINT_BLUR_LEAD_RADIUS {
            write_retry(RETRY_PRINT_FIRST, print_active, 0u, 0u, true);
            write_retry(RETRY_PRINT_SECOND, print_active, print_radius, print_radius, true);
        } else {
            write_retry(RETRY_PRINT_FIRST, print_active, print_radius, print_radius, true);
            write_retry(RETRY_PRINT_SECOND, print_active, 0u, 0u, true);
        }
    } else {
        let row_lag = atomicLoad(&control[CONTROL_ROW_LAG]);
        let column_lag = atomicLoad(&control[CONTROL_COLUMN_LAG]);
        let pitch_x = select(row_lag, 0u, row_lag == 0xffffffffu);
        let pitch_y = select(column_lag, 0u, column_lag == 0xffffffffu);
        let radius_x = radius_below_module(cell_radius(pitch_x), module_scale);
        let radius_y = radius_below_module(cell_radius(pitch_y), module_scale);
        write_retry(
            RETRY_DESCREEN_FIRST,
            radius_x != 0u || radius_y != 0u,
            radius_x,
            radius_y,
            false,
        );
        let radius_x2 = radius_below_module(radius_x * 2u, module_scale);
        let radius_y2 = radius_below_module(radius_y * 2u, module_scale);
        write_retry(
            RETRY_DESCREEN_SECOND,
            radius_x2 != 0u || radius_y2 != 0u,
            radius_x2,
            radius_y2,
            false,
        );
    }
}

fn select_retry() {
    let selector = atomicLoad(&control[CONTROL_SELECTOR]);
    if selector > RETRY_PRINT_SECOND {
        return;
    }
    let base = selector * RETRY_RECORD_WORDS;
    var active = schedule[base] != 0u && binarizer[BIN_RETRY_ACTIVE] != 0u;
    let print_levels = schedule[base + 3u];
    if print_levels != 0u && retry_control[RETRY_MAX_SURVIVORS] > PRINT_MAX_SURVIVORS {
        active = false;
    }
    let width = binarizer[BIN_WIDTH];
    let height = binarizer[BIN_HEIGHT];
    let block_size = clamp(min(width, height) / BLOCK_DIVISOR, BLOCK_MIN, BLOCK_MAX);
    let row_stride = max(binarizer[BIN_ROW_STRIDE], 1u);
    let capacity = binarizer[BIN_SCAN_CAPACITY];

    descreen[0] = width;
    descreen[1] = height;
    descreen[2] = schedule[base + 1u];
    descreen[3] = schedule[base + 2u];

    binarizer[BIN_BLOCK_SIZE] = block_size;
    binarizer[BIN_BLOCKS_X] = (width + block_size - 1u) / block_size;
    binarizer[BIN_BLOCKS_Y] = (height + block_size - 1u) / block_size;
    binarizer[BIN_FLAGS] = print_levels << 1u;

    scan[0] = width;
    scan[1] = height;
    scan[2] = SCAN_CHANNELS;
    scan[3] = capacity;

    chain[0] = width;
    chain[1] = height;
    chain[2] = capacity;
    chain[3] = print_levels | CHAIN_COLOR_SOURCE | (row_stride << CHAIN_ROW_STRIDE_SHIFT);
    chain[4] = CHAIN_CLASSIFY_CURRENT;
    chain[5] = CHAIN_CLASSIFY_BSI;
    chain[6] = CHAIN_CROSS_COLOR_BITS;
    chain[7] = CHAIN_COMPACT_CAPACITY;

    let active_word = select(0u, 1u, active);
    binarizer[BIN_RETRY_ACTIVE] = active_word;
    pass_indirect[0] = active_word;
    pass_indirect[1] = 1u;
    pass_indirect[2] = 1u;
    if !active {
        write_dispatch(RETRY_INDIRECT_CANVAS, 0u, 1u);
        write_dispatch(RETRY_INDIRECT_BLOCKS, 0u, 1u);
        write_dispatch(RETRY_INDIRECT_PACK, 0u, 1u);
        write_dispatch(RETRY_INDIRECT_SCAN, 0u, 1u);
        write_dispatch(RETRY_INDIRECT_CHAIN, 0u, 1u);
        return;
    }
    write_dispatch(RETRY_INDIRECT_CANVAS, (width + 7u) / 8u, (height + 7u) / 8u);
    write_dispatch(
        RETRY_INDIRECT_BLOCKS,
        binarizer[BIN_BLOCKS_X],
        binarizer[BIN_BLOCKS_Y],
    );
    let packed_words = (width * height + 7u) / 8u;
    write_dispatch(RETRY_INDIRECT_PACK, (packed_words + 63u) / 64u, 1u);
    write_dispatch(RETRY_INDIRECT_SCAN, (height + 63u) / 64u, 1u);
    write_dispatch(RETRY_INDIRECT_CHAIN, (capacity + 63u) / 64u, 1u);
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let stage = atomicLoad(&control[CONTROL_STAGE]);
    if stage == PITCH_STAGE_SCHEDULE || stage == PITCH_STAGE_PRINT {
        reduce_schedule(local_id.x, stage == PITCH_STAGE_PRINT);
        return;
    }
    if stage == PITCH_STAGE_SELECT {
        if global_id.x == 0u {
            select_retry();
        }
        return;
    }

    let lags = max_lag() + 1u;
    if global_id.x >= 2u * lags {
        return;
    }
    let axis = global_id.x / lags;
    let lag = global_id.x % lags;
    if stage == PITCH_STAGE_VALLEY {
        if lag >= 1u && lag < max_lag() &&
            acf[acf_index(axis, lag)] < acf[acf_index(axis, lag + 1u)] {
            atomicMin(&control[control_axis(CONTROL_ROW_VALLEY, axis)], lag);
        }
        return;
    }

    let valley = atomicLoad(&control[control_axis(CONTROL_ROW_VALLEY, axis)]);
    if valley == 0xffffffffu || lag < valley || lag > max_lag() {
        return;
    }
    let value = acf[acf_index(axis, lag)];
    if stage == PITCH_STAGE_PEAK {
        if value > 0.0 {
            atomicMax(
                &control[control_axis(CONTROL_ROW_PEAK, axis)],
                bitcast<u32>(value),
            );
        }
        return;
    }
    let peak = atomicLoad(&control[control_axis(CONTROL_ROW_PEAK, axis)]);
    if peak != 0u && bitcast<u32>(value) == peak {
        atomicMin(&control[control_axis(CONTROL_ROW_LAG, axis)], lag);
    }
}
