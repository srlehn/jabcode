// Builds the one decoder parity-check matrix selected by resident payload
// control. The Gallager source stays sparse, while only the elimination copy is
// expanded into packed words. This keeps retained storage proportional to one
// selected shape instead of precomputing every colour and ECC candidate.
//
// The source permutation is a Fisher-Yates walk and is inherently serial. One
// lane runs that walk. Pivot search and row elimination own the expensive dense
// work, so they are spread across the whole workgroup: words search for the
// lowest pivot together and lanes own target rows during each XOR step.
//
// MATRIX_SLOT is prepended by the host when compiling the two fixed pipelines.
// Slot zero builds the regular sub-block and slot one the optional tail. They
// dispatch sequentially and reuse one workspace, so the uncommon tail does not
// double the largest retained matrix allocation.

const WORKGROUP: u32 = 256u;
const MAX_SUB: u32 = 2816u;
const MAX_ROW_DEGREE: u32 = 16u;
const MAX_STRIDE: u32 = (MAX_SUB + 31u) / 32u;
const DENSE_WORDS: u32 = MAX_SUB * MAX_STRIDE;
const SOURCE_WORDS: u32 = MAX_SUB * MAX_ROW_DEGREE;

const DENSE_BASE: u32 = 0u;
const SOURCE_BASE: u32 = DENSE_BASE + DENSE_WORDS;
const ARRANGEMENT_BASE: u32 = SOURCE_BASE + SOURCE_WORDS;
const PROCESSED_BASE: u32 = ARRANGEMENT_BASE + MAX_SUB;
const ZERO_LINES_BASE: u32 = PROCESSED_BASE + MAX_STRIDE;
const SWAPS_BASE: u32 = ZERO_LINES_BASE + MAX_SUB;
const ROW_COUNTS_BASE: u32 = SWAPS_BASE + 2u * MAX_SUB;
const PERM_BASE: u32 = ROW_COUNTS_BASE + MAX_SUB;
const SCRATCH_WORDS: u32 = PERM_BASE + MAX_SUB;

const ROW_SET_WORDS: u32 = MAX_SUB * MAX_ROW_DEGREE;
const MESSAGE_SEED: u32 = 785465u;
const GENERATOR_LCG: u32 = 1u;

// The C-family multiplier, 0x5851F42D4C957F2D, split into halves.
const LCG_MULTIPLIER_HIGH: u32 = 0x5851F42Du;
const LCG_MULTIPLIER_LOW: u32 = 0x4C957F2Du;

const PAYLOAD_PARAM_GENERATOR: u32 = 9u;
const PAYLOAD_PARAM_ADMISSION: u32 = 1714u;
const PAYLOAD_PARAM_WC: u32 = 1717u;
const PAYLOAD_PARAM_WR: u32 = 1718u;

const LDPC_PARAM_LENGTH: u32 = 0u;
const LDPC_PARAM_HEIGHT: u32 = 1u;
const LDPC_PARAM_RANK: u32 = 2u;
const LDPC_PARAM_ROW_DEGREE: u32 = 5u;
const LDPC_PARAM_TAIL_LENGTH: u32 = 7u;
const LDPC_PARAM_TAIL_HEIGHT: u32 = 8u;
const LDPC_PARAM_TAIL_RANK: u32 = 9u;
const LDPC_PARAM_TAIL_ROW_DEGREE: u32 = 11u;
const LDPC_PARAM_TAIL_ROW_BASE: u32 = 12u;
const LDPC_PARAM_ADMISSION: u32 = 13u;

const CACHE_WORDS: u32 = 8u;
const CACHE_VALID: u32 = 0u;
const CACHE_LENGTH: u32 = 1u;
const CACHE_WC: u32 = 2u;
const CACHE_WR: u32 = 3u;
const CACHE_GENERATOR: u32 = 4u;
const CACHE_HEIGHT: u32 = 5u;
const CACHE_RANK: u32 = 6u;

@group(0) @binding(0) var<storage, read> payload_params: array<u32>;
@group(0) @binding(1) var<storage, read_write> ldpc_params: array<u32>;
@group(0) @binding(2) var<storage, read_write> rows_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<u32>;
@group(0) @binding(4) var<storage, read_write> cache: array<u32>;

var<workgroup> pivot_min: atomic<u32>;
var<workgroup> cache_hit: u32;
var<workgroup> failed: u32;
var<workgroup> zero_count: u32;
var<workgroup> swap_count: u32;
var<workgroup> matrix_rank: u32;

fn scratch_load(at: u32) -> u32 {
    return scratch[at];
}

fn scratch_store(at: u32, value: u32) {
    scratch[at] = value;
}

fn processed(column: u32) -> bool {
    let word = scratch_load(PROCESSED_BASE + column / 32u);
    return ((word >> (column % 32u)) & 1u) != 0u;
}

fn set_processed(column: u32, value: bool) {
    let at = PROCESSED_BASE + column / 32u;
    let mask = 1u << (column % 32u);
    let old = scratch_load(at);
    scratch_store(at, select(old & ~mask, old | mask, value));
}

fn matrix_length() -> u32 {
    return select(ldpc_params[LDPC_PARAM_LENGTH], ldpc_params[LDPC_PARAM_TAIL_LENGTH], MATRIX_SLOT == 1u);
}

fn row_base() -> u32 {
    return select(0u, ROW_SET_WORDS, MATRIX_SLOT == 1u);
}

fn publish_control(height: u32, rank: u32) {
    if MATRIX_SLOT == 0u {
        ldpc_params[LDPC_PARAM_HEIGHT] = height;
        ldpc_params[LDPC_PARAM_RANK] = rank;
        ldpc_params[LDPC_PARAM_ROW_DEGREE] = payload_params[PAYLOAD_PARAM_WR];
        return;
    }
    ldpc_params[LDPC_PARAM_TAIL_HEIGHT] = height;
    ldpc_params[LDPC_PARAM_TAIL_RANK] = rank;
    ldpc_params[LDPC_PARAM_TAIL_ROW_DEGREE] = payload_params[PAYLOAD_PARAM_WR];
    ldpc_params[LDPC_PARAM_TAIL_ROW_BASE] = ROW_SET_WORDS;
}

fn reject() {
    ldpc_params[LDPC_PARAM_ADMISSION] = 2u;
}

fn matrix_barrier() {
    storageBarrier();
    workgroupBarrier();
}

// wide_product is a 32 by 32 bit multiply kept whole as (high, low).
fn wide_product(a: u32, b: u32) -> vec2<u32> {
    let a_low = a & 0xffffu;
    let a_high = a >> 16u;
    let b_low = b & 0xffffu;
    let b_high = b >> 16u;
    let low_low = a_low * b_low;
    let cross_a = a_high * b_low;
    let cross_b = a_low * b_high;
    let middle = (low_low >> 16u) + (cross_a & 0xffffu) + (cross_b & 0xffffu);
    return vec2<u32>(
        a_high * b_high + (cross_a >> 16u) + (cross_b >> 16u) + (middle >> 16u),
        (low_low & 0xffffu) | (middle << 16u),
    );
}

fn lcg_step(state: vec2<u32>) -> vec2<u32> {
    var product = wide_product(state.y, LCG_MULTIPLIER_LOW);
    product.x += state.y * LCG_MULTIPLIER_HIGH + state.x * LCG_MULTIPLIER_LOW;
    let low = product.y + 1u;
    if low < product.y {
        product.x += 1u;
    }
    return vec2<u32>(product.x, low);
}

fn temper(value: u32) -> u32 {
    var x = value;
    x ^= x >> 11u;
    x ^= (x << 7u) & 0x9D2C5680u;
    x ^= (x << 15u) & 0xEFC60000u;
    x ^= x >> 18u;
    return x;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
    if payload_params[PAYLOAD_PARAM_ADMISSION] != 0u {
        return;
    }
    let lane = local.x;
    let length = matrix_length();
    let wc = payload_params[PAYLOAD_PARAM_WC];
    let wr = payload_params[PAYLOAD_PARAM_WR];
    let generator = payload_params[PAYLOAD_PARAM_GENERATOR];
    let cache_base = MATRIX_SLOT * CACHE_WORDS;

    if lane == 0u {
        cache_hit = 0u;
        failed = 0u;
        if MATRIX_SLOT == 1u && length == 0u {
            publish_control(0u, 0u);
            cache_hit = 1u;
        } else if cache[cache_base + CACHE_VALID] != 0u &&
            cache[cache_base + CACHE_LENGTH] == length &&
            cache[cache_base + CACHE_WC] == wc &&
            cache[cache_base + CACHE_WR] == wr &&
            cache[cache_base + CACHE_GENERATOR] == generator {
            publish_control(
                cache[cache_base + CACHE_HEIGHT],
                cache[cache_base + CACHE_RANK],
            );
            cache_hit = 1u;
        } else {
            cache[cache_base + CACHE_VALID] = 0u;
        }
        if length == 0u || length > MAX_SUB || wc < 3u || wc >= wr ||
            wr > 11u || wr > MAX_ROW_DEGREE {
            failed = 1u;
        }
    }
    matrix_barrier();
    if cache_hit != 0u {
        return;
    }
    if failed != 0u {
        if lane == 0u {
            reject();
        }
        return;
    }

    for (var at = lane; at < SCRATCH_WORDS; at += WORKGROUP) {
        scratch_store(at, 0u);
    }
    matrix_barrier();

    let block_rows = length / wr;
    let height = block_rows * wc;
    let stride = (length + 31u) / 32u;
    if lane == 0u {
        if height == 0u || height > MAX_SUB || stride > MAX_STRIDE {
            failed = 1u;
        } else {
            for (var row = 0u; row < block_rows; row += 1u) {
                for (var slot = 0u; slot < wr; slot += 1u) {
                    scratch_store(SOURCE_BASE + row * wr + slot, row * wr + slot);
                }
            }
            for (var at = 0u; at < length; at += 1u) {
                scratch_store(PERM_BASE + at, at);
            }

            let use_lcg = generator == GENERATOR_LCG;
            var iso_state = MESSAGE_SEED;
            var lcg_state = vec2<u32>(0u, MESSAGE_SEED);
            for (var copy = 1u; copy < wc; copy += 1u) {
                let dst = copy * block_rows;
                for (var column = 0u; column < length; column += 1u) {
                    let remaining = length - column;
                    var position = 0u;
                    if use_lcg {
                        lcg_state = lcg_step(lcg_state);
                        position = u32(f32(temper(lcg_state.x)) / 4294967296.0 * f32(remaining));
                    } else {
                        iso_state = iso_state * 1103515245u + 12345u;
                        let value = (iso_state / 65536u) % 32768u;
                        position = u32(f32(value) / 32768.0 * f32(remaining));
                    }
                    let source_column = scratch_load(PERM_BASE + position);
                    let row = dst + source_column / wr;
                    let count_at = ROW_COUNTS_BASE + row;
                    let slot = scratch_load(count_at);
                    if slot >= wr {
                        failed = 1u;
                    } else {
                        scratch_store(SOURCE_BASE + row * wr + slot, column);
                        scratch_store(count_at, slot + 1u);
                    }
                    let last = length - 1u - column;
                    let held = scratch_load(PERM_BASE + last);
                    scratch_store(PERM_BASE + last, scratch_load(PERM_BASE + position));
                    scratch_store(PERM_BASE + position, held);
                }
            }
        }
    }
    matrix_barrier();
    if failed != 0u {
        if lane == 0u {
            reject();
        }
        return;
    }

    // One lane owns a whole source row, so building its packed words needs no
    // atomics between lanes even when several set columns share one word.
    for (var row = lane; row < height; row += WORKGROUP) {
        for (var word = 0u; word < stride; word += 1u) {
            var value = 0u;
            for (var slot = 0u; slot < wr; slot += 1u) {
                let column = scratch_load(SOURCE_BASE + row * wr + slot);
                if column / 32u == word {
                    value |= 1u << (column % 32u);
                }
            }
            scratch_store(DENSE_BASE + row * stride + word, value);
        }
    }
    if lane == 0u {
        zero_count = 0u;
        swap_count = 0u;
    }
    matrix_barrier();

    // Rows are processed in dependency order, but the lowest pivot search and
    // every target-row XOR use all lanes. The matrix never leaves storage and
    // no dispatch boundary is needed per pivot.
    for (var row = 0u; row < height; row += 1u) {
        if lane == 0u {
            atomicStore(&pivot_min, 0xFFFFFFFFu);
        }
        matrix_barrier();
        for (var word = lane; word < stride; word += WORKGROUP) {
            let value = scratch_load(DENSE_BASE + row * stride + word);
            if value != 0u {
                atomicMin(&pivot_min, word * 32u + firstTrailingBit(value));
            }
        }
        matrix_barrier();
        let pivot = atomicLoad(&pivot_min);
        if lane == 0u {
            if pivot == 0xFFFFFFFFu {
                scratch_store(ZERO_LINES_BASE + zero_count, row);
                zero_count += 1u;
            } else {
                set_processed(pivot, true);
                scratch_store(ARRANGEMENT_BASE + pivot, row);
                if pivot >= height {
                    scratch_store(SWAPS_BASE + 2u * swap_count, pivot);
                    scratch_store(SWAPS_BASE + 2u * swap_count + 1u, 0u);
                    swap_count += 1u;
                }
            }
        }
        matrix_barrier();
        if pivot != 0xFFFFFFFFu {
            for (var target = lane; target < height; target += WORKGROUP) {
                if target == row {
                    continue;
                }
                let pivot_word = scratch_load(DENSE_BASE + target * stride + pivot / 32u);
                if ((pivot_word >> (pivot % 32u)) & 1u) == 0u {
                    continue;
                }
                for (var word = 0u; word < stride; word += 1u) {
                    let at = DENSE_BASE + target * stride + word;
                    scratch_store(at, scratch_load(at) ^
                        scratch_load(DENSE_BASE + row * stride + word));
                }
            }
        }
        matrix_barrier();
    }

    if lane == 0u {
        let rank = height - zero_count;
        matrix_rank = rank;
        var relocated = 0u;
        for (var column = rank; column < height; column += 1u) {
            let arranged = scratch_load(ARRANGEMENT_BASE + column);
            if arranged > 0u {
                for (var free = 0u; free < height; free += 1u) {
                    if !processed(free) {
                        scratch_store(ARRANGEMENT_BASE + free, arranged);
                        set_processed(free, true);
                        set_processed(column, false);
                        scratch_store(SWAPS_BASE + 2u * swap_count, column);
                        scratch_store(SWAPS_BASE + 2u * swap_count + 1u, free);
                        swap_count += 1u;
                        scratch_store(ARRANGEMENT_BASE + column, free);
                        relocated += 1u;
                        break;
                    }
                }
            }
        }

        var paired = 0u;
        let original_swaps = swap_count - relocated;
        for (var column = 0u; column < height && paired < original_swaps; column += 1u) {
            if !processed(column) {
                let source_column = scratch_load(SWAPS_BASE + 2u * paired);
                scratch_store(ARRANGEMENT_BASE + column,
                    scratch_load(ARRANGEMENT_BASE + source_column));
                set_processed(column, true);
                scratch_store(SWAPS_BASE + 2u * paired + 1u, column);
                paired += 1u;
            }
        }

        var zero = 0u;
        for (var column = 0u; column < height; column += 1u) {
            if !processed(column) {
                scratch_store(ARRANGEMENT_BASE + column,
                    scratch_load(ZERO_LINES_BASE + zero));
                zero += 1u;
            }
        }
    }
    matrix_barrier();

    // Rearrange the untouched sparse source and apply the recorded column
    // swaps. Sorting each small row reproduces the ascending adjacency the hard
    // and soft correctors share without ever materializing a second dense copy.
    let output_base = row_base();
    for (var row = lane; row < height; row += WORKGROUP) {
        var columns: array<u32, 16>;
        let source_row = scratch_load(ARRANGEMENT_BASE + row);
        for (var slot = 0u; slot < wr; slot += 1u) {
            var column = scratch_load(SOURCE_BASE + source_row * wr + slot);
            for (var swap = 0u; swap < swap_count; swap += 1u) {
                let from = scratch_load(SWAPS_BASE + 2u * swap);
                let to = scratch_load(SWAPS_BASE + 2u * swap + 1u);
                if column == from {
                    column = to;
                } else if column == to {
                    column = from;
                }
            }
            columns[slot] = column;
        }
        for (var slot = 1u; slot < wr; slot += 1u) {
            let held = columns[slot];
            var insert = slot;
            while insert > 0u && columns[insert - 1u] > held {
                columns[insert] = columns[insert - 1u];
                insert -= 1u;
            }
            columns[insert] = held;
        }
        for (var slot = 0u; slot < wr; slot += 1u) {
            rows_out[output_base + row * wr + slot] = columns[slot];
        }
    }
    matrix_barrier();

    if lane == 0u {
        publish_control(height, matrix_rank);
        cache[cache_base + CACHE_LENGTH] = length;
        cache[cache_base + CACHE_WC] = wc;
        cache[cache_base + CACHE_WR] = wr;
        cache[cache_base + CACHE_GENERATOR] = generator;
        cache[cache_base + CACHE_HEIGHT] = height;
        cache[cache_base + CACHE_RANK] = matrix_rank;
        cache[cache_base + CACHE_VALID] = 1u;
    }
}
