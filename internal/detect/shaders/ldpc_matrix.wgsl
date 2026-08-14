// Builds the one decoder parity-check matrix selected by resident payload
// control, from the sparse Gallager source and a precomputed pivot transcript.
//
// MATRIX_SLOT is prepended by the host when compiling the pipelines. Slot zero
// builds the regular sub-block and slot one the optional tail; they dispatch
// sequentially and reuse one workspace, so the uncommon tail does not double the
// largest retained allocation.
//
// There is no elimination here. Systematic reduction discovers exactly one
// number per parity row - the column that row pivots at - and everything after
// it is a deterministic function of that sequence: the rank, the dependent rows,
// the arrangement, and both kinds of column swap. The sequence is precomputed
// for the complete legal key space and arrives in the catalog binding, so this
// kernel replays it instead of rediscovering it. That removes the dense copy of
// the matrix, the sequential pivot loop, and the panel and apply dispatches
// along with their barriers.
//
// The leading Gallager block is not stored. Its rows hold wr consecutive ones
// and have pairwise disjoint support, so none is modified before the sweep
// reaches it and each pivots at its own first column.

// The workgroup width the rest of the resident stage already requires, so this
// kernel adds no device capability of its own. Every stage strides its work over
// whatever width is declared.
const WORKGROUP: u32 = 256u;

const MAX_SUB: u32 = 2816u;
const MAX_ROW_DEGREE: u32 = 16u;
const MAX_STRIDE: u32 = (MAX_SUB + 31u) / 32u;
const SOURCE_WORDS: u32 = MAX_SUB * MAX_ROW_DEGREE;

const SOURCE_BASE: u32 = 0u;
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

// Catalog layout. The two generator catalogs share one binding behind a short
// prefix of base words, so selecting a generator is an index rather than a
// second buffer and a second binding.
const CATALOG_MAGIC: u32 = 0x4A424C50u;
const CATALOG_FORMAT: u32 = 1u;
const CATALOG_PREFIX_WORDS: u32 = 4u;
const CATALOG_HEADER_WORDS: u32 = 4u;
const CATALOG_GENERATORS: u32 = 2u;
const PIVOT_BITS: u32 = 12u;
const PIVOT_NONE: u32 = 0xFFFu;
const MIN_COL_WEIGHT: u32 = 3u;
const MIN_ROW_WEIGHT: u32 = 4u;

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
const LDPC_PARAM_ROW_BASE: u32 = 14u;

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
@group(0) @binding(5) var<storage, read> catalog: array<u32>;

var<workgroup> live: u32;
var<workgroup> failed: u32;
var<workgroup> matrix_rank: u32;
var<workgroup> swap_count: u32;

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
        ldpc_params[LDPC_PARAM_ROW_BASE] = 0u;
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

// catalog_base is where one generator's catalog starts, or zero when this build
// does not carry it. The prefix is checked here rather than trusted, so a
// catalog regenerated under a changed format fails closed instead of producing
// a wrong arrangement.
fn catalog_base(generator: u32) -> u32 {
    if arrayLength(&catalog) < CATALOG_PREFIX_WORDS ||
        catalog[0] != CATALOG_MAGIC || catalog[1] != CATALOG_FORMAT ||
        generator >= CATALOG_GENERATORS {
        return 0u;
    }
    let base = catalog[2u + generator];
    if base < CATALOG_PREFIX_WORDS || base + CATALOG_HEADER_WORDS > arrayLength(&catalog) ||
        catalog[base] != CATALOG_MAGIC || catalog[base + 1u] != CATALOG_FORMAT {
        return 0u;
    }
    return base;
}

// catalog_slot numbers a key inside one generator's directory. Capacities are
// every multiple of wr up to MAX_SUB and column weights run from MIN_COL_WEIGHT
// to wr-1, so the space is dense and the slot is a perfect hash. The base loop
// runs at most seven times and avoids indexing a constant array with a runtime
// value.
fn catalog_slot(wc: u32, wr: u32, length: u32) -> u32 {
    var base = 0u;
    for (var w = MIN_ROW_WEIGHT; w < wr; w += 1u) {
        base += (w - MIN_COL_WEIGHT) * (MAX_SUB / w);
    }
    return base + (wc - MIN_COL_WEIGHT) * (MAX_SUB / wr) + length / wr - 1u;
}

// catalog_pivot reads one twelve-bit field of the packed record area. The stream
// is filled least-significant bits first, so a field crossing a word boundary
// continues in the low bits of the next word.
fn catalog_pivot(record_base: u32, index: u32) -> u32 {
    let bit = index * PIVOT_BITS;
    let at = record_base + bit / 32u;
    let shift = bit % 32u;
    var value = catalog[at] >> shift;
    if shift > 32u - PIVOT_BITS {
        value |= catalog[at + 1u] << (32u - shift);
    }
    return value & (1u << PIVOT_BITS) - 1u;
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

// build_source lays down the sparse Gallager rows. The permutation walk is a
// Fisher-Yates and inherently serial, so one lane runs it; it is the only serial
// stage left and is bounded by wc times the capacity.
fn build_source(length: u32, wc: u32, wr: u32, generator: u32) {
    let block_rows = length / wr;
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

// replay_transcript reproduces what the forward elimination would have recorded:
// which column each row claims, which rows are dependent, and which pivots
// landed outside the parity-check region. Order matters for the dependent-row
// and swap lists, so this stays on one lane; it is a single pass over the rows.
fn replay_transcript(length: u32, wc: u32, wr: u32, record_base: u32, start: u32) -> u32 {
    let block_rows = length / wr;
    let height = block_rows * wc;
    var zero_count = 0u;
    swap_count = 0u;
    for (var row = 0u; row < height; row += 1u) {
        var pivot = row * wr;
        if row >= block_rows {
            pivot = catalog_pivot(record_base, start + row - block_rows);
            if pivot == PIVOT_NONE {
                scratch_store(ZERO_LINES_BASE + zero_count, row);
                zero_count += 1u;
                continue;
            }
        }
        if pivot >= length {
            failed = 1u;
            return 0u;
        }
        set_processed(pivot, true);
        scratch_store(ARRANGEMENT_BASE + pivot, row);
        if pivot >= height {
            scratch_store(SWAPS_BASE + 2u * swap_count, pivot);
            scratch_store(SWAPS_BASE + 2u * swap_count + 1u, 0u);
            swap_count += 1u;
        }
    }
    return zero_count;
}

// resolve_arrangement finishes the column arrangement: pivots that landed past
// the rank move back into the identity region, the out-of-region pivots pair
// with free columns, and the remaining columns take the dependent rows.
fn resolve_arrangement(height: u32, rank: u32, zero_count: u32) {
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
    let base = catalog_base(generator);

    if lane == 0u {
        live = 1u;
        failed = 0u;
        if MATRIX_SLOT == 1u && length == 0u {
            publish_control(0u, 0u);
            live = 0u;
        } else if cache[cache_base + CACHE_VALID] != 0u &&
            cache[cache_base + CACHE_LENGTH] == length &&
            cache[cache_base + CACHE_WC] == wc &&
            cache[cache_base + CACHE_WR] == wr &&
            cache[cache_base + CACHE_GENERATOR] == generator {
            publish_control(
                cache[cache_base + CACHE_HEIGHT],
                cache[cache_base + CACHE_RANK],
            );
            live = 0u;
        } else {
            cache[cache_base + CACHE_VALID] = 0u;
        }
        // A resident key that already answers the request wins over the shape
        // check: the answer was admitted when it was built, and re-rejecting it
        // here would retract it.
        //
        // A key outside the catalog fails closed. The catalog covers the whole
        // legal space, so a miss is a wiring defect or an unsupported build
        // rather than an ordinary outcome, and answering it by eliminating on
        // the device would hide exactly the defect worth seeing.
        if length == 0u || length > MAX_SUB || wc < MIN_COL_WEIGHT || wc >= wr ||
            wr > 11u || wr > MAX_ROW_DEGREE || wr < MIN_ROW_WEIGHT ||
            length % wr != 0u || base == 0u {
            if live != 0u {
                reject();
            }
            live = 0u;
        }
    }
    matrix_barrier();
    if live == 0u {
        return;
    }

    let block_rows = length / wr;
    let height = block_rows * wc;
    for (var row = lane; row < height; row += WORKGROUP) {
        scratch_store(ARRANGEMENT_BASE + row, 0u);
        scratch_store(ROW_COUNTS_BASE + row, 0u);
    }
    for (var word = lane; word < (length + 31u) / 32u; word += WORKGROUP) {
        scratch_store(PROCESSED_BASE + word, 0u);
    }
    matrix_barrier();

    if lane == 0u {
        if height == 0u || height > MAX_SUB {
            failed = 1u;
        } else {
            build_source(length, wc, wr, generator);
        }
        if failed == 0u {
            let directory = base + CATALOG_HEADER_WORDS;
            let records = directory + catalog[base + 2u];
            let start = catalog[directory + catalog_slot(wc, wr, length)];
            let zero_count = replay_transcript(length, wc, wr, records, start);
            if failed == 0u {
                matrix_rank = height - zero_count;
                resolve_arrangement(height, matrix_rank, zero_count);
            }
        }
        if failed != 0u {
            reject();
            live = 0u;
        }
    }
    matrix_barrier();
    if live == 0u {
        return;
    }

    // Rearrange the sparse source and apply the recorded column swaps. Sorting
    // each small row reproduces the ascending adjacency the hard and soft
    // correctors share without ever materializing a dense copy.
    let output_base = row_base();
    let swaps = swap_count;
    for (var row = lane; row < height; row += WORKGROUP) {
        var columns: array<u32, 16>;
        let source_row = scratch_load(ARRANGEMENT_BASE + row);
        for (var slot = 0u; slot < wr; slot += 1u) {
            var column = scratch_load(SOURCE_BASE + source_row * wr + slot);
            for (var swap = 0u; swap < swaps; swap += 1u) {
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
