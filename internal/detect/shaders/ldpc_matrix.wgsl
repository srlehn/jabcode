// Builds the one decoder parity-check matrix selected by resident payload
// control. The Gallager source stays sparse, while only the elimination copy is
// expanded into packed words. This keeps retained storage proportional to one
// selected shape instead of precomputing every colour and ECC candidate.
//
// MATRIX_SLOT and MATRIX_PHASE are prepended by the host when compiling the
// pipelines. Slot zero builds the regular sub-block and slot one the optional
// tail; they dispatch sequentially and reuse one workspace, so the uncommon tail
// does not double the largest retained matrix allocation. The phases are stages
// of one elimination, and only the slot-dependent ones are compiled twice.
//
// The elimination is blocked so its expensive part leaves a single workgroup.
// Pivots are sequential and WGSL has no grid-wide barrier, but that orders the
// pivots only against each other. A panel of PANEL consecutive pivots reduces
// the panel's own rows in workgroup memory, and one grid-wide dispatch then
// updates every other row of the matrix. So the matrix streams through memory
// once per panel and from the whole device, instead of once per pivot through
// whichever multiprocessor holds the single workgroup.
//
// The blocked update is exact, not an approximation of the sweep. Panel rows end
// mutually reduced, so each carries a one at its own pivot column and a zero at
// every other pivot column of the panel. A row outside the panel has to end with
// a zero at all of them, and that fixes its combination uniquely: it takes final
// panel row j exactly when it held a one at pivot column j before the panel ran.
// The panel writes no row but its own, so the apply pass reads that predicate
// out of the matrix itself rather than carrying a transform across the boundary.

const PHASE_SETUP: u32 = 0u;
const PHASE_PANEL: u32 = 1u;
const PHASE_APPLY: u32 = 2u;
const PHASE_FINISH: u32 = 3u;

// The workgroup width the rest of the resident stage already requires, so this
// kernel adds no device capability of its own. Every phase strides its work over
// whatever width is declared, so the size is a tuning choice rather than an
// assumption the algorithm makes.
const WORKGROUP: u32 = 256u;

// PANEL is bounded by the 32 bits of the per-row selection mask the apply pass
// carries, and PANEL * MAX_STRIDE words of workgroup memory have to hold the
// staged panel, which is what keeps it inside the 16 KB a conformant device
// guarantees. TILE_ROWS is how many matrix rows one apply workgroup owns; it
// shares the staging cost of a panel across that many rows.
const PANEL: u32 = 32u;
const TILE_ROWS: u32 = 32u;

const MAX_SUB: u32 = 2816u;
const MAX_ROW_DEGREE: u32 = 16u;
const MAX_STRIDE: u32 = (MAX_SUB + 31u) / 32u;
const MAX_PANEL_STEPS: u32 = (MAX_SUB + PANEL - 1u) / PANEL;
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
const STATE_BASE: u32 = PERM_BASE + MAX_SUB;

// Elimination state outlives a dispatch now, so it sits in the workspace rather
// than in workgroup memory. The apply dispatch dimensions live here too: the
// panel phase is the only stage that knows how much matrix is left, so it writes
// its own successor's grid and writes zeroes once the sweep is done.
const STATE_ZERO_COUNT: u32 = 0u;
const STATE_SWAP_COUNT: u32 = 1u;
const STATE_ROW: u32 = 2u;
const STATE_HEIGHT: u32 = 3u;
const STATE_STRIDE: u32 = 4u;
const STATE_LIVE: u32 = 5u;
const STATE_PANEL_BASE: u32 = 6u;
const STATE_PANEL_SPAN: u32 = 7u;
const STATE_PANEL_COUNT: u32 = 8u;
const STATE_APPLY_DIMS: u32 = 9u;
const STATE_PIVOT_COLUMN: u32 = 12u;
const STATE_PIVOT_ROW: u32 = STATE_PIVOT_COLUMN + PANEL;
const STATE_WORDS: u32 = STATE_PIVOT_ROW + PANEL;

const SCRATCH_WORDS: u32 = STATE_BASE + STATE_WORDS;

const ROW_SET_WORDS: u32 = MAX_SUB * MAX_ROW_DEGREE;
const MESSAGE_SEED: u32 = 785465u;
const GENERATOR_LCG: u32 = 1u;

const NO_PIVOT: u32 = 0xFFFFFFFFu;

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

var<workgroup> pivot_min: atomic<u32>;
var<workgroup> eliminate: atomic<u32>;
var<workgroup> live: u32;
var<workgroup> failed: u32;
var<workgroup> zero_count: u32;
var<workgroup> swap_count: u32;
var<workgroup> panel_count: u32;
var<workgroup> matrix_rank: u32;

// One staging array serves both blocked phases: the panel phase holds the rows
// it is reducing, the apply phase holds the finished panel rows it distributes.
// Declaring it once is what keeps workgroup memory at a single figure across the
// pipelines instead of the sum of two layouts.
var<workgroup> tile: array<u32, PANEL * MAX_STRIDE>;
var<workgroup> tile_mask: array<u32, TILE_ROWS>;

fn scratch_load(at: u32) -> u32 {
    return scratch[at];
}

fn scratch_store(at: u32, value: u32) {
    scratch[at] = value;
}

fn state_load(at: u32) -> u32 {
    return scratch[STATE_BASE + at];
}

fn state_store(at: u32, value: u32) {
    scratch[STATE_BASE + at] = value;
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

// The apply dispatches are recorded to the worst-case panel count and read their
// grid from here, so a stage that runs out of panels has to zero it or the tail
// of that recording repeats the last panel it was given.
fn clear_apply() {
    state_store(STATE_APPLY_DIMS, 0u);
    state_store(STATE_APPLY_DIMS + 1u, 1u);
    state_store(STATE_APPLY_DIMS + 2u, 1u);
}

// halt abandons the build outright, which only setup can decide: the shape is
// invalid, or the resident key already answers it. Running out of panels is not
// this, because the arrangement still has to be resolved and published.
fn halt() {
    state_store(STATE_LIVE, 0u);
    clear_apply();
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

// setup validates the selected shape, answers from the resident key when it
// still matches, and otherwise lays down the sparse Gallager source and its
// packed elimination copy. The permutation walk is a Fisher-Yates and inherently
// serial, so one lane runs it; everything after it is width-striped.
fn run_setup(lane: u32) {
    let length = matrix_length();
    let wc = payload_params[PAYLOAD_PARAM_WC];
    let wr = payload_params[PAYLOAD_PARAM_WR];
    let generator = payload_params[PAYLOAD_PARAM_GENERATOR];
    let cache_base = MATRIX_SLOT * CACHE_WORDS;

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
        // check, exactly as it did when this was one dispatch: the answer was
        // admitted when it was built and re-rejecting it here would retract it.
        if length == 0u || length > MAX_SUB || wc < 3u || wc >= wr ||
            wr > 11u || wr > MAX_ROW_DEGREE {
            if live != 0u {
                reject();
            }
            live = 0u;
        }
        if live == 0u {
            halt();
        }
    }
    matrix_barrier();
    if live == 0u {
        return;
    }

    let block_rows = length / wr;
    let height = block_rows * wc;
    let stride = (length + 31u) / 32u;
    // Dense cells, sparse rows and the permutation are overwritten completely.
    // Only the counters and state read before assignment need clearing. Avoid
    // zeroing the full worst-case workspace for every selected payload shape.
    for (var row = lane; row < height; row += WORKGROUP) {
        scratch_store(ARRANGEMENT_BASE + row, 0u);
        scratch_store(ROW_COUNTS_BASE + row, 0u);
    }
    for (var word = lane; word < stride; word += WORKGROUP) {
        scratch_store(PROCESSED_BASE + word, 0u);
    }
    matrix_barrier();
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
        if failed != 0u {
            reject();
            live = 0u;
            halt();
        }
    }
    matrix_barrier();
    if live == 0u {
        return;
    }

    // Flatten row-word cells across the workgroup. Adjacent lanes then write
    // adjacent storage words instead of walking rows a full stride apart.
    for (var cell = lane; cell < height * stride; cell += WORKGROUP) {
        let row = cell / stride;
        let word = cell % stride;
        var value = 0u;
        for (var slot = 0u; slot < wr; slot += 1u) {
            let column = scratch_load(SOURCE_BASE + row * wr + slot);
            if column / 32u == word {
                value |= 1u << (column % 32u);
            }
        }
        scratch_store(DENSE_BASE + cell, value);
    }
    if lane == 0u {
        state_store(STATE_ZERO_COUNT, 0u);
        state_store(STATE_SWAP_COUNT, 0u);
        state_store(STATE_ROW, 0u);
        state_store(STATE_HEIGHT, height);
        state_store(STATE_STRIDE, stride);
        state_store(STATE_PANEL_COUNT, 0u);
        state_store(STATE_APPLY_DIMS, 0u);
        state_store(STATE_APPLY_DIMS + 1u, 1u);
        state_store(STATE_APPLY_DIMS + 2u, 1u);
        state_store(STATE_LIVE, 1u);
    }
    matrix_barrier();
}

// run_panel reduces PANEL consecutive rows against each other entirely in
// workgroup memory, recording the pivot columns the apply pass then distributes.
// Rows outside the panel are neither read for their contents nor written here,
// which is what lets the apply pass recover each one's combination from its own
// bits at those pivot columns.
fn run_panel(lane: u32) {
    if lane == 0u {
        live = state_load(STATE_LIVE);
        if live != 0u && state_load(STATE_ROW) >= state_load(STATE_HEIGHT) {
            live = 0u;
            clear_apply();
        }
    }
    matrix_barrier();
    if live == 0u {
        return;
    }

    let height = state_load(STATE_HEIGHT);
    let stride = state_load(STATE_STRIDE);
    let base = state_load(STATE_ROW);
    let span = min(PANEL, height - base);

    for (var cell = lane; cell < span * stride; cell += WORKGROUP) {
        let row = cell / stride;
        let word = cell % stride;
        tile[row * MAX_STRIDE + word] = scratch_load(DENSE_BASE + (base + row) * stride + word);
    }
    if lane == 0u {
        zero_count = state_load(STATE_ZERO_COUNT);
        swap_count = state_load(STATE_SWAP_COUNT);
        panel_count = 0u;
    }
    workgroupBarrier();

    for (var index = 0u; index < span; index += 1u) {
        if lane == 0u {
            atomicStore(&pivot_min, NO_PIVOT);
            atomicStore(&eliminate, 0u);
        }
        workgroupBarrier();
        for (var word = lane; word < stride; word += WORKGROUP) {
            let value = tile[index * MAX_STRIDE + word];
            if value != 0u {
                atomicMin(&pivot_min, word * 32u + firstTrailingBit(value));
            }
        }
        workgroupBarrier();
        let pivot = atomicLoad(&pivot_min);
        // One lane per panel row snapshots the elimination predicate before any
        // lane clears a pivot bit. The pivot row is excluded, so it stays
        // readable as the source while the targets are updated in place.
        if pivot != NO_PIVOT && lane < span && lane != index {
            let word = tile[lane * MAX_STRIDE + pivot / 32u];
            if ((word >> (pivot % 32u)) & 1u) != 0u {
                atomicOr(&eliminate, 1u << lane);
            }
        }
        if lane == 0u {
            let row = base + index;
            if pivot == NO_PIVOT {
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
                state_store(STATE_PIVOT_COLUMN + panel_count, pivot);
                state_store(STATE_PIVOT_ROW + panel_count, row);
                panel_count += 1u;
            }
        }
        matrix_barrier();
        if pivot != NO_PIVOT {
            let selected = atomicLoad(&eliminate);
            for (var cell = lane; cell < span * stride; cell += WORKGROUP) {
                let row = cell / stride;
                if ((selected >> row) & 1u) == 0u {
                    continue;
                }
                let word = cell % stride;
                tile[row * MAX_STRIDE + word] ^= tile[index * MAX_STRIDE + word];
            }
        }
        workgroupBarrier();
    }

    for (var cell = lane; cell < span * stride; cell += WORKGROUP) {
        let row = cell / stride;
        let word = cell % stride;
        scratch_store(DENSE_BASE + (base + row) * stride + word, tile[row * MAX_STRIDE + word]);
    }
    if lane == 0u {
        state_store(STATE_ZERO_COUNT, zero_count);
        state_store(STATE_SWAP_COUNT, swap_count);
        state_store(STATE_PANEL_BASE, base);
        state_store(STATE_PANEL_SPAN, span);
        state_store(STATE_PANEL_COUNT, panel_count);
        state_store(STATE_ROW, base + span);
        state_store(STATE_APPLY_DIMS, select(0u, (height + TILE_ROWS - 1u) / TILE_ROWS, panel_count > 0u));
        state_store(STATE_APPLY_DIMS + 1u, 1u);
        state_store(STATE_APPLY_DIMS + 2u, 1u);
    }
    matrix_barrier();
}

// run_apply distributes one finished panel across the rest of the matrix. Each
// workgroup owns TILE_ROWS rows and stages the panel once for all of them, so
// the panel is read from memory a number of times proportional to the row count
// rather than to the cell count.
fn run_apply(lane: u32, group: u32) {
    if lane == 0u {
        live = state_load(STATE_LIVE);
    }
    matrix_barrier();
    if live == 0u {
        return;
    }

    let height = state_load(STATE_HEIGHT);
    let stride = state_load(STATE_STRIDE);
    let base = state_load(STATE_PANEL_BASE);
    let span = state_load(STATE_PANEL_SPAN);
    let count = state_load(STATE_PANEL_COUNT);
    let first = group * TILE_ROWS;

    for (var cell = lane; cell < count * stride; cell += WORKGROUP) {
        let index = cell / stride;
        let word = cell % stride;
        let row = state_load(STATE_PIVOT_ROW + index);
        tile[index * MAX_STRIDE + word] = scratch_load(DENSE_BASE + row * stride + word);
    }
    // The panel wrote only its own rows, so every other row still holds the bits
    // it had before the panel ran and the selection reads straight off it.
    for (var index = lane; index < TILE_ROWS; index += WORKGROUP) {
        let row = first + index;
        var mask = 0u;
        if row < height && (row < base || row >= base + span) {
            for (var slot = 0u; slot < count; slot += 1u) {
                let column = state_load(STATE_PIVOT_COLUMN + slot);
                let word = scratch_load(DENSE_BASE + row * stride + column / 32u);
                if ((word >> (column % 32u)) & 1u) != 0u {
                    mask |= 1u << slot;
                }
            }
        }
        tile_mask[index] = mask;
    }
    matrix_barrier();

    for (var cell = lane; cell < TILE_ROWS * stride; cell += WORKGROUP) {
        let index = cell / stride;
        var selected = tile_mask[index];
        if selected == 0u {
            continue;
        }
        let word = cell % stride;
        let at = DENSE_BASE + (first + index) * stride + word;
        var value = scratch_load(at);
        while selected != 0u {
            let slot = firstTrailingBit(selected);
            value ^= tile[slot * MAX_STRIDE + word];
            selected &= selected - 1u;
        }
        scratch_store(at, value);
    }
    matrix_barrier();
}

// run_finish resolves the column arrangement the sweep recorded and emits the
// sparse rows. It reads the untouched Gallager source rather than the eliminated
// copy, because the decoder wants the original matrix in systematic order.
fn run_finish(lane: u32) {
    if lane == 0u {
        live = state_load(STATE_LIVE);
    }
    matrix_barrier();
    if live == 0u {
        return;
    }

    let height = state_load(STATE_HEIGHT);
    let wr = payload_params[PAYLOAD_PARAM_WR];

    if lane == 0u {
        zero_count = state_load(STATE_ZERO_COUNT);
        swap_count = state_load(STATE_SWAP_COUNT);
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
        let length = matrix_length();
        let cache_base = MATRIX_SLOT * CACHE_WORDS;
        publish_control(height, matrix_rank);
        cache[cache_base + CACHE_LENGTH] = length;
        cache[cache_base + CACHE_WC] = payload_params[PAYLOAD_PARAM_WC];
        cache[cache_base + CACHE_WR] = wr;
        cache[cache_base + CACHE_GENERATOR] = payload_params[PAYLOAD_PARAM_GENERATOR];
        cache[cache_base + CACHE_HEIGHT] = height;
        cache[cache_base + CACHE_RANK] = matrix_rank;
        cache[cache_base + CACHE_VALID] = 1u;
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    if payload_params[PAYLOAD_PARAM_ADMISSION] != 0u {
        return;
    }
    let lane = local.x;
    if MATRIX_PHASE == PHASE_SETUP {
        run_setup(lane);
    } else if MATRIX_PHASE == PHASE_PANEL {
        run_panel(lane);
    } else if MATRIX_PHASE == PHASE_APPLY {
        run_apply(lane, group.x);
    } else {
        run_finish(lane);
    }
}
