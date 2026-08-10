// Builds the data map of a sampled symbol and numbers its data modules, so the
// classification kernel can write each module's bits straight into the codeword
// without the host telling it where they go.
//
// The host walks the whole grid once to reserve function-pattern modules, then
// walks it again appending data modules to a slice, which makes a module's
// position in the codeword an artefact of iteration order. A lane cannot append
// to a slice, so the position is derived instead: the reservation is a scatter
// over alignment-pattern cells, and the column-major numbering the host produces
// by appending is exactly an exclusive prefix sum over the data flags in that
// order. One workgroup owns the whole map because every phase feeds the next and
// a workgroup barrier is the only free way to order them.
//
// The metadata modules are marked by replaying the metadata walk rather than by
// having the host send their positions: the walk is geometry plus the number of
// modules it consumed, and that count is one word in the parameter block.

const MAX_SIDE: u32 = 145u;
const WORKGROUP: u32 = 256u;
const NOT_DATA: u32 = 0xffffffffu;

const PARAM_SIDE_X: u32 = 0u;
const PARAM_SIDE_Y: u32 = 1u;
const PARAM_META_MODULES: u32 = 2u;
const PARAM_BITS_PER_MODULE: u32 = 4u;
const PARAM_AP_NUM_X: u32 = 6u;
const PARAM_AP_NUM_Y: u32 = 7u;
const PARAM_GROSS_BITS: u32 = 8u;
const PARAM_SYMBOL_TYPE: u32 = 11u;
const PARAM_AP_POS_X: u32 = 12u;
const PARAM_AP_POS_Y: u32 = 21u;
const PARAM_ADMISSION: u32 = 1714u;
const PARAM_DATA_MODULES: u32 = 1715u;
const PARAM_NET_BITS: u32 = 1716u;
const PARAM_WC: u32 = 1717u;
const PARAM_WR: u32 = 1718u;

const LDPC_PARAM_LENGTH: u32 = 0u;
const LDPC_PARAM_NET: u32 = 3u;
const LDPC_PARAM_BLOCKS: u32 = 4u;
const LDPC_PARAM_TAIL_BLOCK: u32 = 6u;
const LDPC_PARAM_TAIL_LENGTH: u32 = 7u;
const LDPC_PARAM_TAIL_NET: u32 = 10u;
const LDPC_PARAM_ADMISSION: u32 = 13u;

@group(0) @binding(0) var<storage, read_write> params: array<u32>;
@group(0) @binding(1) var<storage, read_write> map: array<u32>;
@group(0) @binding(2) var<storage, read_write> ldpc_params: array<u32>;

var<workgroup> column_total: array<u32, MAX_SIDE>;

fn reject_shape() {
    params[PARAM_ADMISSION] = 2u;
    params[PARAM_GROSS_BITS] = 0u;
    params[PARAM_NET_BITS] = 0u;
    ldpc_params[LDPC_PARAM_BLOCKS] = 0u;
    ldpc_params[LDPC_PARAM_ADMISSION] = 2u;
}

// reserve marks one module as a function pattern. Alignment-pattern cells are
// spaced further apart than the two modules a cell reaches, so no two lanes
// address the same module and every write is the same value regardless.
fn reserve(x: i32, y: i32) {
    let side_x = i32(params[PARAM_SIDE_X]);
    let side_y = i32(params[PARAM_SIDE_Y]);
    if x < 0 || y < 0 || x >= side_x || y >= side_y {
        return;
    }
    map[u32(x) * u32(side_y) + u32(y)] = 1u;
}

// reserve_cell marks the modules one alignment-pattern cell occupies. The four
// corner cells carry a finder pattern instead, which is two layers deep on a
// primary symbol and one on a secondary.
fn reserve_cell(cell: u32) {
    let ap_x = params[PARAM_AP_NUM_X];
    let ap_y = params[PARAM_AP_NUM_Y];
    let j = cell % ap_x;
    let i = cell / ap_x;
    let xo = i32(params[PARAM_AP_POS_X + j]) - 1;
    let yo = i32(params[PARAM_AP_POS_Y + i]) - 1;

    reserve(xo, yo);
    reserve(xo - 1, yo);
    reserve(xo + 1, yo);
    reserve(xo, yo - 1);
    reserve(xo, yo + 1);

    let corner = j == 0u || j == ap_x - 1u;
    let primary = params[PARAM_SYMBOL_TYPE] == 0u;
    if i == 0u && corner {
        reserve(xo - 1, yo - 1);
        reserve(xo + 1, yo + 1);
        if primary {
            reserve(xo - 2, yo - 2);
            reserve(xo - 1, yo - 2);
            reserve(xo, yo - 2);
            reserve(xo - 2, yo - 1);
            reserve(xo - 2, yo);
            reserve(xo + 2, yo + 2);
            reserve(xo + 1, yo + 2);
            reserve(xo, yo + 2);
            reserve(xo + 2, yo + 1);
            reserve(xo + 2, yo);
        }
        return;
    }
    if i == ap_y - 1u && corner {
        reserve(xo + 1, yo - 1);
        reserve(xo - 1, yo + 1);
        if primary {
            reserve(xo + 2, yo - 2);
            reserve(xo + 1, yo - 2);
            reserve(xo, yo - 2);
            reserve(xo + 2, yo - 1);
            reserve(xo + 2, yo);
            reserve(xo - 2, yo + 2);
            reserve(xo - 1, yo + 2);
            reserve(xo, yo + 2);
            reserve(xo - 2, yo + 1);
            reserve(xo - 2, yo);
        }
        return;
    }
    if (i % 2u) == (j % 2u) {
        reserve(xo - 1, yo - 1);
        reserve(xo + 1, yo + 1);
        return;
    }
    reserve(xo + 1, yo - 1);
    reserve(xo - 1, yo + 1);
}

// reserve_metadata replays the primary metadata walk for the number of modules
// the host's interpretation consumed. The walk is a fixed sequence of reflections
// about the symbol's axes, so it is serial by construction and one lane runs it.
fn reserve_metadata() {
    let width = i32(params[PARAM_SIDE_X]);
    let height = i32(params[PARAM_SIDE_Y]);
    let modules = params[PARAM_META_MODULES];
    var x = 6;
    var y = 1;
    for (var taken = 0u; taken < modules; taken += 1u) {
        reserve(x, y);
        let count = i32(taken) + 1;
        let phase = count % 4;
        if phase == 0 || phase == 2 {
            y = height - 1 - y;
        }
        if phase == 1 || phase == 3 {
            x = width - 1 - x;
        }
        if phase == 0 {
            if count <= 20 || (count >= 44 && count <= 68) ||
                (count >= 96 && count <= 124) || (count >= 156 && count <= 172) {
                y += 1;
            } else if (count > 20 && count < 44) || (count > 68 && count < 96) ||
                (count > 124 && count < 156) {
                x -= 1;
            }
        }
        if count == 44 || count == 96 || count == 156 {
            let swap = x;
            x = y;
            y = swap;
        }
    }
}

// write_payload_shape is the device twin of ecc.HardBlockSplit for message
// codes. The exact data-module count exists only after the column prefix, so
// resolving the split here avoids both a map-count download and a host-selected
// control upload. Matrix rank and sparse row layout are filled by the selected
// matrix builder that follows this fold.
fn write_payload_shape(data_modules: u32) {
    let wc = params[PARAM_WC];
    let wr = params[PARAM_WR];
    let length = data_modules * params[PARAM_BITS_PER_MODULE];
    if data_modules == 0u || wc < 3u || wc >= wr || wr > 11u {
        reject_shape();
        return;
    }
    let gross = wr * (length / wr);
    if gross == 0u {
        reject_shape();
        return;
    }
    let net = gross * (wr - wc) / wr;

    var split = 0u;
    for (var candidate = 1u; candidate < 10000u; candidate += 1u) {
        if gross / candidate < 2700u {
            split = candidate;
            break;
        }
    }
    if split == 0u {
        reject_shape();
        return;
    }
    let gross_sub = ((gross / split) / wr) * wr;
    if gross_sub == 0u {
        reject_shape();
        return;
    }
    let net_sub = gross_sub * (wr - wc) / wr;
    let blocks = gross / gross_sub;
    var uniform_blocks = blocks;
    if net_sub * blocks < net {
        uniform_blocks -= 1u;
    }

    params[PARAM_DATA_MODULES] = data_modules;
    params[PARAM_GROSS_BITS] = gross;
    params[PARAM_NET_BITS] = net;
    ldpc_params[LDPC_PARAM_LENGTH] = gross_sub;
    ldpc_params[LDPC_PARAM_NET] = net_sub;
    ldpc_params[LDPC_PARAM_BLOCKS] = blocks;
    ldpc_params[LDPC_PARAM_TAIL_BLOCK] = blocks;
    ldpc_params[LDPC_PARAM_TAIL_LENGTH] = 0u;
    ldpc_params[LDPC_PARAM_TAIL_NET] = 0u;
    if uniform_blocks != blocks {
        let tail = gross - uniform_blocks * gross_sub;
        ldpc_params[LDPC_PARAM_TAIL_BLOCK] = blocks - 1u;
        ldpc_params[LDPC_PARAM_TAIL_LENGTH] = tail;
        ldpc_params[LDPC_PARAM_TAIL_NET] = tail * (wr - wc) / wr;
    }
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
    let lane = local.x;
    let side_x = params[PARAM_SIDE_X];
    let side_y = params[PARAM_SIDE_Y];
    let modules = side_x * side_y;

    for (var at = lane; at < modules; at += WORKGROUP) {
        map[at] = 0u;
    }
    workgroupBarrier();

    let cells = params[PARAM_AP_NUM_X] * params[PARAM_AP_NUM_Y];
    for (var cell = lane; cell < cells; cell += WORKGROUP) {
        reserve_cell(cell);
    }
    workgroupBarrier();

    if lane == 0u {
        reserve_metadata();
    }
    workgroupBarrier();

    // Each column numbers its own data modules; the columns are then offset by
    // the running total of the columns before them, which reproduces the host's
    // column-major append order without any lane knowing that order.
    if lane < side_x {
        var run = 0u;
        for (var y = 0u; y < side_y; y += 1u) {
            let at = lane * side_y + y;
            if map[at] == 0u {
                map[at] = run;
                run += 1u;
            } else {
                map[at] = NOT_DATA;
            }
        }
        column_total[lane] = run;
    }
    workgroupBarrier();

    if lane == 0u {
        var base = 0u;
        for (var column = 0u; column < side_x; column += 1u) {
            let total = column_total[column];
            column_total[column] = base;
            base += total;
        }
        write_payload_shape(base);
    }
    workgroupBarrier();

    if lane < side_x {
        let base = column_total[lane];
        for (var y = 0u; y < side_y; y += 1u) {
            let at = lane * side_y + y;
            if map[at] != NOT_DATA {
                map[at] += base;
            }
        }
    }
}
