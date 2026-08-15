// Builds the reverse Tanner adjacency from the hard corrector's resident parity
// rows only when a hard syndrome failed. Edges append in parallel, then each
// variable lane sorts its at-most-ten edge indexes so the fixed-point variable
// sum has one deterministic order on every device.

const MAX_SUB: u32 = 2816u;
const MAX_COLUMN_DEGREE: u32 = 10u;
const COLUMN_GRAPH_WORDS: u32 = MAX_SUB * (1u + MAX_COLUMN_DEGREE);
const WORKGROUP: u32 = 256u;

const PARAM_LENGTH: u32 = 0u;
const PARAM_HEIGHT: u32 = 1u;
const PARAM_BLOCKS: u32 = 4u;
const PARAM_ROW_DEGREE: u32 = 5u;
const PARAM_TAIL_BLOCK: u32 = 6u;
const PARAM_TAIL_LENGTH: u32 = 7u;
const PARAM_TAIL_HEIGHT: u32 = 8u;
const PARAM_TAIL_ROW_DEGREE: u32 = 11u;
const PARAM_TAIL_ROW_BASE: u32 = 12u;
const PARAM_ROW_BASE: u32 = 14u;

// A block that the hard stage left unsolved is the one this graph exists for.
// The count is that block's residual, not a flag: comparing it against one
// built the adjacency only for the single-check case and left the min-sum to
// run over an empty graph for every other failure, where it diverged instead of
// correcting. The predicate is the retry predicate in ldpc_soft_prepare.wgsl.
const RESIDUAL_INVALID: u32 = 0xFFFFFFFFu;

fn block_failed(residual: u32) -> bool {
    return residual != 0u && residual != RESIDUAL_INVALID;
}

@group(0) @binding(0) var<storage, read> rows: array<u32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;
@group(0) @binding(2) var<storage, read> net: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> columns: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    if group.x > 1u {
        return;
    }
    var retry = false;
    let blocks = params[PARAM_BLOCKS];
    for (var block = 0u; block < blocks; block += 1u) {
        retry = retry || block_failed(atomicLoad(&net[block]));
    }
    if !retry {
        return;
    }

    let tail = group.x == 1u;
    if tail && params[PARAM_TAIL_BLOCK] == blocks {
        return;
    }
    let length = select(params[PARAM_LENGTH], params[PARAM_TAIL_LENGTH], tail);
    let height = select(params[PARAM_HEIGHT], params[PARAM_TAIL_HEIGHT], tail);
    let degree = select(params[PARAM_ROW_DEGREE], params[PARAM_TAIL_ROW_DEGREE], tail);
    let row_base = select(params[PARAM_ROW_BASE], params[PARAM_TAIL_ROW_BASE], tail);
    let column_base = select(0u, COLUMN_GRAPH_WORDS, tail);
    let edge_base = column_base + MAX_SUB;
    let lane = local.x;

    for (var column = lane; column < length; column += WORKGROUP) {
        atomicStore(&columns[column_base + column], 0u);
    }
    workgroupBarrier();

    for (var edge = lane; edge < height * degree; edge += WORKGROUP) {
        let column = rows[row_base + edge];
        let slot = atomicAdd(&columns[column_base + column], 1u);
        if slot < MAX_COLUMN_DEGREE {
            atomicStore(&columns[edge_base + column * MAX_COLUMN_DEGREE + slot], edge);
        }
    }
    storageBarrier();
    workgroupBarrier();

    for (var column = lane; column < length; column += WORKGROUP) {
        let count = min(atomicLoad(&columns[column_base + column]), MAX_COLUMN_DEGREE);
        let base = edge_base + column * MAX_COLUMN_DEGREE;
        for (var at = 1u; at < count; at += 1u) {
            let value = atomicLoad(&columns[base + at]);
            var before = at;
            while before > 0u && atomicLoad(&columns[base + before - 1u]) > value {
                atomicStore(
                    &columns[base + before],
                    atomicLoad(&columns[base + before - 1u]),
                );
                before -= 1u;
            }
            atomicStore(&columns[base + before], value);
        }
    }
}
