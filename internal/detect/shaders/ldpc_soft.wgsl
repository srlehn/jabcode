// Soft-decision LDPC retry. One workgroup owns one failed gross sub-block and
// keeps all min-sum iterations on the device. Check rows and variables alternate
// exclusive ownership of the same compact edge-message array, so neither sweep
// needs floating-point atomics or a dense row-by-column matrix.

const MAX_SUB: u32 = 2816u;
const WORKGROUP: u32 = 256u;
const MAX_ITER: u32 = 25u;
const MAX_COLUMN_DEGREE: u32 = 10u;
const COLUMN_GRAPH_WORDS: u32 = MAX_SUB * (1u + MAX_COLUMN_DEGREE);

// Classifier margins arrive in [0, 32767]. Iterative sums are clamped well
// below i32 overflow, including the subtraction that forms an extrinsic value.
const MESSAGE_CAP: i32 = 16777216;

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
const PARAM_ROW_BASE: u32 = 14u;

const EVIDENCE_BASE: u32 = 180288u;
const EVIDENCE_SOFT_USED: u32 = EVIDENCE_BASE + 3u;
const EVIDENCE_SOFT_RESIDUAL: u32 = EVIDENCE_BASE + 4u;
const EVIDENCE_SOFT_ITERATIONS: u32 = EVIDENCE_BASE + 5u;
const RESIDUAL_INVALID: u32 = 0xFFFFFFFFu;

// rows is the hard corrector's existing row-major parity graph. columns is the
// retry-only reverse view: one count and ten sorted edge slots per variable.
@group(0) @binding(0) var<storage, read> rows: array<u32>;
@group(0) @binding(1) var<storage, read> columns: array<u32>;
@group(0) @binding(2) var<storage, read> bits: array<u32>;
@group(0) @binding(3) var<storage, read> reliability: array<u32>;
@group(0) @binding(4) var<storage, read> params: array<u32>;
@group(0) @binding(5) var<storage, read_write> messages: array<i32>;
@group(0) @binding(6) var<storage, read_write> net: array<atomic<u32>>;

var<workgroup> decisions: array<u32, MAX_SUB>;
var<workgroup> failed_checks: atomic<u32>;
var<workgroup> solved: u32;
var<workgroup> iterations: u32;

fn clamp_message(value: i32) -> i32 {
    return clamp(value, -MESSAGE_CAP, MESSAGE_CAP);
}

fn channel_value(start: u32, column: u32, fixed_from: u32) -> i32 {
    if column >= fixed_from {
        return MESSAGE_CAP;
    }
    let magnitude = i32(min(reliability[start + column], 32767u));
    return select(magnitude, -magnitude, (bits[start + column] & 1u) != 0u);
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let block = group.x;
    if block >= params[PARAM_BLOCKS] {
        return;
    }
    // A zero hard residual is already solved. Admission and shape failures
    // suppress this dispatch through its indirect record, so every nonzero
    // residual reaching this workgroup is a real retry candidate.
    let hard_residual = atomicLoad(&net[block]);
    if hard_residual == 0u || hard_residual == RESIDUAL_INVALID {
        return;
    }

    let tail = block == params[PARAM_TAIL_BLOCK];
    let length = select(params[PARAM_LENGTH], params[PARAM_TAIL_LENGTH], tail);
    let height = select(params[PARAM_HEIGHT], params[PARAM_TAIL_HEIGHT], tail);
    let rank = select(params[PARAM_RANK], params[PARAM_TAIL_RANK], tail);
    let net_length = select(params[PARAM_NET], params[PARAM_TAIL_NET], tail);
    let degree = select(params[PARAM_ROW_DEGREE], params[PARAM_TAIL_ROW_DEGREE], tail);
    let row_base = select(params[PARAM_ROW_BASE], params[PARAM_TAIL_ROW_BASE], tail);
    let column_base = select(0u, COLUMN_GRAPH_WORDS, tail);
    let column_edges_base = column_base + MAX_SUB;
    if length > MAX_SUB || rank > height || height > length {
        return;
    }

    let uniform_edges = params[PARAM_HEIGHT] * params[PARAM_ROW_DEGREE];
    let edges = height * degree;
    let message_base = block * uniform_edges;
    let start = block * params[PARAM_LENGTH];
    let fixed_from = length - (height - rank);
    let lane = local.x;
    if lane == 0u {
        atomicAdd(&net[EVIDENCE_SOFT_USED], 1u);
        iterations = 0u;
    }
    workgroupBarrier();

    for (var column = lane; column < length; column += WORKGROUP) {
        decisions[column] = select(bits[start + column] & 1u, 0u, column >= fixed_from);
    }
    workgroupBarrier();

    // Variable-to-check messages start as the signed channel evidence. The
    // row-major edge number is shared by both graph directions.
    for (var edge = lane; edge < edges; edge += WORKGROUP) {
        let column = rows[row_base + edge];
        messages[message_base + edge] = channel_value(start, column, fixed_from);
    }
    storageBarrier();
    workgroupBarrier();

    for (var iteration = 0u; iteration < MAX_ITER; iteration += 1u) {
        if lane == 0u {
            iterations = iteration + 1u;
        }
        // Check-node update. A row lane finds the two lowest magnitudes once,
        // then writes every outgoing value without communicating with another
        // row lane.
        for (var row = lane; row < height; row += WORKGROUP) {
            let first = row * degree;
            let last = first + degree;
            var negative = false;
            var minimum = MESSAGE_CAP;
            var second = MESSAGE_CAP;
            for (var edge = first; edge < last; edge += 1u) {
                let value = messages[message_base + edge];
                negative = negative != (value < 0);
                let magnitude = abs(value);
                if magnitude < minimum {
                    second = minimum;
                    minimum = magnitude;
                } else if magnitude < second {
                    second = magnitude;
                }
            }
            for (var edge = first; edge < last; edge += 1u) {
                let incoming = messages[message_base + edge];
                let magnitude = select(minimum, second, abs(incoming) == minimum);
                let outgoing_negative = negative != (incoming < 0);
                messages[message_base + edge] = select(magnitude, -magnitude, outgoing_negative);
            }
        }
        storageBarrier();
        workgroupBarrier();

        // Variable-node update. A variable lane owns all incident edges, so it
        // can form its posterior and write every extrinsic message directly.
        for (var column = lane; column < length; column += WORKGROUP) {
            let count = min(columns[column_base + column], MAX_COLUMN_DEGREE);
            let first = column_edges_base + column * MAX_COLUMN_DEGREE;
            let last = first + count;
            if column >= fixed_from {
                decisions[column] = 0u;
                for (var at = first; at < last; at += 1u) {
                    let edge = columns[at];
                    messages[message_base + edge] = MESSAGE_CAP;
                }
                continue;
            }
            var posterior = channel_value(start, column, fixed_from);
            for (var at = first; at < last; at += 1u) {
                let edge = columns[at];
                posterior = clamp_message(posterior + messages[message_base + edge]);
            }
            decisions[column] = select(0u, 1u, posterior < 0);
            for (var at = first; at < last; at += 1u) {
                let edge = columns[at];
                messages[message_base + edge] = clamp_message(
                    posterior - messages[message_base + edge],
                );
            }
        }
        storageBarrier();
        workgroupBarrier();

        if lane == 0u {
            atomicStore(&failed_checks, 0u);
        }
        workgroupBarrier();
        for (var row = lane; row < rank; row += WORKGROUP) {
            let first = row * degree;
            let last = first + degree;
            var parity = 0u;
            for (var edge = first; edge < last; edge += 1u) {
                parity ^= decisions[rows[row_base + edge]];
            }
            if parity != 0u {
                atomicAdd(&failed_checks, 1u);
            }
        }
        workgroupBarrier();
        if lane == 0u {
            solved = select(0u, 1u, atomicLoad(&failed_checks) == 0u);
        }
        workgroupBarrier();
        if solved != 0u {
            break;
        }
    }

    if lane == 0u {
        let residual = atomicLoad(&failed_checks);
        atomicStore(&net[block], residual);
        atomicAdd(&net[EVIDENCE_SOFT_RESIDUAL], residual);
        atomicAdd(&net[EVIDENCE_SOFT_ITERATIONS], iterations);
    }
    let out = params[PARAM_BLOCKS] + block * params[PARAM_NET];
    for (var index = lane; index < net_length; index += WORKGROUP) {
        atomicStore(&net[out + index], decisions[rank + index]);
    }
}
