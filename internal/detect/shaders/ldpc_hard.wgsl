// Hard-decision LDPC correction: one workgroup per gross sub-block, every
// iteration resident in workgroup memory.
//
// The sub-blocks are independent, which is the outer parallelism, and inside
// one the bit-flipping iteration is two parallel sweeps rather than the host's
// two serial loops. The host walks every parity row accumulating implication
// counts, then walks every bit collecting the most-implicated ones into a
// candidate slice. Here a lane owns a parity row for the first sweep and a bit
// for the second, the implication counts are workgroup atomics, and "the
// most-implicated bits" comes out of a max reduction instead of a running best
// with a slice that gets truncated whenever the best improves. The candidate
// list never exists: a bit knows it is a candidate when its own count equals
// the reduced maximum.
//
// The iteration count stays serial because each flip depends on the last, but
// none of the 25 iterations leaves the workgroup, so the whole correction is
// one dispatch with no host contact.

// MAX_SUB bounds a gross sub-block. subBlockCount splits until Pg/i < 2700, so
// no sub-block exceeds that; this is the next multiple of 32 above it, which
// keeps the bit-packed arrays whole-word.
const MAX_SUB: u32 = 2816u;
const MAX_SUB_WORDS: u32 = MAX_SUB / 32u;
const WORKGROUP: u32 = 256u;
const MAX_ITER: u32 = 25u;

// Below this the reference flips one bit per iteration rather than every
// most-implicated bit, so a short block converges by the same path it does on
// the host.
const SINGLE_FLIP_BELOW: u32 = 36u;

// Matches ecc.ParityRowPad.
const ROW_PAD: u32 = 0xFFFFFFFFu;

const PARAM_LENGTH: u32 = 0u;
const PARAM_HEIGHT: u32 = 1u;
const PARAM_RANK: u32 = 2u;
const PARAM_NET: u32 = 3u;
const PARAM_BLOCKS: u32 = 4u;
const PARAM_ROW_DEGREE: u32 = 5u;

// rows holds each parity row's set columns, row_degree slots per row, so a
// row's slice starts at j * row_degree; a row shorter than the maximum degree
// is padded with ROW_PAD, which must be skipped rather than folded in. The matrix depends only on the code
// parameters, never on the image, so it is uploaded once per context and reused
// by every decode that shares them.
@group(0) @binding(0) var<storage, read> rows: array<u32>;
@group(0) @binding(1) var<storage, read> bits: array<u32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
// net carries the compacted message bits and, in its head, one status word per
// block. The corrected codeword is never written back over its own input:
// blocks run concurrently and a block's compacted destination can overlap the
// next block's source, which is safe only in the reference's sequential loop.
@group(0) @binding(3) var<storage, read_write> net: array<atomic<u32>>;

var<workgroup> implicated: array<atomic<u32>, MAX_SUB>;
var<workgroup> data_bits: array<u32, MAX_SUB_WORDS>;
var<workgroup> used_bits: array<u32, MAX_SUB_WORDS>;
var<workgroup> flip_bits: array<u32, MAX_SUB_WORDS>;
var<workgroup> reduce: array<u32, WORKGROUP>;
var<workgroup> best: u32;
var<workgroup> unsatisfied: atomic<u32>;

fn bit_at(index: u32) -> u32 {
    return (data_bits[index / 32u] >> (index % 32u)) & 1u;
}

fn flag_at(store: ptr<workgroup, array<u32, MAX_SUB_WORDS>>, index: u32) -> u32 {
    return ((*store)[index / 32u] >> (index % 32u)) & 1u;
}

// row_parity is the modulo-two sum of one parity row's bits. An odd sum means
// the check failed and every bit it touches is implicated.
fn row_parity(row: u32, degree: u32) -> u32 {
    var ones = 0u;
    let base = row * degree;
    for (var s = 0u; s < degree; s++) {
        let column = rows[base + s];
        if column == ROW_PAD {
            continue;
        }
        ones += bit_at(column);
    }
    return ones & 1u;
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
    let length = params[PARAM_LENGTH];
    let height = params[PARAM_HEIGHT];
    let rank = params[PARAM_RANK];
    let degree = params[PARAM_ROW_DEGREE];
    if length > MAX_SUB {
        if local.x == 0u {
            atomicStore(&net[block], 2u);
        }
        return;
    }
    let start = block * length;
    let lane = local.x;

    // Stage the sub-block's bits into workgroup memory once. Every iteration
    // reads them repeatedly, and the flips stay local until the block is done.
    //
    // A lane owns whole words rather than scattered bits throughout this
    // kernel. Scattering would make two lanes read-modify-write the same word,
    // which workgroup memory does not make safe on its own.
    for (var w = lane; w < MAX_SUB_WORDS; w += WORKGROUP) {
        var value = 0u;
        for (var b = 0u; b < 32u; b++) {
            let i = w * 32u + b;
            if i < length && bits[start + i] != 0u {
                value |= 1u << b;
            }
        }
        data_bits[w] = value;
        used_bits[w] = 0u;
    }
    workgroupBarrier();

    for (var iter = 0u; iter < MAX_ITER; iter++) {
        for (var i = lane; i < length; i += WORKGROUP) {
            atomicStore(&implicated[i], 0u);
        }
        workgroupBarrier();

        // One lane per parity row: an unsatisfied check implicates its bits.
        for (var j = lane; j < height; j += WORKGROUP) {
            if row_parity(j, degree) == 1u {
                let base = j * degree;
                for (var s = 0u; s < degree; s++) {
                    let column = rows[base + s];
                    if column == ROW_PAD {
                        continue;
                    }
                    atomicAdd(&implicated[column], 1u);
                }
            }
        }
        workgroupBarrier();

        // The most-implicated bit not flipped by the previous iteration.
        var mine = 0u;
        for (var i = lane; i < length; i += WORKGROUP) {
            if flag_at(&used_bits, i) == 0u {
                mine = max(mine, atomicLoad(&implicated[i]));
            }
        }
        reduce[lane] = mine;
        workgroupBarrier();
        for (var stride = WORKGROUP / 2u; stride > 0u; stride >>= 1u) {
            if lane < stride {
                reduce[lane] = max(reduce[lane], reduce[lane + stride]);
            }
            workgroupBarrier();
        }
        if lane == 0u {
            best = reduce[0];
        }
        workgroupBarrier();
        if best == 0u {
            break;
        }

        // Mark this iteration's flips. A short block flips only its lowest
        // candidate, which is the reference's deterministic tie-break.
        if length < SINGLE_FLIP_BELOW {
            var lowest = MAX_SUB;
            for (var i = lane; i < length; i += WORKGROUP) {
                if flag_at(&used_bits, i) == 0u && atomicLoad(&implicated[i]) == best {
                    lowest = min(lowest, i);
                }
            }
            reduce[lane] = lowest;
            workgroupBarrier();
            for (var stride = WORKGROUP / 2u; stride > 0u; stride >>= 1u) {
                if lane < stride {
                    reduce[lane] = min(reduce[lane], reduce[lane + stride]);
                }
                workgroupBarrier();
            }
            for (var w = lane; w < MAX_SUB_WORDS; w += WORKGROUP) {
                flip_bits[w] = 0u;
            }
            workgroupBarrier();
            if lane == 0u && reduce[0] < length {
                let i = reduce[0];
                flip_bits[i / 32u] = 1u << (i % 32u);
            }
        } else {
            for (var w = lane; w < MAX_SUB_WORDS; w += WORKGROUP) {
                var value = 0u;
                for (var b = 0u; b < 32u; b++) {
                    let i = w * 32u + b;
                    if i < length && flag_at(&used_bits, i) == 0u &&
                        atomicLoad(&implicated[i]) == best {
                        value |= 1u << b;
                    }
                }
                flip_bits[w] = value;
            }
        }
        workgroupBarrier();

        // used carries exactly this iteration's flips, which both clears the
        // previous set and records the new one.
        for (var w = lane; w < MAX_SUB_WORDS; w += WORKGROUP) {
            data_bits[w] ^= flip_bits[w];
            used_bits[w] = flip_bits[w];
        }
        workgroupBarrier();
    }

    // Post-correction syndrome: a nonzero weight means the corrector gave up
    // and the stream is garbage of the right length.
    if lane == 0u {
        atomicStore(&unsatisfied, 0u);
    }
    workgroupBarrier();
    for (var j = lane; j < rank; j += WORKGROUP) {
        if row_parity(j, degree) == 1u {
            atomicAdd(&unsatisfied, 1u);
        }
    }
    workgroupBarrier();
    if lane == 0u {
        atomicStore(&net[block], select(1u, 0u, atomicLoad(&unsatisfied) == 0u));
    }

    // Compact this block's systematic message bits, which is where the host
    // expects the net payload of every block to land.
    let net_len = params[PARAM_NET];
    let out = params[PARAM_BLOCKS] + block * net_len;
    for (var i = lane; i < net_len; i += WORKGROUP) {
        atomicStore(&net[out + i], bit_at(rank + i));
    }
}
