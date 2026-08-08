// Orders directional finder candidates the way the fold after it needs them:
// by centre row, then centre column, then module size.
//
// The order is not cosmetic. The fold merges a candidate into the first
// accumulated entry it matches and moves that entry in doing so, so the whole
// merged pattern set is a function of this sequence. The compaction that
// produced these candidates reserved its output slots through an atomic whose
// ordering is unspecified, which is why an order has to be imposed at all.
//
// It is a bitonic sort in a single workgroup, over the storage buffer rather
// than through shared memory. The sequence can reach tens of thousands of
// records, far past what fits in workgroup storage, and one workgroup is the
// only scope in which WGSL can synchronize storage writes at all - a
// multi-workgroup network would need a dispatch per stage and a barrier
// between each.
//
// Records past the count are given an infinite key so they sort to the end and
// the network can assume a power-of-two length. That is why the buffer holds a
// power of two rather than the compaction's own capacity.

const WORKGROUP: u32 = 256u;
const CAND_WORDS: u32 = 6u;
const CAND_X: u32 = 3u;
const CAND_Y: u32 = 4u;
const CAND_MODULE: u32 = 5u;

const PARAM_COUNT: u32 = 0u;
const PARAM_PADDED: u32 = 1u;

@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read_write> candidates: array<u32>;

fn field(index: u32, at: u32) -> f32 {
    return bitcast<f32>(candidates[index * CAND_WORDS + at]);
}

// before reports whether record a sorts ahead of record b. Exact ties in all
// three keys are left to the network, which resolves them the same way every
// run; the host's own sort leaves them unspecified.
fn before(a: u32, b: u32) -> bool {
    let ay = field(a, CAND_Y);
    let by = field(b, CAND_Y);
    if ay != by { return ay < by; }
    let ax = field(a, CAND_X);
    let bx = field(b, CAND_X);
    if ax != bx { return ax < bx; }
    return field(a, CAND_MODULE) < field(b, CAND_MODULE);
}

fn exchange(a: u32, b: u32) {
    for (var word = 0u; word < CAND_WORDS; word += 1u) {
        let tmp = candidates[a * CAND_WORDS + word];
        candidates[a * CAND_WORDS + word] = candidates[b * CAND_WORDS + word];
        candidates[b * CAND_WORDS + word] = tmp;
    }
}

@compute @workgroup_size(WORKGROUP)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let count = params[PARAM_COUNT];
    let padded = params[PARAM_PADDED];
    let infinity = bitcast<f32>(0x7f800000u);

    for (var i = count + lid.x; i < padded; i += WORKGROUP) {
        candidates[i * CAND_WORDS + CAND_X] = bitcast<u32>(infinity);
        candidates[i * CAND_WORDS + CAND_Y] = bitcast<u32>(infinity);
        candidates[i * CAND_WORDS + CAND_MODULE] = bitcast<u32>(infinity);
    }
    storageBarrier();

    for (var k = 2u; k <= padded; k = k << 1u) {
        for (var j = k >> 1u; j > 0u; j = j >> 1u) {
            for (var i = lid.x; i < padded; i += WORKGROUP) {
                let partner = i ^ j;
                if partner > i {
                    // Each i belongs to a run of length k that the network
                    // builds ascending or descending in alternation; the bit of
                    // i at k selects which.
                    let ascending = (i & k) == 0u;
                    if ascending == before(partner, i) {
                        exchange(i, partner);
                    }
                }
            }
            storageBarrier();
        }
    }
}
