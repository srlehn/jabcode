// Merges one direction's finder patterns into a pool that outlives it.
//
// This is the accumulation accumulateFamilyCandidates and
// accumulateContextualFinderCandidates both perform. They are the same
// algorithm - the same match predicate, keep the better-supported entry,
// otherwise append - and differ only in what they are given and how much of it
// they will hold, so both arrive here.
//
// Two merges live here because the same traversal serves both. Across
// directions a pool replaces a matched entry outright when the newcomer is
// better supported and leaves it alone otherwise: averaging there would blend
// two views of one finder taken from different bases into a centre neither of
// them saw. Within one direction, grouping the weak seeds is saveFinderPattern
// itself, running average and all, which is why the fold's arithmetic appears
// again under the averaging mode. That mode weights the incoming entry as one
// crossing, which is what a seed is; feeding it accumulated groups would count
// their support as one.
//
// The outer loop is sequential for the fold's reason: a replacement moves the
// entry it replaced, so which entry the next candidate matches depends on every
// decision before it. The inner scan runs across the lanes and reduces to the
// lowest matching index, which is the entry the host's first-match loop takes.
//
// The pool is resumed rather than rebuilt: its running total comes back out of
// its own record, so a level's directions accumulate into it one dispatch at a
// time.

const WORKGROUP: u32 = 256u;

const PAT_WORDS: u32 = 6u;
const PAT_TYP: u32 = 0u;
const PAT_DIRECTION: u32 = 1u;
const PAT_X: u32 = 2u;
const PAT_Y: u32 = 3u;
const PAT_MODULE: u32 = 4u;
const PAT_FOUND: u32 = 5u;

const PARAM_CAPACITY: u32 = 0u;
const PARAM_MIN_FOUND: u32 = 1u;
const PARAM_COUNT_WORD: u32 = 2u;
const PARAM_MODE: u32 = 3u;

const MODE_REPLACE: u32 = 0u;
const MODE_AVERAGE: u32 = 1u;

// The pool's length, how many entries it had no room for, and a bit per finder
// type present. The type mask is what the selection's prune reads, and building
// it here keeps the pool's contents from having to cross the bus to answer a
// four-bit question.
const POOL_COUNT: u32 = 0u;
const POOL_DROPPED: u32 = 1u;
const POOL_TYPE_MASK: u32 = 2u;
const POOL_WORDS: u32 = 4u;

@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> source: array<u32>;
@group(0) @binding(2) var<storage, read> source_record: array<u32>;
@group(0) @binding(3) var<storage, read_write> pool: array<u32>;
@group(0) @binding(4) var<storage, read_write> pool_record: array<u32>;

var<workgroup> matched: atomic<u32>;
var<workgroup> total: u32;

fn pool_f32(index: u32, field: u32) -> f32 {
    return bitcast<f32>(pool[index * PAT_WORDS + field]);
}

fn source_f32(index: u32, field: u32) -> f32 {
    return bitcast<f32>(source[index * PAT_WORDS + field]);
}

// merges is the fold's predicate, including its asymmetry: the centre tolerance
// is the incoming candidate's module size while the module tolerance is the
// pooled entry's.
fn merges(slot: u32, typ: u32, cx: f32, cy: f32, ms: f32) -> bool {
    if pool[slot * PAT_WORDS + PAT_TYP] != typ {
        return false;
    }
    if abs(cx - pool_f32(slot, PAT_X)) > ms || abs(cy - pool_f32(slot, PAT_Y)) > ms {
        return false;
    }
    let slot_ms = pool_f32(slot, PAT_MODULE);
    let delta = abs(ms - slot_ms);
    return delta <= slot_ms || delta <= 1.0;
}

@compute @workgroup_size(WORKGROUP)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let capacity = params[PARAM_CAPACITY];
    let min_found = params[PARAM_MIN_FOUND];
    let count = source_record[params[PARAM_COUNT_WORD]];
    let mode = params[PARAM_MODE];
    if lane == 0u {
        total = pool_record[POOL_COUNT];
    }
    workgroupBarrier();

    for (var c = 0u; c < count; c += 1u) {
        let found = source[c * PAT_WORDS + PAT_FOUND];
        // The support floor is the only filter: the contextual pool takes
        // grouped seeds only, the family pool takes everything it is given.
        let admitted = found >= min_found && found > 0u;
        let typ = source[c * PAT_WORDS + PAT_TYP];
        let cx = source_f32(c, PAT_X);
        let cy = source_f32(c, PAT_Y);
        let ms = source_f32(c, PAT_MODULE);

        if lane == 0u {
            atomicStore(&matched, capacity);
        }
        workgroupBarrier();
        if admitted {
            for (var slot = lane; slot < total; slot += WORKGROUP) {
                if merges(slot, typ, cx, cy, ms) {
                    atomicMin(&matched, slot);
                }
            }
        }
        workgroupBarrier();

        if lane == 0u && admitted {
            let slot = atomicLoad(&matched);
            if slot < total {
                if mode == MODE_AVERAGE {
                    let held = f32(pool[slot * PAT_WORDS + PAT_FOUND]);
                    pool[slot * PAT_WORDS + PAT_X] =
                        bitcast<u32>((held * pool_f32(slot, PAT_X) + cx) / (held + 1.0));
                    pool[slot * PAT_WORDS + PAT_Y] =
                        bitcast<u32>((held * pool_f32(slot, PAT_Y) + cy) / (held + 1.0));
                    pool[slot * PAT_WORDS + PAT_MODULE] =
                        bitcast<u32>((held * pool_f32(slot, PAT_MODULE) + ms) / (held + 1.0));
                    pool[slot * PAT_WORDS + PAT_FOUND] += 1u;
                    pool[slot * PAT_WORDS + PAT_DIRECTION] +=
                        source[c * PAT_WORDS + PAT_DIRECTION];
                } else if found > pool[slot * PAT_WORDS + PAT_FOUND] {
                    for (var word = 0u; word < PAT_WORDS; word += 1u) {
                        pool[slot * PAT_WORDS + word] = source[c * PAT_WORDS + word];
                    }
                }
            } else if total < capacity {
                for (var word = 0u; word < PAT_WORDS; word += 1u) {
                    pool[total * PAT_WORDS + word] = source[c * PAT_WORDS + word];
                }
                total += 1u;
                if typ < 4u {
                    pool_record[POOL_TYPE_MASK] |= 1u << typ;
                }
            } else {
                // The host pool grows without a bound, so a full one here is a
                // pool that no longer answers the same question. Reported rather
                // than absorbed.
                pool_record[POOL_DROPPED] += 1u;
            }
        }
        workgroupBarrier();
    }

    if lane == 0u {
        pool_record[POOL_COUNT] = total;
    }
}
