// Merges directional finder candidates into the accumulated pattern list, the
// same fold saveFinderPattern performs on the host.
//
// The fold cannot be parallelized over candidates. A candidate merges into the
// first accumulated entry within one module size of it, and merging moves that
// entry's centre, so which entry the next candidate matches depends on every
// merge before it. The outer loop is therefore sequential and the whole stage
// is one workgroup. What does parallelize is the search inside each step: the
// scan for a matching entry runs across the lanes and reduces to the lowest
// matching index, which is the one the host's first-match loop would have
// taken.
//
// Candidates arrive in the order the caller fixed. This kernel does not order
// them and must not: the sequence is what makes the result reproducible, and
// choosing it here would hide that.
//
// Survivors and contextual seeds arrive in one stream because the host consumes
// them that way: a seed's place in the weak list depends on how many survivors
// preceded it, and the rule that stops consumption counts both.

const MAX_PATTERNS: u32 = 500u;
const WORKGROUP: u32 = 256u;

// A candidate and an accumulated pattern are both six words. The candidate's
// flags word says which of the two lists it belongs in; everything else about
// its admission was decided where the evidence was.
const CAND_WORDS: u32 = 6u;
const CAND_FLAGS: u32 = 0u;
const CAND_TYP: u32 = 1u;
const CAND_DIRECTION: u32 = 2u;
const CAND_X: u32 = 3u;
const CAND_Y: u32 = 4u;
const CAND_MODULE: u32 = 5u;

const PAT_WORDS: u32 = 6u;
const PAT_TYP: u32 = 0u;
const PAT_DIRECTION: u32 = 1u;
const PAT_X: u32 = 2u;
const PAT_Y: u32 = 3u;
const PAT_MODULE: u32 = 4u;
const PAT_FOUND: u32 = 5u;

const FLAG_SURVIVOR: u32 = 1u << 4u;

const PARAM_COUNT: u32 = 0u;
// The accumulated total at which the consumer stops taking candidates at all,
// and how many weak seeds it keeps. Both are the host's own bounds and are
// passed in rather than restated here, so there is one definition of each.
const PARAM_STOP: u32 = 4u;
const PARAM_WEAK_CAPACITY: u32 = 5u;

// The pattern count, one count per finder type, how many candidates found no
// slot, the weak list's length, how far into the sequence consumption got, and
// the per-type survivor counts the pass stats carry. A drop means the fold was
// asked to hold more distinct patterns than its list can, and silently keeping
// the first 500 would look like a successful fold.
const RECORD_TOTAL: u32 = 0u;
const RECORD_TYPE_COUNT: u32 = 1u;
const RECORD_DROPPED: u32 = 5u;
const RECORD_WEAK_TOTAL: u32 = 6u;
const RECORD_CONSUMED: u32 = 7u;
const RECORD_CROSS_SURVIVORS: u32 = 8u;
const RECORD_WORDS: u32 = 16u;

@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> candidates: array<u32>;
@group(0) @binding(2) var<storage, read_write> patterns: array<u32>;
@group(0) @binding(3) var<storage, read_write> record: array<u32>;
@group(0) @binding(4) var<storage, read_write> weak: array<u32>;

// matched is the lowest accumulated index this candidate merges into, or
// MAX_PATTERNS when it merges into none and has to be appended.
var<workgroup> matched: atomic<u32>;
var<workgroup> total: u32;
var<workgroup> weak_total: u32;

fn pattern_f32(index: u32, field: u32) -> f32 {
    return bitcast<f32>(patterns[index * PAT_WORDS + field]);
}

fn set_pattern_f32(index: u32, field: u32, value: f32) {
    patterns[index * PAT_WORDS + field] = bitcast<u32>(value);
}

fn candidate_f32(index: u32, field: u32) -> f32 {
    return bitcast<f32>(candidates[index * CAND_WORDS + field]);
}

// merges reproduces the host predicate exactly, including its asymmetry: the
// centre tolerance is the incoming candidate's module size while the module
// tolerance is the accumulated entry's. Making those the same value is the
// obvious simplification and it is wrong.
fn merges(slot: u32, typ: u32, cx: f32, cy: f32, ms: f32) -> bool {
    if patterns[slot * PAT_WORDS + PAT_FOUND] == 0u || patterns[slot * PAT_WORDS + PAT_TYP] != typ {
        return false;
    }
    if abs(cx - pattern_f32(slot, PAT_X)) > ms || abs(cy - pattern_f32(slot, PAT_Y)) > ms {
        return false;
    }
    let slot_ms = pattern_f32(slot, PAT_MODULE);
    let delta = abs(ms - slot_ms);
    return delta <= slot_ms || delta <= 1.0;
}

@compute @workgroup_size(WORKGROUP)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    if lane == 0u {
        total = 0u;
        weak_total = 0u;
        for (var word = 0u; word < RECORD_WORDS; word += 1u) {
            record[word] = 0u;
        }
    }
    workgroupBarrier();

    let count = params[PARAM_COUNT];
    let stop = params[PARAM_STOP];
    let weak_capacity = params[PARAM_WEAK_CAPACITY];
    var consumed = 0u;
    for (var c = 0u; c < count; c += 1u) {
        // Once the accumulated list reaches the consumer's stop the sequence is
        // abandoned where it stands, seeds included. The loop still runs to the
        // end rather than breaking out of it: the body carries workgroup
        // barriers, and every lane has to reach every one of them.
        let active = stop == 0u || total < stop;
        let survivor = (candidates[c * CAND_WORDS + CAND_FLAGS] & FLAG_SURVIVOR) != 0u;
        let typ = candidates[c * CAND_WORDS + CAND_TYP];
        let cx = candidate_f32(c, CAND_X);
        let cy = candidate_f32(c, CAND_Y);
        let ms = candidate_f32(c, CAND_MODULE);

        if lane == 0u {
            atomicStore(&matched, MAX_PATTERNS);
        }
        workgroupBarrier();
        // A seed never merges and never grows the pattern list, so it takes
        // none of this search.
        if active && survivor {
            for (var slot = lane; slot < total; slot += WORKGROUP) {
                if merges(slot, typ, cx, cy, ms) {
                    atomicMin(&matched, slot);
                }
            }
        }
        workgroupBarrier();

        if lane == 0u && active {
            consumed += 1u;
        }
        if lane == 0u && active && !survivor {
            if weak_total < weak_capacity {
                for (var word = 0u; word < CAND_WORDS; word += 1u) {
                    weak[weak_total * CAND_WORDS + word] = candidates[c * CAND_WORDS + word];
                }
                weak_total += 1u;
            }
        }
        if lane == 0u && active && survivor {
            if typ < 4u {
                record[RECORD_CROSS_SURVIVORS + typ] += 1u;
            }
            let slot = atomicLoad(&matched);
            if slot < total {
                let found = f32(patterns[slot * PAT_WORDS + PAT_FOUND]);
                set_pattern_f32(slot, PAT_X, (found * pattern_f32(slot, PAT_X) + cx) / (found + 1.0));
                set_pattern_f32(slot, PAT_Y, (found * pattern_f32(slot, PAT_Y) + cy) / (found + 1.0));
                set_pattern_f32(slot, PAT_MODULE, (found * pattern_f32(slot, PAT_MODULE) + ms) / (found + 1.0));
                patterns[slot * PAT_WORDS + PAT_FOUND] += 1u;
                patterns[slot * PAT_WORDS + PAT_DIRECTION] += candidates[c * CAND_WORDS + CAND_DIRECTION];
            } else if total < MAX_PATTERNS {
                patterns[total * PAT_WORDS + PAT_TYP] = typ;
                patterns[total * PAT_WORDS + PAT_DIRECTION] = candidates[c * CAND_WORDS + CAND_DIRECTION];
                set_pattern_f32(total, PAT_X, cx);
                set_pattern_f32(total, PAT_Y, cy);
                set_pattern_f32(total, PAT_MODULE, ms);
                patterns[total * PAT_WORDS + PAT_FOUND] = 1u;
                total += 1u;
                if typ < 4u {
                    record[RECORD_TYPE_COUNT + typ] += 1u;
                }
            } else {
                record[RECORD_DROPPED] += 1u;
            }
        }
        workgroupBarrier();
    }

    if lane == 0u {
        record[RECORD_TOTAL] = total;
        record[RECORD_WEAK_TOTAL] = weak_total;
        record[RECORD_CONSUMED] = consumed;
    }
}
