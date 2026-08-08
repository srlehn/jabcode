// Turns one direction's compacted chain outcomes into the candidate stream the
// ordering and fold stages read, so the fold takes its input from where the
// chain already left it instead of from a list the host downloaded and sent
// back.
//
// The admission is consumeDirectionalFamilyOutcomes': a record enters only if
// the chain marked it a survivor or a contextual seed, and an FP1 or FP2
// candidate additionally needs the colour verdict. Both kinds enter the same
// stream, because the host consumes them interleaved in one sorted order and
// the fold's stopping rule counts them together.
//
// A candidate whose colour the chain never evaluated cannot be judged here at
// all - deciding it means reading source RGB - so it is counted and dropped,
// and the count is what tells the caller this route cannot answer for this
// direction.
//
// Lanes take contiguous blocks and claim output space through a prefix sum over
// their own counts, rather than reserving slots with an atomic. What that buys
// is reproducibility, not agreement with the host: the ordering stage after this
// one imposes a total order on centre and module size, so the sequence here
// decides only records whose three sort keys are exactly equal. That is a narrow
// case, and it is still worth a scan over 256 counters - an atomic would make
// the same frame fold differently from one run to the next, which is the
// property the ordering stage exists to remove.

const WORKGROUP: u32 = 256u;

// An outcome and a candidate share one six-word layout: flags, type, direction,
// then centre and module size as f32. The copy is a straight one because of it.
const RECORD_WORDS: u32 = 6u;
const RECORD_FLAGS: u32 = 0u;
const RECORD_TYP: u32 = 1u;

const FLAG_SURVIVOR: u32 = 1u << 4u;
const FLAG_CONTEXTUAL_SEED: u32 = 1u << 5u;
const FLAG_COLOR_EVALUATED: u32 = 1u << 6u;
const FLAG_COLOR_OK: u32 = 1u << 7u;

const FP1: u32 = 1u;
const FP2: u32 = 2u;

const PARAM_BASE: u32 = 0u;
const PARAM_COUNT: u32 = 1u;

const FOLD_PARAM_COUNT: u32 = 0u;
const FOLD_PARAM_PADDED: u32 = 1u;

const ASSEMBLY_COUNT: u32 = 0u;
const ASSEMBLY_DEFERRED: u32 = 1u;
const ASSEMBLY_WORDS: u32 = 4u;

const VERDICT_REJECT: u32 = 0u;
const VERDICT_ADMIT: u32 = 1u;
const VERDICT_DEFER: u32 = 2u;

@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> outcomes: array<u32>;
@group(0) @binding(2) var<storage, read_write> candidates: array<u32>;
@group(0) @binding(3) var<storage, read_write> fold_params: array<u32>;
@group(0) @binding(4) var<storage, read_write> record: array<u32>;

var<workgroup> lane_admitted: array<u32, WORKGROUP>;
var<workgroup> lane_deferred: array<u32, WORKGROUP>;

fn verdict(at: u32) -> u32 {
    let flags = outcomes[at + RECORD_FLAGS];
    if (flags & (FLAG_SURVIVOR | FLAG_CONTEXTUAL_SEED)) == 0u {
        return VERDICT_REJECT;
    }
    let typ = outcomes[at + RECORD_TYP];
    if typ == FP1 || typ == FP2 {
        if (flags & FLAG_COLOR_EVALUATED) == 0u {
            return VERDICT_DEFER;
        }
        if (flags & FLAG_COLOR_OK) == 0u {
            return VERDICT_REJECT;
        }
    }
    return VERDICT_ADMIT;
}

@compute @workgroup_size(WORKGROUP)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let base = params[PARAM_BASE];
    let count = params[PARAM_COUNT];

    let chunk = (count + WORKGROUP - 1u) / WORKGROUP;
    let start = min(lane * chunk, count);
    let end = min(start + chunk, count);

    var admitted = 0u;
    var deferred = 0u;
    for (var i = start; i < end; i += 1u) {
        let v = verdict((base + i) * RECORD_WORDS);
        if v == VERDICT_ADMIT {
            admitted += 1u;
        } else if v == VERDICT_DEFER {
            deferred += 1u;
        }
    }
    lane_admitted[lane] = admitted;
    lane_deferred[lane] = deferred;
    workgroupBarrier();

    // Inclusive scan over the per-lane counts; each lane's own count subtracted
    // back off gives it the first output slot it owns.
    for (var offset = 1u; offset < WORKGROUP; offset = offset << 1u) {
        var carry = 0u;
        if lane >= offset {
            carry = lane_admitted[lane - offset];
        }
        workgroupBarrier();
        lane_admitted[lane] += carry;
        workgroupBarrier();
    }
    var at = lane_admitted[lane] - admitted;

    for (var i = start; i < end; i += 1u) {
        let src = (base + i) * RECORD_WORDS;
        if verdict(src) != VERDICT_ADMIT {
            continue;
        }
        let dst = at * RECORD_WORDS;
        for (var word = 0u; word < RECORD_WORDS; word += 1u) {
            candidates[dst + word] = outcomes[src + word];
        }
        at += 1u;
    }

    if lane == 0u {
        let total = lane_admitted[WORKGROUP - 1u];
        var padded = 1u;
        while padded < total {
            padded = padded << 1u;
        }
        fold_params[FOLD_PARAM_COUNT] = total;
        fold_params[FOLD_PARAM_PADDED] = padded;
        for (var word = 0u; word < ASSEMBLY_WORDS; word += 1u) {
            record[word] = 0u;
        }
        record[ASSEMBLY_COUNT] = total;
        var undecided = 0u;
        for (var i = 0u; i < WORKGROUP; i += 1u) {
            undecided += lane_deferred[i];
        }
        record[ASSEMBLY_DEFERRED] = undecided;
    }
}
