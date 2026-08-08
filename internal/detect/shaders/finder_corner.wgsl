// Completes a three-corner selection with the fourth, where the candidates that
// could confirm it already are.
//
// This is interpolateMissingPattern followed by pickPooledCorner. The
// interpolation is closed form: three corners determine the fourth exactly under
// any affine transform, so a frontal capture leaves only centre noise. The
// search that follows exists because captures are not always affine, and it
// prefers a corner the scan actually found over an exact construction only when
// that candidate agrees with the partial quad on every side at once.
//
// When no pooled candidate qualifies, the same quad is offered to the
// contextual pool: candidates that cleared branch and colour classification and
// were crossed repeatedly, but failed the standalone cross-check. Those become
// ranked hypotheses rather than a chosen corner, because the strict chain stays
// the admission boundary and the complete quad's geometry is checked before any
// of them is sampled.
//
// The local seek is not here. It reads source pixels rather than any of the
// lists on this device, and it is the last of the four outcomes.
//
// The pool scan runs across the lanes because the pool can be tens of thousands
// of entries and each one costs a whole quad score. The host takes the first
// candidate at the best score, so the reduction is exact rather than by
// argument: the lanes agree on the score first, then on the lowest index that
// reaches it. Scores are non-negative by construction, which is what lets the
// first of those be an integer minimum over their bit patterns.
//
// There are no early returns. Every path runs to the same barriers, because a
// lane that left before one would leave the rest waiting on it.

const WORKGROUP: u32 = 256u;

const PAT_WORDS: u32 = 6u;
const PAT_TYP: u32 = 0u;
const PAT_DIRECTION: u32 = 1u;
const PAT_X: u32 = 2u;
const PAT_Y: u32 = 3u;
const PAT_MODULE: u32 = 4u;
const PAT_FOUND: u32 = 5u;

const MIN_CROSSINGS: u32 = 3u;
const POOLED_CORNER_NOISE_MODULES: f32 = 2.0;
const QUAD_EDGE_TOL: f32 = 1.35;
const QUAD_MODULE_TOL: f32 = 1.6;
const QUAD_CONSISTENCY_TOL: f32 = 1.4;

const PARAM_WIDTH: u32 = 0u;
const PARAM_HEIGHT: u32 = 1u;

const SEL_MISSING: u32 = 12u;
const SEL_PATTERNS: u32 = 16u;

const POOL_COUNT: u32 = 0u;

// The corner's provenance, which of the four it is, whether the completion is
// usable at all, and the pattern itself. The source values are CornerSource's.
const CORNER_SOURCE: u32 = 0u;
const CORNER_MISS: u32 = 1u;
const CORNER_OK: u32 = 2u;
const CORNER_ALTERNATIVES: u32 = 3u;
const CORNER_PATTERN: u32 = 4u;
const CORNER_ALTERNATIVE_PATTERNS: u32 = 16u;
const CORNER_WORDS: u32 = 64u;

const MAX_ALTERNATIVES: u32 = 8u;

const SOURCE_FOUND: u32 = 0u;
const SOURCE_CONSTRUCTED: u32 = 1u;
const SOURCE_POOLED: u32 = 2u;

const NONE: u32 = 4u;
const NO_INDEX: u32 = 0xffffffffu;

@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> selection: array<u32>;
@group(0) @binding(2) var<storage, read> pool: array<u32>;
@group(0) @binding(3) var<storage, read> pool_record: array<u32>;
@group(0) @binding(4) var<storage, read_write> corner: array<u32>;
@group(0) @binding(5) var<storage, read> contextual: array<u32>;
@group(0) @binding(6) var<storage, read> contextual_record: array<u32>;

// The four selected patterns with the missing one filled in, so the pool scan
// and the record both read one place.
var<workgroup> qx: array<f32, 4>;
var<workgroup> qy: array<f32, 4>;
var<workgroup> qms: array<f32, 4>;
var<workgroup> qtyp: array<u32, 4>;
var<workgroup> qdir: array<i32, 4>;
var<workgroup> qfound: array<u32, 4>;
var<workgroup> miss: u32;
var<workgroup> searching: bool;
var<workgroup> span: f32;
var<workgroup> best_score: atomic<u32>;
var<workgroup> best_index: atomic<u32>;
var<workgroup> lane_score: array<f32, WORKGROUP>;
var<workgroup> lane_index: array<u32, WORKGROUP>;
var<workgroup> chosen: array<u32, MAX_ALTERNATIVES>;
var<workgroup> chosen_count: u32;
var<workgroup> ranking: bool;

fn point_distance(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    return sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by));
}

fn infinity() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

// ratio is the larger over the smaller, or infinity when either is
// non-positive, exactly as the host's gates read it.
fn ratio(a: f32, b: f32) -> f32 {
    if a <= 0.0 || b <= 0.0 {
        return infinity();
    }
    return max(a, b) / min(a, b);
}

fn convex(px: array<f32, 4>, py: array<f32, 4>) -> bool {
    var orientation = 0.0;
    for (var i = 0u; i < 4u; i += 1u) {
        let j = (i + 1u) & 3u;
        let k = (i + 2u) & 3u;
        let cross = (px[j] - px[i]) * (py[k] - py[j]) - (py[j] - py[i]) * (px[k] - px[j]);
        if cross == 0.0 {
            return false;
        }
        if i == 0u {
            orientation = cross;
        } else if (cross > 0.0) != (orientation > 0.0) {
            return false;
        }
    }
    return true;
}

fn edge_module_span(ax: f32, ay: f32, ams: f32, bx: f32, by: f32, bms: f32) -> f32 {
    if ams <= 0.0 || bms <= 0.0 {
        return -1.0;
    }
    let d = point_distance(ax, ay, bx, by);
    if d <= 0.0 {
        return -1.0;
    }
    return d / ((ams + bms) / 2.0);
}

// side_size rounds a raw module count to the nearest legal side and reports how
// reliable that rounding is: 1 settled, 0 guessed, -1 no legal side. It is
// arithmetic rather than a table, which is what lets the whole quad score run
// here.
fn side_size(size: i32) -> vec2<i32> {
    var rounded = size;
    var flag = 1;
    switch rounded & 3 {
        case 0: { rounded += 1; }
        case 2: { rounded -= 1; }
        case 3: { rounded -= 2; flag = 0; }
        default: {}
    }
    if rounded < 21 || rounded > 145 {
        return vec2<i32>(-1, -1);
    }
    return vec2<i32>(rounded, flag);
}

fn choose_side_size(a: vec2<i32>, b: vec2<i32>) -> i32 {
    if a.y == -1 && b.y == -1 {
        return -1;
    }
    if a.y == b.y {
        return max(a.x, b.x);
    }
    if a.y > b.y {
        return a.x;
    }
    return b.x;
}

fn side_from_spans(a: f32, b: f32) -> i32 {
    return choose_side_size(side_size(i32(a + 0.5) + 7), side_size(i32(b + 0.5) + 7));
}

// score_quad is ScoreFinderQuad: every geometric gate, then a badness score
// that is zero for an ideal quad. The x component is the score and the y
// component says whether it passed the gates at all.
fn score_quad(px: array<f32, 4>, py: array<f32, 4>, ms: array<f32, 4>) -> vec2<f32> {
    let fail = vec2<f32>(0.0, 0.0);
    if !convex(px, py) {
        return fail;
    }
    let top = point_distance(px[0], py[0], px[1], py[1]);
    let right = point_distance(px[1], py[1], px[2], py[2]);
    let bot = point_distance(px[2], py[2], px[3], py[3]);
    let left = point_distance(px[3], py[3], px[0], py[0]);
    let edge_dev = max(ratio(top, bot), ratio(left, right));
    if edge_dev > QUAD_EDGE_TOL {
        return fail;
    }
    let ms_min = min(min(ms[0], ms[1]), min(ms[2], ms[3]));
    let ms_max = max(max(ms[0], ms[1]), max(ms[2], ms[3]));
    if ms_min <= 0.0 || ms_max / ms_min > QUAD_MODULE_TOL {
        return fail;
    }
    let span_top = edge_module_span(px[0], py[0], ms[0], px[1], py[1], ms[1]);
    let span_right = edge_module_span(px[1], py[1], ms[1], px[2], py[2], ms[2]);
    let span_bottom = edge_module_span(px[3], py[3], ms[3], px[2], py[2], ms[2]);
    let span_left = edge_module_span(px[0], py[0], ms[0], px[3], py[3], ms[3]);
    if min(min(span_top, span_right), min(span_bottom, span_left)) <= 0.0 {
        return fail;
    }
    let side_x = side_from_spans(span_top, span_bottom);
    let side_y = side_from_spans(span_left, span_right);
    if side_x <= 0 || side_y <= 0 {
        return fail;
    }
    // Finder centres sit 3.5 modules inside each edge, so their span is side-7.
    // Each edge uses its own endpoint scales rather than one four-corner mean.
    let want_x = f32(side_x - 7);
    let want_y = f32(side_y - 7);
    let consist = max(
        max(ratio(span_top, want_x), ratio(span_bottom, want_x)),
        max(ratio(span_left, want_y), ratio(span_right, want_y)),
    );
    if consist > QUAD_CONSISTENCY_TOL {
        return fail;
    }
    return vec2<f32>((edge_dev - 1.0) + (ms_max / ms_min - 1.0) + (consist - 1.0), 1.0);
}

// interpolate completes the missing corner as a parallelogram whose offset is
// rescaled from the far edge's module sizes onto the near one's. The four cases
// are written out because that is how the host writes them: which corner the
// offset is taken from and which pair scales it differ per corner, and folding
// them into one expression hides that.
fn interpolate() {
    let m = miss;
    var from = 0u;
    var head = 0u;
    var tail = 0u;
    var far_a = 0u;
    var far_b = 0u;
    var near_a = 0u;
    var near_b = 0u;
    if m == 0u {
        head = 3u; tail = 2u; from = 1u;
        far_a = 2u; far_b = 3u; near_a = 1u; near_b = 3u;
    } else if m == 1u {
        head = 2u; tail = 3u; from = 0u;
        far_a = 2u; far_b = 3u; near_a = 0u; near_b = 2u;
    } else if m == 2u {
        head = 1u; tail = 0u; from = 3u;
        far_a = 0u; far_b = 1u; near_a = 1u; near_b = 3u;
    } else {
        head = 0u; tail = 1u; from = 2u;
        far_a = 0u; far_b = 1u; near_a = 0u; near_b = 2u;
    }
    let far_scale = (qms[far_a] + qms[far_b]) / 2.0;
    let near_scale = (qms[near_a] + qms[near_b]) / 2.0;
    qx[m] = (qx[head] - qx[tail]) / far_scale * near_scale + qx[from];
    qy[m] = (qy[head] - qy[tail]) / far_scale * near_scale + qy[from];
    // The two lower corners inherit their neighbour's scan direction and the
    // two upper ones invert it, which is the host's own asymmetry.
    if m == 0u || m == 1u {
        qdir[m] = -qdir[from];
    } else {
        qdir[m] = qdir[from];
    }
    qtyp[m] = m;
    qfound[m] = 1u;
    var total = 0.0;
    for (var i = 0u; i < 4u; i += 1u) {
        if i != m {
            total += qms[i];
        }
    }
    qms[m] = total / 3.0;
}

// pooled_radius is pooledCornerRadius: centre noise plus what completing the
// parallelogram overshoots by when the three present corners disagree on scale.
fn pooled_radius() -> f32 {
    var ms_min = infinity();
    var ms_max = 0.0;
    for (var i = 0u; i < 4u; i += 1u) {
        if i != miss {
            ms_min = min(ms_min, qms[i]);
            ms_max = max(ms_max, qms[i]);
        }
    }
    if ms_min <= 0.0 || qms[miss] <= 0.0 {
        return 0.0;
    }
    let edge = max(
        point_distance(qx[miss], qy[miss], qx[(miss + 1u) & 3u], qy[(miss + 1u) & 3u]),
        point_distance(qx[miss], qy[miss], qx[(miss + 3u) & 3u], qy[(miss + 3u) & 3u]),
    );
    let skew = ms_max / ms_min;
    return POOLED_CORNER_NOISE_MODULES * qms[miss] + edge * (skew - 1.0) / skew;
}

// candidate_score is a pooled candidate's badness, or a negative value when it
// fails any gate. Distance is normalized by the radius so the two disagreements
// are comparable: a nearer candidate at the wrong scale loses to a further one
// the partial quad agrees with.
fn candidate_score(index: u32) -> f32 {
    let at = index * PAT_WORDS;
    if pool[at + PAT_TYP] != miss || pool[at + PAT_FOUND] < MIN_CROSSINGS {
        return -1.0;
    }
    let cms = bitcast<f32>(pool[at + PAT_MODULE]);
    if cms <= 0.0 {
        return -1.0;
    }
    let cx = bitcast<f32>(pool[at + PAT_X]);
    let cy = bitcast<f32>(pool[at + PAT_Y]);
    let off = point_distance(cx, cy, qx[miss], qy[miss]);
    if off > span {
        return -1.0;
    }
    var scale = 1.0;
    for (var i = 0u; i < 4u; i += 1u) {
        if i != miss {
            scale = max(scale, ratio(cms, qms[i]));
        }
    }
    if scale > QUAD_MODULE_TOL {
        return -1.0;
    }
    var px = array<f32, 4>(qx[0], qx[1], qx[2], qx[3]);
    var py = array<f32, 4>(qy[0], qy[1], qy[2], qy[3]);
    var ms = array<f32, 4>(qms[0], qms[1], qms[2], qms[3]);
    px[miss] = cx;
    py[miss] = cy;
    ms[miss] = cms;
    if score_quad(px, py, ms).y == 0.0 {
        return -1.0;
    }
    return off / span + (scale - 1.0);
}

// alternative_score is the completed quad's badness with this contextual
// candidate in the missing corner, or a negative value when it fails a gate.
// There is no proximity gate here, unlike the pooled search: these are ranked
// hypotheses for the caller to try in order, not a corner chosen outright, so
// the quad's own geometry is the whole of the filter.
fn alternative_score(index: u32) -> f32 {
    let at = index * PAT_WORDS;
    if contextual[at + PAT_TYP] != miss || contextual[at + PAT_FOUND] < MIN_CROSSINGS {
        return -1.0;
    }
    let cms = bitcast<f32>(contextual[at + PAT_MODULE]);
    if cms <= 0.0 {
        return -1.0;
    }
    var scale = 1.0;
    for (var i = 0u; i < 4u; i += 1u) {
        if i != miss {
            scale = max(scale, ratio(cms, qms[i]));
        }
    }
    if scale > QUAD_MODULE_TOL {
        return -1.0;
    }
    var px = array<f32, 4>(qx[0], qx[1], qx[2], qx[3]);
    var py = array<f32, 4>(qy[0], qy[1], qy[2], qy[3]);
    var ms = array<f32, 4>(qms[0], qms[1], qms[2], qms[3]);
    px[miss] = bitcast<f32>(contextual[at + PAT_X]);
    py[miss] = bitcast<f32>(contextual[at + PAT_Y]);
    ms[miss] = cms;
    let scored = score_quad(px, py, ms);
    if scored.y == 0.0 {
        return -1.0;
    }
    // A passing quad can score exactly zero, and a negative value is how this
    // reports a rejection, so the score is offset to keep the two apart.
    return scored.x + 1.0;
}

// ranks_before is the host's ordering: cheapest quad first, then the
// better-supported candidate, then the topmost and leftmost, and finally the
// earlier pool entry. The last key is what the host's stable sort gives for
// free, and it is what makes this ordering a function of the pool alone.
fn ranks_before(sa: f32, ia: u32, sb: f32, ib: u32) -> bool {
    if ib == NO_INDEX {
        return true;
    }
    if ia == NO_INDEX {
        return false;
    }
    if sa != sb {
        return sa < sb;
    }
    let founda = contextual[ia * PAT_WORDS + PAT_FOUND];
    let foundb = contextual[ib * PAT_WORDS + PAT_FOUND];
    if founda != foundb {
        return founda > foundb;
    }
    let ya = bitcast<f32>(contextual[ia * PAT_WORDS + PAT_Y]);
    let yb = bitcast<f32>(contextual[ib * PAT_WORDS + PAT_Y]);
    if ya != yb {
        return ya < yb;
    }
    let xa = bitcast<f32>(contextual[ia * PAT_WORDS + PAT_X]);
    let xb = bitcast<f32>(contextual[ib * PAT_WORDS + PAT_X]);
    if xa != xb {
        return xa < xb;
    }
    return ia < ib;
}

fn already_chosen(index: u32) -> bool {
    for (var i = 0u; i < chosen_count; i += 1u) {
        if chosen[i] == index {
            return true;
        }
    }
    return false;
}

fn write_alternative(slot: u32, index: u32) {
    let at = index * PAT_WORDS;
    let out = CORNER_ALTERNATIVE_PATTERNS + slot * PAT_WORDS;
    corner[out + PAT_TYP] = contextual[at + PAT_TYP];
    // The candidate takes the estimate's scan direction, as the pooled corner
    // does: it is the direction this quad was found along.
    corner[out + PAT_DIRECTION] = bitcast<u32>(qdir[miss]);
    corner[out + PAT_X] = contextual[at + PAT_X];
    corner[out + PAT_Y] = contextual[at + PAT_Y];
    corner[out + PAT_MODULE] = contextual[at + PAT_MODULE];
    corner[out + PAT_FOUND] = contextual[at + PAT_FOUND];
}

fn write_corner() {
    corner[CORNER_PATTERN + PAT_TYP] = qtyp[miss];
    corner[CORNER_PATTERN + PAT_DIRECTION] = bitcast<u32>(qdir[miss]);
    corner[CORNER_PATTERN + PAT_X] = bitcast<u32>(qx[miss]);
    corner[CORNER_PATTERN + PAT_Y] = bitcast<u32>(qy[miss]);
    corner[CORNER_PATTERN + PAT_MODULE] = bitcast<u32>(qms[miss]);
    corner[CORNER_PATTERN + PAT_FOUND] = qfound[miss];
}

@compute @workgroup_size(WORKGROUP)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    if lane == 0u {
        for (var word = 0u; word < CORNER_WORDS; word += 1u) {
            corner[word] = 0u;
        }
        atomicStore(&best_score, NO_INDEX);
        atomicStore(&best_index, NO_INDEX);
        searching = false;
        span = 0.0;
        miss = NONE;
        for (var slot = 0u; slot < 4u; slot += 1u) {
            let at = SEL_PATTERNS + slot * PAT_WORDS;
            qtyp[slot] = selection[at + PAT_TYP];
            qdir[slot] = bitcast<i32>(selection[at + PAT_DIRECTION]);
            qx[slot] = bitcast<f32>(selection[at + PAT_X]);
            qy[slot] = bitcast<f32>(selection[at + PAT_Y]);
            qms[slot] = bitcast<f32>(selection[at + PAT_MODULE]);
            qfound[slot] = selection[at + PAT_FOUND];
        }
        // Only a single absent corner is completed. None means there is nothing
        // to do, and two or more is not recoverable at all - the caller has
        // already read that from the selection.
        if selection[SEL_MISSING] == 1u {
            for (var slot = 0u; slot < 4u; slot += 1u) {
                if qfound[slot] == 0u {
                    miss = slot;
                    break;
                }
            }
        }
        corner[CORNER_MISS] = NONE;
        corner[CORNER_SOURCE] = SOURCE_FOUND;
        if miss != NONE {
            interpolate();
            corner[CORNER_MISS] = miss;
            corner[CORNER_SOURCE] = SOURCE_CONSTRUCTED;
            corner[CORNER_OK] = 1u;
            // An estimate outside the frame is not a corner at all: the
            // direction is finished, and the pattern is reported with no
            // crossings behind it.
            let inside = qx[miss] >= 0.0 && qx[miss] <= f32(params[PARAM_WIDTH] - 1u) &&
                qy[miss] >= 0.0 && qy[miss] <= f32(params[PARAM_HEIGHT] - 1u);
            if inside {
                span = pooled_radius();
                searching = span > 0.0;
            } else {
                qfound[miss] = 0u;
                corner[CORNER_OK] = 0u;
            }
        }
    }
    workgroupBarrier();

    if searching {
        let count = pool_record[POOL_COUNT];
        for (var i = lane; i < count; i += WORKGROUP) {
            let score = candidate_score(i);
            if score >= 0.0 {
                atomicMin(&best_score, bitcast<u32>(score));
            }
        }
    }
    workgroupBarrier();
    if searching && atomicLoad(&best_score) != NO_INDEX {
        let winning = atomicLoad(&best_score);
        let count = pool_record[POOL_COUNT];
        for (var i = lane; i < count; i += WORKGROUP) {
            let score = candidate_score(i);
            if score >= 0.0 && bitcast<u32>(score) == winning {
                atomicMin(&best_index, i);
            }
        }
    }
    workgroupBarrier();

    if lane == 0u && miss != NONE {
        let winner = atomicLoad(&best_index);
        if winner != NO_INDEX {
            let at = winner * PAT_WORDS;
            // The pooled candidate keeps the estimate's scan direction: that is
            // the direction this quad was found along, not the one that first
            // saw the candidate.
            qtyp[miss] = pool[at + PAT_TYP];
            qx[miss] = bitcast<f32>(pool[at + PAT_X]);
            qy[miss] = bitcast<f32>(pool[at + PAT_Y]);
            qms[miss] = bitcast<f32>(pool[at + PAT_MODULE]);
            qfound[miss] = pool[at + PAT_FOUND];
            corner[CORNER_SOURCE] = SOURCE_POOLED;
        }
        write_corner();
        chosen_count = 0u;
        // Alternatives exist for the construction the caller is left holding.
        // A pooled corner is a detection, and a completion that fell outside
        // the frame has nothing to offer them.
        ranking = winner == NO_INDEX && corner[CORNER_OK] != 0u;
    }
    workgroupBarrier();

    // Each round takes the best remaining candidate, so eight rounds give the
    // eight the caller may try. Ranking them by a full sort would mean ordering
    // a list that can hold tens of thousands to keep eight of them.
    for (var round = 0u; round < MAX_ALTERNATIVES; round += 1u) {
        var best = -1.0;
        var index = NO_INDEX;
        if ranking {
            let count = contextual_record[POOL_COUNT];
            for (var i = lane; i < count; i += WORKGROUP) {
                if already_chosen(i) {
                    continue;
                }
                let score = alternative_score(i);
                if score >= 0.0 && ranks_before(score, i, best, index) {
                    best = score;
                    index = i;
                }
            }
        }
        lane_score[lane] = best;
        lane_index[lane] = index;
        workgroupBarrier();
        if lane == 0u && ranking {
            var winning = -1.0;
            var at = NO_INDEX;
            for (var l = 0u; l < WORKGROUP; l += 1u) {
                if lane_index[l] != NO_INDEX && ranks_before(lane_score[l], lane_index[l], winning, at) {
                    winning = lane_score[l];
                    at = lane_index[l];
                }
            }
            if at != NO_INDEX {
                write_alternative(chosen_count, at);
                chosen[chosen_count] = at;
                chosen_count += 1u;
                corner[CORNER_ALTERNATIVES] = chosen_count;
            } else {
                ranking = false;
            }
        }
        workgroupBarrier();
    }
}
