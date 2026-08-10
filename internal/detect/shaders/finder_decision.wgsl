// Retains the ordered finder answer without exposing a fold to the host.
// A successful inconsistent quad is a fallback, while the first consistent
// quad settles the sweep and writes zero indirect dimensions for later folds.
// Invalid resident evidence declines the whole device batch: continuing with
// an earlier fallback would hide the CPU work needed to answer that direction.

const PAT_WORDS: u32 = 6u;
const PAT_X: u32 = 2u;
const PAT_Y: u32 = 3u;
const PAT_MODULE: u32 = 4u;

const SEL_MISSING: u32 = 12u;
const SEL_PATTERNS: u32 = 16u;

const FOLD_DROPPED: u32 = 5u;
const ASSEMBLY_DEFERRED: u32 = 1u;
const ASSEMBLY_INVALID: u32 = 2u;
const POOL_DROPPED: u32 = 1u;

const CORNER_SOURCE: u32 = 0u;
const CORNER_MISS: u32 = 1u;
const CORNER_OK: u32 = 2u;
const CORNER_ALTERNATIVES: u32 = 3u;
const CORNER_PATTERN: u32 = 4u;
const CORNER_ALTERNATIVE_PATTERNS: u32 = 16u;
const SOURCE_FOUND: u32 = 0u;
const SOURCE_CONSTRUCTED: u32 = 1u;

const DECISION_HAVE: u32 = 0u;
const DECISION_CONSISTENT: u32 = 1u;
const DECISION_DECLINED: u32 = 2u;
const DECISION_MISSING: u32 = 3u;
const DECISION_CORNER_SOURCE: u32 = 4u;
const DECISION_CORNER_MISS: u32 = 5u;
const DECISION_ALTERNATIVES: u32 = 6u;
const DECISION_PATTERNS: u32 = 8u;
const DECISION_ALTERNATIVE_PATTERNS: u32 = 32u;
const MAX_ALTERNATIVES: u32 = 8u;

const QUAD_REJECT_MODULE_TOL: f32 = 1.8;
const QUAD_REJECT_EDGE_TOL: f32 = 1.8;

@group(0) @binding(0) var<storage, read> selection: array<u32>;
@group(0) @binding(1) var<storage, read> fold: array<u32>;
@group(0) @binding(2) var<storage, read> assembly: array<u32>;
@group(0) @binding(3) var<storage, read> family_pool: array<u32>;
@group(0) @binding(4) var<storage, read> contextual_pool: array<u32>;
@group(0) @binding(5) var<storage, read> corner: array<u32>;
@group(0) @binding(6) var<storage, read_write> decision: array<u32>;
@group(0) @binding(7) var<storage, read_write> indirect: array<u32>;

fn pattern_word(slot: u32, field: u32) -> u32 {
    if selection[SEL_MISSING] == 1u && corner[CORNER_SOURCE] != SOURCE_FOUND &&
        corner[CORNER_MISS] == slot {
        return corner[CORNER_PATTERN + field];
    }
    return selection[SEL_PATTERNS + slot * PAT_WORDS + field];
}

fn pattern_f32(slot: u32, field: u32) -> f32 {
    return bitcast<f32>(pattern_word(slot, field));
}

fn point_distance(a: u32, b: u32) -> f32 {
    let dx = pattern_f32(a, PAT_X) - pattern_f32(b, PAT_X);
    let dy = pattern_f32(a, PAT_Y) - pattern_f32(b, PAT_Y);
    return sqrt(dx * dx + dy * dy);
}

fn infinity() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

fn ratio(a: f32, b: f32) -> f32 {
    if a <= 0.0 || b <= 0.0 {
        return infinity();
    }
    return max(a, b) / min(a, b);
}

fn convex() -> bool {
    var orientation = 0.0;
    for (var i = 0u; i < 4u; i += 1u) {
        let j = (i + 1u) & 3u;
        let k = (i + 2u) & 3u;
        let cross =
            (pattern_f32(j, PAT_X) - pattern_f32(i, PAT_X)) *
                (pattern_f32(k, PAT_Y) - pattern_f32(j, PAT_Y)) -
            (pattern_f32(j, PAT_Y) - pattern_f32(i, PAT_Y)) *
                (pattern_f32(k, PAT_X) - pattern_f32(j, PAT_X));
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

fn consistent() -> bool {
    if !convex() {
        return false;
    }
    var module_min = pattern_f32(0u, PAT_MODULE);
    var module_max = module_min;
    for (var slot = 1u; slot < 4u; slot += 1u) {
        let module = pattern_f32(slot, PAT_MODULE);
        module_min = min(module_min, module);
        module_max = max(module_max, module);
    }
    if module_min <= 0.0 || module_max / module_min > QUAD_REJECT_MODULE_TOL {
        return false;
    }
    let top = point_distance(0u, 1u);
    let right = point_distance(1u, 2u);
    let bottom = point_distance(2u, 3u);
    let left = point_distance(3u, 0u);
    return max(ratio(top, bottom), ratio(left, right)) <= QUAD_REJECT_EDGE_TOL;
}

fn stop_later_folds() {
    indirect[0] = 0u;
    indirect[1] = 0u;
    indirect[2] = 0u;
}

@compute @workgroup_size(1)
fn main() {
    let untrusted = assembly[ASSEMBLY_INVALID] != 0u ||
        assembly[ASSEMBLY_DEFERRED] != 0u || fold[FOLD_DROPPED] != 0u ||
        family_pool[POOL_DROPPED] != 0u || contextual_pool[POOL_DROPPED] != 0u;
    if untrusted {
        decision[DECISION_DECLINED] = 1u;
        stop_later_folds();
        return;
    }

    let missing = selection[SEL_MISSING];
    let success = missing == 0u ||
        (missing == 1u && corner[CORNER_OK] != 0u);
    if !success {
        return;
    }

    let is_consistent = consistent();
    let replace = decision[DECISION_HAVE] == 0u ||
        (decision[DECISION_CONSISTENT] == 0u && is_consistent);
    if replace {
        decision[DECISION_HAVE] = 1u;
        decision[DECISION_CONSISTENT] = select(0u, 1u, is_consistent);
        decision[DECISION_MISSING] = missing;
        decision[DECISION_CORNER_SOURCE] = corner[CORNER_SOURCE];
        decision[DECISION_CORNER_MISS] = corner[CORNER_MISS];
        var alternatives = 0u;
        if missing == 1u && corner[CORNER_SOURCE] == SOURCE_CONSTRUCTED {
            alternatives = min(corner[CORNER_ALTERNATIVES], MAX_ALTERNATIVES);
        }
        decision[DECISION_ALTERNATIVES] = alternatives;
        for (var slot = 0u; slot < 4u; slot += 1u) {
            for (var word = 0u; word < PAT_WORDS; word += 1u) {
                decision[DECISION_PATTERNS + slot * PAT_WORDS + word] =
                    pattern_word(slot, word);
            }
        }
        for (var slot = 0u; slot < alternatives; slot += 1u) {
            for (var word = 0u; word < PAT_WORDS; word += 1u) {
                decision[DECISION_ALTERNATIVE_PATTERNS + slot * PAT_WORDS + word] =
                    corner[CORNER_ALTERNATIVE_PATTERNS + slot * PAT_WORDS + word];
            }
        }
    }
    if is_consistent {
        stop_later_folds();
    }
}
