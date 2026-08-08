// Reduces the folded candidate list to the single best pattern of each of the
// four finder types, and applies the outvoted-type prune.
//
// This is one lane. The work is a handful of passes over at most five hundred
// entries and four types, so spreading it over a workgroup would buy nothing
// and would cost the sequential prune its ordering, which is what decides
// whether a direction stays recoverable.
//
// Every threshold and every ordering here is the host's. The prune in
// particular is not a cleanup: it decides whether a direction that found all
// four types keeps the weakest one, and getting its stopping rule wrong turns a
// readable direction into one that publishes a quad built on background blobs.

const TYPES: u32 = 4u;
const MIN_CROSSINGS: u32 = 3u;

const PAT_WORDS: u32 = 6u;
const PAT_TYP: u32 = 0u;
const PAT_DIRECTION: u32 = 1u;
const PAT_X: u32 = 2u;
const PAT_Y: u32 = 3u;
const PAT_MODULE: u32 = 4u;
const PAT_FOUND: u32 = 5u;

const PARAM_PRINT_PASS: u32 = 2u;
const PARAM_CONTEXTUAL: u32 = 3u;
// Set when the contextual types come from the pool the directions accumulated
// rather than from the caller. The stage harness that exercises this kernel on
// its own has no pool, and reading a stale one there would make its cases
// depend on whatever ran before them.
const PARAM_POOL_TYPES: u32 = 6u;

const POOL_TYPE_MASK: u32 = 2u;

const FOLD_TOTAL: u32 = 0u;

// Pre-prune group sizes, each type's best before the prune, what survived it,
// how many types are absent, then the four selected patterns and the four the
// prune saw. What the prune removes cannot be recovered afterwards, which is
// why both sets are reported.
const SEL_PREPRUNE: u32 = 0u;
const SEL_PRESELECT: u32 = 4u;
const SEL_SELECTED: u32 = 8u;
const SEL_MISSING: u32 = 12u;
const SEL_PATTERNS: u32 = 16u;
const SEL_PRE_PATTERNS: u32 = 40u;
const SEL_WORDS: u32 = 64u;

@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> patterns: array<u32>;
@group(0) @binding(2) var<storage, read> fold: array<u32>;
@group(0) @binding(3) var<storage, read_write> selection: array<u32>;
@group(0) @binding(4) var<storage, read> contextual_pool: array<u32>;

fn pattern_f32(index: u32, field: u32) -> f32 {
    return bitcast<f32>(patterns[index * PAT_WORDS + field]);
}

fn copy_pattern(base: u32, slot: u32, source: u32) {
    for (var word = 0u; word < PAT_WORDS; word += 1u) {
        selection[base + slot * PAT_WORDS + word] = patterns[source * PAT_WORDS + word];
    }
}

fn clear_pattern(base: u32, slot: u32) {
    for (var word = 0u; word < PAT_WORDS; word += 1u) {
        selection[base + slot * PAT_WORDS + word] = 0u;
    }
}

fn selected_found(slot: u32) -> u32 {
    return selection[SEL_PATTERNS + slot * PAT_WORDS + PAT_FOUND];
}

// grouped reports whether an accumulated entry belongs to a type's group. A
// module spans at least three pixels, so a genuine finder is crossed at least
// three times and anything under that is a single stray crossing.
fn grouped(index: u32, typ: u32) -> bool {
    return patterns[index * PAT_WORDS + PAT_FOUND] >= MIN_CROSSINGS &&
        patterns[index * PAT_WORDS + PAT_TYP] == typ;
}

@compute @workgroup_size(1)
fn main() {
    for (var word = 0u; word < SEL_WORDS; word += 1u) {
        selection[word] = 0u;
    }
    let total = fold[FOLD_TOTAL];
    let print_pass = params[PARAM_PRINT_PASS] != 0u;
    var contextual = params[PARAM_CONTEXTUAL];
    if params[PARAM_POOL_TYPES] != 0u {
        contextual = contextual_pool[POOL_TYPE_MASK];
    }

    var max_found = 0u;
    for (var typ = 0u; typ < TYPES; typ += 1u) {
        var members = 0u;
        var module_total = 0.0;
        for (var i = 0u; i < total; i += 1u) {
            if grouped(i, typ) {
                members += 1u;
                module_total += pattern_f32(i, PAT_MODULE);
            }
        }
        selection[SEL_PREPRUNE + typ] = members;
        if members == 0u {
            clear_pattern(SEL_PATTERNS, typ);
            continue;
        }

        // bestPattern's own choice: the most-crossed member, and among equals
        // the one whose module size sits closest to the group mean. The 100.0
        // seed is the host's and is only ever a first-iteration placeholder.
        let mean = module_total / f32(members);
        var best_found = 0u;
        var best_diff = 100.0;
        var best = 0u;
        var seen = false;
        for (var i = 0u; i < total; i += 1u) {
            if !grouped(i, typ) {
                continue;
            }
            let found = patterns[i * PAT_WORDS + PAT_FOUND];
            let diff = abs(pattern_f32(i, PAT_MODULE) - mean);
            if !seen || found > best_found {
                best_found = found;
                best_diff = diff;
                best = i;
                seen = true;
            } else if found == best_found && diff < best_diff {
                best_diff = diff;
                best = i;
            }
        }
        copy_pattern(SEL_PATTERNS, typ, best);
    }

    for (var typ = 0u; typ < TYPES; typ += 1u) {
        let found = selected_found(typ);
        selection[SEL_PRESELECT + typ] = found;
        max_found = max(max_found, found);
        for (var word = 0u; word < PAT_WORDS; word += 1u) {
            selection[SEL_PRE_PATTERNS + typ * PAT_WORDS + word] =
                selection[SEL_PATTERNS + typ * PAT_WORDS + word];
        }
    }

    // The print-level passes skip the prune: colorant-plane misregistration
    // degrades the corners asymmetrically, and a candidate there has already
    // survived the widened multi-channel cross-checks.
    if !print_pass {
        var missing = 0u;
        var outvoted = array<u32, 4>(0u, 0u, 0u, 0u);
        var outvoted_count = 0u;
        for (var typ = 0u; typ < TYPES; typ += 1u) {
            let found = selected_found(typ);
            if found == 0u {
                missing += 1u;
            } else if f32(found) < 0.5 * f32(max_found) {
                outvoted[outvoted_count] = typ;
                outvoted_count += 1u;
            }
        }
        // Weakest first, stably, so equal counts keep type order. Insertion
        // sort over at most four entries is the whole of it.
        for (var i = 1u; i < outvoted_count; i += 1u) {
            let value = outvoted[i];
            var j = i;
            while j > 0u && selected_found(outvoted[j - 1u]) > selected_found(value) {
                outvoted[j] = outvoted[j - 1u];
                j -= 1u;
            }
            outvoted[j] = value;
        }

        // Where all four types were found the prune stops while the selection
        // is still recoverable: one absent type is interpolated from the other
        // three, a second is fatal. A contextual group for the absent type
        // proves it was crossed repeatedly elsewhere, which keeps a
        // three-of-four direction recoverable too.
        var recoverable = missing == 0u;
        if missing == 1u {
            for (var typ = 0u; typ < TYPES; typ += 1u) {
                if selected_found(typ) == 0u && (contextual & (1u << typ)) != 0u {
                    recoverable = true;
                    break;
                }
            }
        }
        for (var i = 0u; i < outvoted_count; i += 1u) {
            if recoverable && missing >= 1u {
                break;
            }
            clear_pattern(SEL_PATTERNS, outvoted[i]);
            missing += 1u;
        }
    }

    var missing = 0u;
    for (var typ = 0u; typ < TYPES; typ += 1u) {
        let found = selected_found(typ);
        if found == 0u {
            missing += 1u;
        } else {
            selection[SEL_SELECTED + typ] = found;
        }
    }
    selection[SEL_MISSING] = missing;
}
