// The row finder chain's shared fold: the per-channel counters every hit
// contributes to and the compacted list only the candidates the host can act
// on reach. It sits after the prelude because it works in the prelude's
// Outcome type.

// Summary word offsets within a channel's block, shared with the host parser.
const ROW_SUMMARY_COMPACTED: u32 = 0u;
const ROW_SUMMARY_RAW_HITS: u32 = 1u;
const ROW_SUMMARY_BRANCH_BLUE: u32 = 2u;
const ROW_SUMMARY_BRANCH_RED: u32 = 3u;
const ROW_SUMMARY_RED_COLOR: u32 = 4u;
const ROW_SUMMARY_RED_CLASSIFIED: u32 = 5u;
const ROW_SUMMARY_OVERFLOW: u32 = 6u;
const ROW_SUMMARY_WORDS: u32 = 7u;
// Quarter-pixel buckets in the shared seed histogram, matching the host
// accumulator it merges into.
const ROW_SUMMARY_BUCKETS: u32 = 1024u;
const ROW_SUMMARY_MODULE_SCALE: f32 = 4.0;

// A compacted record carries its outcome, the row and sequence the host orders
// by, and the raw scan fields the hit came from. The raw fields cost little
// against the records they replace and mean a consumer holding a compacted hit
// reads the same values it would have read from the record itself, rather than
// zeros that would pass unnoticed.
const ROW_COMPACT_WORDS: u32 = 13u;

const CHAIN_FLAG_STRIDE_SHIFT: u32 = 8u;

const CHAIN_SURVIVOR: u32 = 16u;
const CHAIN_SEED: u32 = 32u;

// row_module_size is the hit's layer-size estimate, the same expression the
// host's finderRowHit.moduleSize uses, so both fill the same histogram bucket.
fn row_module_size(inside: u32) -> f32 {
    return f32(inside) / 3.0;
}

// row_stride_skips reports whether the consumer's walk would never visit this
// row, in which case the hit contributes to nothing at all.
fn row_stride_skips(y: u32) -> bool {
    let stride = chain_params.flags >> CHAIN_FLAG_STRIDE_SHIFT;
    return stride > 1u && (y % stride) != 0u;
}

// A raw scan that exceeded its record region never ran a complete chain. Mark
// that in the resident summary so the host needs the tiny record header only on
// the overflow fallback, not before every successful fold.
fn mark_row_scan_overflow(channel: u32) {
    atomicStore(&summary[channel * ROW_SUMMARY_WORDS + ROW_SUMMARY_OVERFLOW], 1u);
}

// summarize_row folds one hit into its channel's counters and appends it to
// that channel's compacted region when it survived. The counters are what let
// the host skip the raw records entirely: every hit still contributes its
// branch tallies and its module size to the histogram, but only candidates the
// host can act on are carried back.
fn summarize_row(channel: u32, y: u32, seq: u32, record: u32, outc: Outcome, module: f32) {
    let block = channel * ROW_SUMMARY_WORDS;
    atomicAdd(&summary[block + ROW_SUMMARY_RAW_HITS], 1u);
    if module > 0.0 {
        var bucket = u32(module * ROW_SUMMARY_MODULE_SCALE);
        if bucket >= ROW_SUMMARY_BUCKETS { bucket = ROW_SUMMARY_BUCKETS - 1u; }
        atomicAdd(&seed_histogram[bucket], 1u);
    }
    if (outc.flags & 1u) != 0u { atomicAdd(&summary[block + ROW_SUMMARY_BRANCH_BLUE], 1u); }
    if (outc.flags & 2u) != 0u { atomicAdd(&summary[block + ROW_SUMMARY_BRANCH_RED], 1u); }
    if (outc.flags & 4u) != 0u { atomicAdd(&summary[block + ROW_SUMMARY_RED_COLOR], 1u); }
    if (outc.flags & 8u) != 0u { atomicAdd(&summary[block + ROW_SUMMARY_RED_CLASSIFIED], 1u); }
    if (outc.flags & (CHAIN_SURVIVOR | CHAIN_SEED)) == 0u { return; }
    let slot = atomicAdd(&summary[block + ROW_SUMMARY_COMPACTED], 1u);
    // The count keeps rising past the region so the host sees the overflow and
    // falls back to reading the raw records rather than acting on a prefix.
    if slot >= chain_params.compact_capacity {
        atomicStore(&summary[block + ROW_SUMMARY_OVERFLOW], 1u);
        return;
    }
    let base = (channel * chain_params.compact_capacity + slot) * ROW_COMPACT_WORDS;
    compacted[base] = outc.flags;
    compacted[base + 1u] = u32(outc.typ);
    compacted[base + 2u] = bitcast<u32>(outc.dir);
    compacted[base + 3u] = bitcast<u32>(outc.cx);
    compacted[base + 4u] = bitcast<u32>(outc.cy);
    compacted[base + 5u] = bitcast<u32>(outc.ms);
    compacted[base + 6u] = y;
    compacted[base + 7u] = seq;
    compacted[base + 8u] = records.data[record + 3u];
    compacted[base + 9u] = records.data[record + 4u];
    compacted[base + 10u] = records.data[record + 5u];
    compacted[base + 11u] = records.data[record + 6u];
    compacted[base + 12u] = records.data[record + 7u];
}
