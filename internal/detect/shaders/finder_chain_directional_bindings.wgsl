// Bindings and geometry for the directional current-family chain. The fused
// window producer writes raw eight-word records without a header, so this
// stage cannot use the row chain's ScanRecords view.

struct ChainDirection {
    dx: f32,
    dy: f32,
    px_per_sample: f32,
}

struct ChainParams {
    width: u32,
    height: u32,
    capacity: u32,
    flags: u32,
    classify_current: u32,
    classify_bsi: u32,
    cross_color_bits: u32,
    // How many compacted candidates the outcome buffer holds.
    compact_capacity: u32,

    geom_dx: f32,
    geom_dy: f32,
    geom_nx: f32,
    geom_ny: f32,
    geom_q_lo: f32,
    geom_q_step: f32,

    base: ChainDirection,
    perpendicular: ChainDirection,
    diagonal_right: ChainDirection,
    diagonal_left: ChainDirection,
}

struct DirectionalRecords { data: array<u32> }

@group(0) @binding(0) var<storage, read> packed_masks: array<u32>;
@group(0) @binding(1) var<storage, read> records: DirectionalRecords;
@group(0) @binding(2) var<storage, read_write> outcomes: array<u32>;
@group(0) @binding(3) var<storage, read> chain_params: ChainParams;

// The balanced source image, one packed RGBA pixel per word. The chain's
// colour-signal test is the only stage that observes source intensities rather
// than mask bits, and reading them here is what keeps the host from having to
// download the whole image to answer the same question per candidate.
@group(0) @binding(4) var<storage, read> balanced_pixels: array<u32>;

// The per-direction summary: the compacted candidate count, the raw hit count
// and the four branch counters. Every hit contributes to it with atomics, so
// the host reads one small block instead of one record per hit.
@group(0) @binding(5) var<storage, read_write> summary: array<atomic<u32>>;

// The dispatch arguments finder_dispatch_args.wgsl wrote from the scan's own
// counter. Word three is how many records that scan actually produced, which is
// this kernel's invocation bound: an indirect dispatch rounds up to whole
// workgroups, and the host never learns the count at all.
@group(0) @binding(6) var<storage, read> dispatch_args: array<u32>;

// The seed module-size histogram, shared with the row chain and accumulated
// across every direction and pass of one locate. It has a single consumer, the
// descreen scale decision, which reads it once - so it stays here and is
// fetched where that decision is made rather than riding every summary.
@group(0) @binding(7) var<storage, read_write> seed_histogram: array<atomic<u32>>;
