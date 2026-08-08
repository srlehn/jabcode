// Bindings and fixed parameters of the row finder-chain kernels. The chain
// math is kept in finder_chain_prelude.wgsl so the directional chain can use
// the same arithmetic with its own record layout.

struct ChainParams {
    width: u32,
    height: u32,
    capacity: u32,
    // Bit 0 selects the print-level slack rule of ccSlack, bit 1 says the
    // balanced image is bound, and bits 8 and up carry the row stride the
    // consumer walks at. The device folds the counters and the module
    // histogram, so it has to skip exactly the rows the host walk would have
    // skipped or the totals it reports describe a different scan.
    flags: u32,
    // Binarized palette bits of the four current-family finder cores,
    // three bits (R, G, B) per type at bit type*3.
    classify_current: u32,
    // The same table for the four BSI-era finder cores.
    classify_bsi: u32,
    // Bit 0: expected red-channel core bit of the blue-branch color check;
    // bit 1: expected blue-channel core bit of the red-branch color check.
    cross_color_bits: u32,
    // How many compacted candidates one channel's region holds.
    compact_capacity: u32,
}

struct ScanRecords {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    data: array<u32>,
}

@group(0) @binding(0) var<storage, read> packed_masks: array<u32>;
@group(0) @binding(1) var<storage, read> records: ScanRecords;
@group(0) @binding(2) var<storage, read_write> outcomes: array<u32>;
@group(0) @binding(3) var<storage, read> chain_params: ChainParams;

// The balanced source image, one packed RGBA pixel per word. Only the
// current-family fragment reads it, for the source-colour signal; the BSI-era
// signature has no such test and leaves the binding untouched.
@group(0) @binding(4) var<storage, read> balanced_pixels: array<u32>;

// One counter block per scan channel, so the two family kernels sharing this
// record buffer fold into disjoint words without coordinating.
@group(0) @binding(5) var<storage, read_write> summary: array<atomic<u32>>;

// The compacted candidates, again one region per channel. Only hits that
// survive the chain or become contextual seeds land here, which is what turns
// a few hundred thousand records into a list the host can read.
@group(0) @binding(6) var<storage, read_write> compacted: array<u32>;

// The seed module-size histogram, shared with the directional chain and
// accumulated across every direction and pass of one locate. It has a single
// consumer, the descreen scale decision, which reads it once - so it stays here
// and is fetched where that decision is made rather than riding every summary.
@group(0) @binding(7) var<storage, read_write> seed_histogram: array<atomic<u32>>;
