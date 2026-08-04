// Bindings and geometry for the directional current-family chain. The fused
// window producer writes raw eight-word records without a header, so this
// stage cannot use the row chain's ScanRecords view.

struct ChainDirection {
    dx: F64,
    dy: F64,
    px_per_sample: F64,
}

struct ChainParams {
    width: u32,
    height: u32,
    capacity: u32,
    flags: u32,
    classify_current: u32,
    classify_bsi: u32,
    cross_color_bits: u32,
    pad: u32,

    geom_dx: F64,
    geom_dy: F64,
    geom_nx: F64,
    geom_ny: F64,
    geom_q_lo: F64,
    geom_q_step: F64,

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
