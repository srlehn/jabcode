// Derives the ordinary current-family binarizer and row-finder controls from
// the resident level shape. Only the level shape and retained raw-record bound
// are command scalars; no stage-specific parameter block crosses from the host.

const BIN_WIDTH: u32 = 0u;
const BIN_HEIGHT: u32 = 1u;
const BIN_BLOCK_SIZE: u32 = 2u;
const BIN_BLOCKS_X: u32 = 3u;
const BIN_BLOCKS_Y: u32 = 4u;
const BIN_FLAGS: u32 = 5u;
const BIN_SCAN_CAPACITY: u32 = 10u;

const BLOCK_DIVISOR: u32 = 24u;
const BLOCK_MIN: u32 = 64u;
const BLOCK_MAX: u32 = 512u;

const SCAN_CHANNELS: u32 = 2u;
const CHAIN_CLASSIFY_CURRENT: u32 = 3264u;
const CHAIN_CLASSIFY_BSI: u32 = 1876u;
const CHAIN_CROSS_COLOR_BITS: u32 = 0u;
const CHAIN_COLOR_SOURCE: u32 = 2u;
const CHAIN_ROW_STRIDE_SHIFT: u32 = 8u;
const CHAIN_COMPACT_CAPACITY: u32 = 33268u;

@group(0) @binding(0) var<storage, read_write> binarizer: array<u32>;
@group(0) @binding(1) var<storage, read_write> scan: array<u32>;
@group(0) @binding(2) var<storage, read_write> chain: array<u32>;

@compute @workgroup_size(1)
fn main() {
    let width = max(binarizer[BIN_WIDTH], 1u);
    let height = max(binarizer[BIN_HEIGHT], 1u);
    let block_size = clamp(min(width, height) / BLOCK_DIVISOR, BLOCK_MIN, BLOCK_MAX);
    let capacity = binarizer[BIN_SCAN_CAPACITY];

    binarizer[BIN_BLOCK_SIZE] = block_size;
    binarizer[BIN_BLOCKS_X] = (width + block_size - 1u) / block_size;
    binarizer[BIN_BLOCKS_Y] = (height + block_size - 1u) / block_size;
    binarizer[BIN_FLAGS] = 0u;

    scan[0] = width;
    scan[1] = height;
    scan[2] = SCAN_CHANNELS;
    scan[3] = capacity;

    chain[0] = width;
    chain[1] = height;
    chain[2] = capacity;
    chain[3] = CHAIN_COLOR_SOURCE | (1u << CHAIN_ROW_STRIDE_SHIFT);
    chain[4] = CHAIN_CLASSIFY_CURRENT;
    chain[5] = CHAIN_CLASSIFY_BSI;
    chain[6] = CHAIN_CROSS_COLOR_BITS;
    chain[7] = CHAIN_COMPACT_CAPACITY;
}
