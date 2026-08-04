// Sizes the directional chain's dispatch from the count the fused window scan
// just wrote, so the count never crosses the bus. Without this the host has to
// submit the scan, wait for it, read sixteen bytes, and submit the chain in a
// second recording: two full pipeline stalls per swept direction to learn a
// number the device already holds.
//
// A scan that overflowed its record buffer dispatches nothing. Its records are
// truncated, the host walks that direction itself rather than acting on a
// short list, and chaining over the surviving prefix would only spend device
// time on a result that gets discarded.

// The record capacity's word index in the shared chain parameter block. One
// scalar out of that struct is all this needs, so it binds the block as words
// instead of restating a layout that would then have two places to drift.
const PARAMS_CAPACITY: u32 = 2u;

// Must match the directional chain kernel's workgroup size.
const CHAIN_WORKGROUP_SIZE: u32 = 64u;

// Where the raw hit count is published for the host's overflow check. The
// chain's atomic counters never touch this word.
const SUMMARY_REQUIRED: u32 = 6u;

@group(0) @binding(0) var<storage, read> counters: array<u32>;
// args[0..2] are the workgroup counts Vulkan reads for the indirect dispatch;
// args[3] is the invocation bound the chain kernel clamps itself to.
@group(0) @binding(1) var<storage, read_write> args: array<u32>;
@group(0) @binding(2) var<storage, read_write> summary: array<u32>;
@group(0) @binding(3) var<storage, read> chain_params: array<u32>;

@compute @workgroup_size(1)
fn main() {
    let required = counters[0];
    summary[SUMMARY_REQUIRED] = required;
    var count = required;
    if count > chain_params[PARAMS_CAPACITY] { count = 0u; }
    args[0] = (count + CHAIN_WORKGROUP_SIZE - 1u) / CHAIN_WORKGROUP_SIZE;
    args[1] = 1u;
    args[2] = 1u;
    args[3] = count;
}
