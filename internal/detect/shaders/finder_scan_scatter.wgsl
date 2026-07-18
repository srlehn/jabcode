// Moves every staged finder row scan record to its walk-order slot: one
// lane per staged record computes the target from the prefix-scanned
// offsets and the record's own (channel, row, sequence), so the sorted
// buffer holds each channel's hits contiguously in the CPU walk's order
// and the host decodes them without sorting. On an overflowed pass some
// staged records are missing and some targets exceed the capacity, so the
// sorted data is incomplete; the header total past the capacity makes the
// host retry with grown buffers and gates the chain kernels off, so the
// holes are never read.

struct Params {
    width: u32,
    height: u32,
    channel_mask: u32,
    capacity: u32,
}

struct Staged {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    data: array<u32>,
}

struct Sorted {
    count: u32,
    channel0: u32,
    channel1: u32,
    channel2: u32,
    data: array<u32>,
}

@group(0) @binding(0) var<storage, read> staged: Staged;
@group(0) @binding(1) var<storage, read_write> sorted: Sorted;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= staged.count || index >= params.capacity {
        return;
    }
    let base = index * 8u;
    let channel = staged.data[base];
    let y = staged.data[base + 1u];
    let seq = staged.data[base + 2u];
    let slot = offsets[channel * params.height + y] + seq;
    if slot >= params.capacity {
        return;
    }
    let target = slot * 8u;
    for (var word = 0u; word < 8u; word++) {
        sorted.data[target + word] = staged.data[base + word];
    }
}
