// Exclusive prefix scan over the finder row scan's per-(channel, row) hit
// tallies, converting them in place into the scatter kernel's write
// offsets. Channel-major order makes each channel's records contiguous and
// ordered by row; the total and the per-channel totals go into the sorted
// record buffer's header, so the host learns the exact record count and
// the per-channel ranges from the 16-byte header alone. One lane folds at
// most three channels times the row count of a canvas, a few thousand
// additions.

struct Params {
    width: u32,
    height: u32,
    channel_mask: u32,
    capacity: u32,
}

@group(0) @binding(0) var<storage, read_write> records: array<u32>;
@group(0) @binding(1) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(2) var<storage, read> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x != 0u {
        return;
    }
    var total = 0u;
    for (var channel = 0u; channel < 3u; channel++) {
        var sum = 0u;
        for (var y = 0u; y < params.height; y++) {
            let index = channel * params.height + y;
            let tally = offsets[index];
            offsets[index] = total;
            total += tally;
            sum += tally;
        }
        records[1u + channel] = sum;
    }
    records[0u] = total;
}
