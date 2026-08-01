// Fused run extraction and five-run testing, with no boundary buffer at all:
// everything the two compaction variants share.
//
// The boundary prototypes write every boundary of every line to device memory
// so a second pass can slide a five-run window over them. That buffer is the
// expensive part of the design, not the scan: it has to be sized for the worst
// case a line can produce, which is one boundary per sample, and at a 12 MP
// frame swept at one line per few pixels that is a hundred megabytes for a
// single angle. It is also written once and read once.
//
// A window is decided by six consecutive boundaries. A workgroup that is
// already walking a line has those six to hand as it goes, so the buffer only
// exists to carry them from one kernel to the next. This kernel keeps them in
// workgroup memory instead and emits only the windows that pass, a count that
// scales with image content rather than image area.
//
// Carrying five boundaries between blocks is what makes the window test exact
// at a block seam: a window is six boundaries, so the first window a block can
// close needs the five that preceded its own first boundary. Carrying four
// silently drops one window per seam.
//
// The window test is checkPatternCross: three inner runs within half a layer of
// their mean, two outer runs at least a quarter layer, and the first and third
// inner runs equal to within the same tolerance. Unlike the CPU sweep this
// tests every window rather than folding runs shorter than three samples into
// their neighbours, so it reaches signatures the CPU walk structurally cannot.
// counters[0] is every accepted window and counters[1] only those whose inner
// runs are at least three samples, which is the subset the CPU sweep could also
// have reached. These are pre-cross-check candidates, not finders: whether the
// extra class contains anything real is a question for the validation stage,
// and the split counter exists so it can be asked rather than assumed.

@group(0) @binding(1) var<storage, read_write> survivors: array<u32>;
@group(0) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;

const WORKGROUP: u32 = 256u;
// A survivor record: key, then the six boundaries that define the window, then
// one word of padding so the stride is a power of two.
const RECORD: u32 = 8u;

// values holds each lane's sample so its right-hand neighbour can read it
// instead of loading it again.
var<workgroup> values: array<u32, WORKGROUP>;
// bpos holds the five carried boundaries followed by this block's own.
var<workgroup> bpos: array<u32, 5u + WORKGROUP>;
var<workgroup> carry: u32;
var<workgroup> block_slot: atomic<u32>;
var<workgroup> block_strict: atomic<u32>;
var<workgroup> block_base: u32;

fn accept(s0: u32, s1: u32, s2: u32, s3: u32, s4: u32) -> bool {
    if s1 == 0u || s2 == 0u || s3 == 0u {
        return false;
    }
    let layer = f32(s1 + s2 + s3) / 3.0;
    let tol = layer / 2.0;
    return abs(layer - f32(s1)) < tol
        && abs(layer - f32(s2)) < tol
        && abs(layer - f32(s3)) < tol
        && f32(s0) > 0.5 * tol
        && f32(s4) > 0.5 * tol
        && abs(f32(i32(s1) - i32(s3))) < tol;
}

// block_flag reports whether sample i opens a run. Lanes 1..255 read their
// predecessor's sample from workgroup memory; only lane 0 crosses a block edge
// and reloads, and its predecessor is always inside the span because it is only
// consulted when i > span_start.
fn block_flag(
    origin: vec2<f32>,
    channel: u32,
    i: i32,
    span_start: i32,
    inside: bool,
    lane: u32,
) -> bool {
    var value = 3u;
    if inside {
        value = sample_at(origin, i, channel);
    }
    values[lane] = value;
    workgroupBarrier();
    var starts = false;
    if inside {
        if i == span_start {
            starts = true;
        } else {
            var prev = 3u;
            if lane == 0u {
                prev = sample_at(origin, i - 1, channel);
            } else {
                prev = values[lane - 1u];
            }
            starts = value != prev;
        }
    }
    return starts;
}

// flush_block tests every window this block closed and then rolls the carry.
// The window count is exactly valid-5, and every window it covers ends on a
// boundary this block produced, so no window is tested twice and none is
// skipped at a seam.
fn flush_block(key: u32, n: u32, lane: u32) {
    let c = carry;
    let valid = c + n;
    let base = 5u - c;

    if lane == 0u {
        atomicStore(&block_slot, 0u);
        atomicStore(&block_strict, 0u);
    }
    workgroupBarrier();

    var hit = false;
    var strict = false;
    var b: array<u32, 6>;
    if valid >= 6u && lane < valid - 5u {
        let t = base + lane;
        for (var k = 0u; k < 6u; k++) {
            b[k] = bpos[t + k];
        }
        let s0 = b[1] - b[0];
        let s1 = b[2] - b[1];
        let s2 = b[3] - b[2];
        let s3 = b[4] - b[3];
        let s4 = b[5] - b[4];
        hit = accept(s0, s1, s2, s3, s4);
        strict = hit && s1 >= 3u && s2 >= 3u && s3 >= 3u;
    }

    // Reserve inside the workgroup first, then take one global slot for the
    // whole block. One global atomic per block instead of one per survivor is
    // what keeps a noisy frame from serializing on the counter.
    var local = 0u;
    if hit {
        local = atomicAdd(&block_slot, 1u);
    }
    if strict {
        atomicAdd(&block_strict, 1u);
    }
    workgroupBarrier();
    if lane == 0u {
        let total = atomicLoad(&block_slot);
        block_base = atomicAdd(&counters[0], total);
        atomicAdd(&counters[1], atomicLoad(&block_strict));
    }
    workgroupBarrier();
    if hit {
        let index = block_base + local;
        if (index + 1u) * RECORD <= arrayLength(&survivors) {
            let at = index * RECORD;
            survivors[at] = key;
            for (var k = 0u; k < 6u; k++) {
                survivors[at + 1u + k] = b[k];
            }
            survivors[at + 7u] = 0u;
        }
    }

    workgroupBarrier();
    let newc = min(5u, valid);
    var moved = 0u;
    if lane < newc {
        moved = bpos[5u + n - newc + lane];
    }
    workgroupBarrier();
    if lane < newc {
        bpos[5u - newc + lane] = moved;
    }
    if lane == 0u {
        carry = newc;
    }
    workgroupBarrier();
}
