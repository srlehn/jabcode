// BASELINE ONLY - do not wire this as the directional route.
//
// This is the CPU sweep transcribed into WGSL: one lane serially walks a whole
// scan line, every sample does float coordinate math, a bounds test, a packed
// bit extraction and a scattered load, the three channels run in series, and
// every hit contends on one global atomic. It runs on the device without using
// it. Its purpose is to be the measured baseline that a parallel transition and
// run-window design has to beat, and to have shown that the run-length machine
// gives sane results at arbitrary angles on device.
//
// The replacement builds directional transition maps in parallel, compacts them
// into run descriptors with prefix scans, tests five-run windows independently,
// and emits only cross-checked survivors instead of raw hits.
//
// Finder-pattern scan along arbitrary directions over the packed binary masks,
// per requested channel, appending the same compact record the row scan emits.
// It covers what the row scan cannot: the row scan only walks rows, so every
// non-zero angle falls back to the CPU over the whole frame.
//
// Lines are indexed by signed perpendicular offset, exactly as the CPU sweep
// indexes them, so lane i covers the line at q_lo + i*q_step and the host can
// reconstruct each hit's centre from the record's sample index alone.
//
// Unlike the row scan this makes no attempt to reproduce host float64 sampling
// bit for bit: it addresses samples in f32, which can land on a different pixel
// than the CPU walk within half a unit of a boundary. The proportion tests are
// still exact integers, so a hit here is a hit the CPU machine would also
// accept on the samples it saw; the two may simply disagree about borderline
// candidates, which the cross-check stage is there to resolve.

struct Params {
    width: u32,
    height: u32,
    channel_mask: u32,
    capacity: u32,
    // Major-axis step, one component being +/-1, matching scanDirection.
    dx: f32,
    dy: f32,
    // Perpendicular unit vector the line offsets are measured along.
    nx: f32,
    ny: f32,
    q_lo: f32,
    q_step: f32,
    line_count: u32,
    pad0: u32,
}

struct Records {
    count: atomic<u32>,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    data: array<u32>,
}

@group(0) @binding(0) var<storage, read> packed_masks: array<u32>;
@group(0) @binding(1) var<storage, read_write> records: Records;
@group(0) @binding(2) var<storage, read> params: Params;

fn mask_bit(pixel: u32, channel: u32) -> u32 {
    let word = packed_masks[pixel / 8u];
    return (word >> ((pixel % 8u) * 3u + channel)) & 1u;
}

fn abs_diff(a: u32, b: u32) -> u32 {
    if a > b {
        return a - b;
    }
    return b - a;
}

// layer_ok and check_cross are the row scan's exact integer proportion tests,
// unchanged: the run lengths they judge are counts either way, so nothing about
// the sampling direction reaches them.
fn layer_ok(inside: u32, s: u32) -> bool {
    let d2 = 2u * abs_diff(inside, 3u * s);
    if d2 < inside {
        return true;
    }
    if d2 > inside || 3u * s < inside || inside % 3u == 0u {
        return false;
    }
    let high = firstLeadingBit(inside);
    var exponent = high - 2u;
    if inside >= (3u << (high - 1u)) {
        exponent = high - 1u;
    }
    let shift = 52u - exponent;
    var remainder = inside % 3u;
    if shift % 2u == 1u {
        remainder = (remainder * 2u) % 3u;
    }
    return remainder == 2u;
}

fn check_cross(s0: u32, s1: u32, s2: u32, s3: u32, s4: u32) -> bool {
    if s1 == 0u || s2 == 0u || s3 == 0u {
        return false;
    }
    let inside = s1 + s2 + s3;
    return layer_ok(inside, s1) &&
        layer_ok(inside, s2) &&
        layer_ok(inside, s3) &&
        12u * s0 > inside &&
        12u * s4 > inside &&
        6u * abs_diff(s1, s3) < inside;
}

fn emit(channel: u32, line: u32, seq: u32, end_pos: u32, s2: u32, s3: u32, s4: u32, inside: u32) {
    let index = atomicAdd(&records.count, 1u);
    if index < params.capacity {
        let base = index * 8u;
        records.data[base] = channel;
        records.data[base + 1u] = line;
        records.data[base + 2u] = seq;
        records.data[base + 3u] = end_pos;
        records.data[base + 4u] = s2;
        records.data[base + 5u] = s3;
        records.data[base + 6u] = s4;
        records.data[base + 7u] = inside;
    }
}

// sample_at returns the mask bit at sample i along the line through origin,
// and whether that sample is inside the frame. The bounds test stays per
// sample rather than being folded into the clip because truncation can put a
// sample one pixel outside a range the clip computed in reals.
fn sample_at(origin: vec2<f32>, i: i32, channel: u32) -> vec2<u32> {
    let p = origin + f32(i) * vec2<f32>(params.dx, params.dy);
    let x = i32(p.x);
    let y = i32(p.y);
    if x < 0 || x >= i32(params.width) || y < 0 || y >= i32(params.height) {
        return vec2<u32>(0u, 0u);
    }
    let pixel = u32(y) * params.width + u32(x);
    return vec2<u32>(mask_bit(pixel, channel), 1u);
}

// clip_line restricts a line to the frame, returning the first in-frame sample
// index and the last one. It is clipScanLine: each axis contributes the
// interval of i keeping that coordinate in range, and the walk is their
// intersection. A zero component means the coordinate never moves, so the line
// either holds throughout or misses the frame.
fn clip_line(origin: vec2<f32>) -> vec3<i32> {
    var lo = -3.4e38;
    var hi = 3.4e38;
    let step = vec2<f32>(params.dx, params.dy);
    let limit = vec2<f32>(f32(params.width - 1u), f32(params.height - 1u));
    for (var axis = 0u; axis < 2u; axis++) {
        let p = origin[axis];
        let s = step[axis];
        let l = limit[axis];
        if s == 0.0 {
            if p < 0.0 || p > l {
                return vec3<i32>(0, 0, 0);
            }
            continue;
        }
        var t0 = -p / s;
        var t1 = (l - p) / s;
        if t0 > t1 {
            let swap = t0;
            t0 = t1;
            t1 = swap;
        }
        lo = max(lo, t0);
        hi = min(hi, t1);
    }
    let start = i32(ceil(lo));
    let end = i32(floor(hi));
    if end < start {
        return vec3<i32>(0, 0, 0);
    }
    return vec3<i32>(start, end, 1);
}

fn scan_line_channel(line: u32, channel: u32) {
    let q = params.q_lo + f32(line) * params.q_step;
    let origin = q * vec2<f32>(params.nx, params.ny);
    let clip = clip_line(origin);
    if clip.z == 0 {
        return;
    }
    let line_end = clip.y;

    var start_i = clip.x;
    var skip = 0;
    var seq = 0u;
    var first = true;
    loop {
        if !first && start_i > line_end {
            break;
        }
        first = false;
        start_i = start_i + skip;
        if start_i > line_end {
            break;
        }

        var sc = array<u32, 5>(0u, 0u, 0u, 0u, 0u);
        var cur_state = 0u;
        var res_start = start_i;
        var ok = false;
        var res_skip = 0;
        var hit_end_pos = 0;

        let head = sample_at(origin, start_i, channel);
        if head.y == 0u {
            break;
        }
        sc[0] = 1u;
        var prev = head.x;
        for (var j = start_i + 1; j <= line_end; j++) {
            let s = sample_at(origin, j, channel);
            if s.y == 0u {
                break;
            }
            let curr = s.x;
            if curr == prev {
                sc[cur_state] += 1u;
            }
            if curr != prev || j == line_end {
                if cur_state < 4u {
                    if sc[cur_state] < 3u {
                        if cur_state == 0u {
                            sc[0] = 1u;
                            res_start = j;
                        } else {
                            sc[cur_state - 1u] += sc[cur_state];
                            sc[cur_state] = 0u;
                            cur_state -= 1u;
                            sc[cur_state] += 1u;
                        }
                    } else {
                        cur_state += 1u;
                        sc[cur_state] += 1u;
                    }
                } else {
                    if sc[4] < 3u {
                        sc[3] += sc[4];
                        sc[4] = 0u;
                        cur_state = 3u;
                        sc[3] += 1u;
                        prev = curr;
                        continue;
                    }
                    if check_cross(sc[0], sc[1], sc[2], sc[3], sc[4]) {
                        res_skip = i32(sc[0]);
                        hit_end_pos = j;
                        if j == line_end && curr == prev {
                            hit_end_pos = j + 1;
                        }
                        ok = true;
                        break;
                    }
                    res_start += i32(sc[0]);
                    sc[0] = sc[1];
                    sc[1] = sc[2];
                    sc[2] = sc[3];
                    sc[3] = sc[4];
                    sc[4] = 1u;
                    cur_state = 4u;
                }
            }
            prev = curr;
        }
        if !ok {
            break;
        }
        // The sample index is signed along the line and the host needs it back
        // as one, so it is biased by the clip start rather than truncated at
        // zero; the host subtracts the same bias.
        emit(channel, line, seq, u32(hit_end_pos - clip.x), sc[2], sc[3], sc[4], sc[1] + sc[2] + sc[3]);
        seq += 1u;
        // Resume where the CPU walk resumes, and never before the next sample:
        // the leading run is normally at least three samples, but taking the
        // maximum is what makes non-advancement structurally impossible rather
        // than merely unobserved. A lane that failed to advance would hang the
        // whole dispatch, not just lose a hit.
        start_i = max(res_start + res_skip, start_i + 1);
        skip = 0;
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let line = id.x;
    if line >= params.line_count {
        return;
    }
    for (var channel = 0u; channel < 3u; channel++) {
        if (params.channel_mask & (1u << channel)) != 0u {
            scan_line_channel(line, channel);
        }
    }
}
