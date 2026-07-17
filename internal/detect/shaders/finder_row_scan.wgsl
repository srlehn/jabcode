// Finder-pattern row scan over the packed binary masks: one lane runs the
// complete five-state run-length machine over one image row, per requested
// channel, and appends a compact integer record per raw hit. The machine and
// its driver loop mirror seekPatternHorizontal and the per-row scan drivers
// in the CPU detector exactly; every proportion check is reformulated in
// exact integer arithmetic so the emitted hits are bit-identical to the CPU
// scan (the host derives the float centre and module size from the record's
// integers with the CPU's own float64 expressions).

struct Params {
    width: u32,
    height: u32,
    channel_mask: u32,
    capacity: u32,
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

// layer_ok evaluates the float64 comparison |layer - s| < layer/2 with
// layer = float64(inside)/3.0 exactly, in integers. Off the boundary the
// rational form 2*|inside - 3*s| < inside decides (the integer margin dwarfs
// the float64 rounding error). On the exact boundary the outcome follows the
// rounding of inside/3: the lower boundary (3*s < inside) forces inside
// divisible by 6, where the division is exact and the strict comparison
// fails; the upper boundary s = inside/2 with inside not divisible by 3
// accepts exactly when float64 rounds inside/3 up, which is when
// (inside << (52 - e)) mod 3 == 2 for inside/3 in [2^e, 2^(e+1)).
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

// check_cross is checkPatternCross in exact integers: with inside = s1+s2+s3
// and tol = inside/6, the layer comparisons go through layer_ok, the outer
// conditions s > tol/2 translate to 12*s > inside and |s1 - s3| < tol to
// 6*|s1 - s3| < inside - their boundary equalities force inside divisible by
// 12 or 6, where the float64 evaluation is exact and strict, so the integer
// forms never disagree there.
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

fn emit(channel: u32, y: u32, seq: u32, end_pos: u32, s2: u32, s3: u32, s4: u32, inside: u32) {
    let index = atomicAdd(&records.count, 1u);
    if index < params.capacity {
        let base = index * 8u;
        records.data[base] = channel;
        records.data[base + 1u] = y;
        records.data[base + 2u] = seq;
        records.data[base + 3u] = end_pos;
        records.data[base + 4u] = s2;
        records.data[base + 5u] = s3;
        records.data[base + 6u] = s4;
        records.data[base + 7u] = inside;
    }
}

fn scan_row_channel(y: u32, channel: u32) {
    let w = params.width;
    let row_base = y * w;
    var start_x = 0u;
    var end_x = w;
    var skip = 0u;
    var seq = 0u;
    var first = true;
    loop {
        if !first && !(start_x < w && end_x < w) {
            break;
        }
        first = false;
        start_x = start_x + skip;
        end_x = w;

        // seekPatternHorizontal from start_x to w.
        var sc = array<u32, 5>(0u, 0u, 0u, 0u, 0u);
        var cur_state = 0u;
        var res_start = start_x;
        var ok = false;
        var res_skip = 0u;
        var hit_end_pos = 0u;
        if start_x < w {
            sc[0] = 1u;
            var prev = mask_bit(row_base + start_x, channel);
            for (var j = start_x + 1u; j < w; j++) {
                let curr = mask_bit(row_base + j, channel);
                if curr == prev {
                    sc[cur_state] += 1u;
                }
                if curr != prev || j == w - 1u {
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
                            end_x = j + 1u;
                            res_skip = sc[0];
                            hit_end_pos = j;
                            if j == w - 1u && curr == prev {
                                hit_end_pos = j + 1u;
                            }
                            ok = true;
                            break;
                        }
                        res_start += sc[0];
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
        }
        if !ok {
            break;
        }
        emit(channel, y, seq, hit_end_pos, sc[2], sc[3], sc[4], sc[1] + sc[2] + sc[3]);
        seq += 1u;
        start_x = res_start;
        skip = res_skip;
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let y = id.x;
    if y >= params.height {
        return;
    }
    for (var channel = 0u; channel < 3u; channel++) {
        if (params.channel_mask & (1u << channel)) != 0u {
            scan_row_channel(y, channel);
        }
    }
}
