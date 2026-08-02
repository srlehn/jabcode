// Line addressing and frame clipping, shared by every directional prototype.

// mask_at returns the mask bit under a point, or 2 when the point is outside
// the frame. Every sampler here goes through it, including the off-line walks,
// so there is one definition of where a coordinate lands.
//
// Coordinates floor rather than truncate toward zero. Truncation maps a
// coordinate in (-1, 0) onto row or column 0, reading a pixel on the far side
// of the frame edge as though the line were still inside - and not by one
// sample but by 1/|component| of them, about 3.7 at 15 degrees. The CPU walk
// has that artifact; nothing requires reproducing it.
fn mask_at(p: vec2<f32>, channel: u32) -> u32 {
    let q = floor(p);
    let x = i32(q.x);
    let y = i32(q.y);
    if x < 0 || x >= i32(params.width) || y < 0 || y >= i32(params.height) {
        return 2u;
    }
    return mask_bit(u32(y) * params.width + u32(x), channel);
}

// sample_at returns the mask bit at sample i along one line. Callers only ask
// about samples inside the clipped span, so the bounds test is a guard rather
// than the mechanism that finds the frame edge.
fn sample_at(origin: vec2<f32>, i: i32, channel: u32) -> u32 {
    return mask_at(origin + f32(i) * vec2<f32>(params.dx, params.dy), channel);
}

fn in_frame(origin: vec2<f32>, i: i32) -> bool {
    return mask_at(origin + f32(i) * vec2<f32>(params.dx, params.dy), 0u) < 2u;
}

// clip_line restricts a line to the frame, returning the first and last in-frame
// sample index and whether any exist. Each axis contributes the interval of i
// keeping that coordinate in range and the span is their intersection; a zero
// step means that coordinate never moves, so the line either holds throughout or
// misses the frame entirely.
//
// Clipping here rather than letting an out-of-frame sentinel mark the edge is
// what keeps a sentinel run out of the output. If the boundary list contained
// the transition into and out of the frame, the window stage could not tell that
// run from a real one, and an out-of-frame run could take part in a five-run
// finder signature as though the mask had produced it.
fn clip_line(origin: vec2<f32>) -> vec3<i32> {
    var lo = -3.4e38;
    var hi = 3.4e38;
    let step = vec2<f32>(params.dx, params.dy);
    // Exclusive upper bounds: with floor addressing a sample is in frame when
    // its coordinate is in [0, dim), so the interval must be half-open. Using
    // dim-1 here excluded every sample whose coordinate fell in the last pixel,
    // which is 1/|component| samples at the far edge - two of them at 75
    // degrees, not the one that widening the span by a constant would cover.
    let limit = vec2<f32>(f32(params.width), f32(params.height));
    for (var axis = 0u; axis < 2u; axis++) {
        let p = origin[axis];
        let s = step[axis];
        if s == 0.0 {
            if p < 0.0 || p >= limit[axis] {
                return vec3<i32>(0, 0, 0);
            }
            continue;
        }
        var t0 = -p / s;
        var t1 = (limit[axis] - p) / s;
        if t0 > t1 {
            let swap = t0;
            t0 = t1;
            t1 = swap;
        }
        lo = max(lo, t0);
        hi = min(hi, t1);
    }
    // The half-open interval and the floor addressing now agree, so the trim
    // below only absorbs the rounding at each endpoint. It is kept because a
    // span that claims a sample the sampler rejects would break the contract
    // that every emitted run lies inside the frame.
    var start = max(i32(ceil(lo)) - 1, 0);
    var end = min(i32(floor(hi)) + 1, i32(params.line_length) - 1);
    loop {
        if start > end || in_frame(origin, start) {
            break;
        }
        start += 1;
    }
    loop {
        if end < start || in_frame(origin, end) {
            break;
        }
        end -= 1;
    }
    if end < start {
        return vec3<i32>(0, 0, 0);
    }
    return vec3<i32>(start, end, 1);
}

// line_origin is where line index l starts: its signed perpendicular offset
// projected back onto the frame.
fn line_origin(line: u32) -> vec2<f32> {
    let q = params.q_lo + f32(line) * params.q_step;
    return q * vec2<f32>(params.nx, params.ny);
}
