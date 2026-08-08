// Scores one candidate plane displacement per workgroup, where the balanced
// pixels already are. The host search reads the whole frame to do this, which is
// the largest single transfer a print capture pays; here the frame never moves
// and only three offsets leave the device.
//
// One workgroup owns one (candidate, channel, parity) triple. The candidates are
// independent and there are a few hundred of them, so the dispatch is the
// parallelism and each workgroup only has to reduce its own module population.
//
// The score is the host's: each module's value is a nine-tap footprint average,
// and the score is the mean distance of those values to the nearer of the
// channel's low and high deciles - small when the population is bimodal, which
// is what a correctly registered plane looks like.

const WORKGROUP: u32 = 256u;

// Values are byte averages, so they span [0,256). Binning them at quarter
// resolution puts the deciles within an eighth of a level of the host's sorted
// answer, against a search that only adopts a winner beating the nominal
// position by a quarter of its score. Storing the values instead would need one
// workgroup array per module and there are thousands of them.
const BINS: u32 = 1024u;
const BIN_SCALE: f32 = 4.0;

const PARAM_WIDTH: u32 = 0u;
const PARAM_HEIGHT: u32 = 1u;
const PARAM_SIDE_X: u32 = 2u;
const PARAM_SIDE_Y: u32 = 3u;
const PARAM_CANDIDATES: u32 = 4u;
const PARAM_MOD_W: u32 = 5u;
const PARAM_MOD_H: u32 = 6u;
const PARAM_MIN_RANGE: u32 = 7u;
const PARAM_TRANSFORM: u32 = 8u;
const PARAM_GRID: u32 = 17u;

@group(0) @binding(0) var<storage, read> image: array<u32>;
@group(0) @binding(1) var<storage, read_write> scores: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;

var<workgroup> bins: array<atomic<u32>, BINS>;
var<workgroup> lane_sum: array<f32, WORKGROUP>;
var<workgroup> lane_count: array<u32, WORKGROUP>;
var<workgroup> decile_lo: f32;
var<workgroup> decile_hi: f32;
var<workgroup> flat: u32;

fn word(index: u32) -> u32 {
    return params[index];
}

fn scalar(index: u32) -> f32 {
    return bitcast<f32>(params[index]);
}

// warp applies the perspective transform the host built from the four finder
// centres, in the coefficient order core.Perspective.Coefficients packs and
// sample_symbol.wgsl already reads.
fn warp(x: f32, y: f32) -> vec2<f32> {
    let a11 = scalar(PARAM_TRANSFORM + 0u);
    let a12 = scalar(PARAM_TRANSFORM + 1u);
    let a13 = scalar(PARAM_TRANSFORM + 2u);
    let a21 = scalar(PARAM_TRANSFORM + 3u);
    let a22 = scalar(PARAM_TRANSFORM + 4u);
    let a23 = scalar(PARAM_TRANSFORM + 5u);
    let a31 = scalar(PARAM_TRANSFORM + 6u);
    let a32 = scalar(PARAM_TRANSFORM + 7u);
    let a33 = scalar(PARAM_TRANSFORM + 8u);
    let denom = a13 * x + a23 * y + a33;
    return vec2<f32>(
        (a11 * x + a21 * y + a31) / denom,
        (a12 * x + a22 * y + a32) / denom,
    );
}

// footprint is the module's value: the mean of nine taps at the host's fixed
// fractions of a module. A single pixel on a halftoned print is ink or paper
// whatever the offset, so a point sample would score the screen rather than the
// displacement.
fn footprint(cx: f32, cy: f32, channel: u32, modW: f32, modH: f32) -> f32 {
    let width = i32(word(PARAM_WIDTH));
    let height = i32(word(PARAM_HEIGHT));
    var sum = 0.0;
    for (var t = 0u; t < 9u; t += 1u) {
        var fx = 0.0;
        var fy = 0.0;
        switch t {
            case 1u: { fx = -0.2; }
            case 2u: { fx = 0.2; }
            case 3u: { fy = -0.2; }
            case 4u: { fy = 0.2; }
            case 5u: { fx = -0.2; fy = -0.2; }
            case 6u: { fx = 0.2; fy = -0.2; }
            case 7u: { fx = -0.2; fy = 0.2; }
            case 8u: { fx = 0.2; fy = 0.2; }
            default: {}
        }
        // The host truncates toward zero and then clamps, so a negative
        // coordinate lands on column or row zero either way.
        let px = clamp(i32(cx + fx * modW), 0, width - 1);
        let py = clamp(i32(cy + fy * modH), 0, height - 1);
        let texel = image[u32(py * width + px)];
        sum += f32((texel >> (channel * 8u)) & 0xffu);
    }
    return sum / 9.0;
}

// moduleValue returns the nth scored module's value for this candidate, where n
// counts only the modules of this parity. The host walks a stride-two subset of
// the grid and then takes every other one of those, so the parity halves are
// interleaved and a winner has to hold on both.
fn moduleValue(
    n: u32, parity: u32, channel: u32,
    dx: f32, dy: f32, modW: f32, modH: f32,
) -> f32 {
    let sideX = word(PARAM_SIDE_X);
    let spotsPerRow = (sideX + 1u) / 2u;
    let k = n * 2u + parity;
    let row = (k / spotsPerRow) * 2u;
    let col = (k % spotsPerRow) * 2u;
    let p = warp(f32(col) + 0.5, f32(row) + 0.5);
    return footprint(p.x + dx, p.y + dy, channel, modW, modH);
}

fn spotCount(parity: u32) -> u32 {
    let sideX = word(PARAM_SIDE_X);
    let sideY = word(PARAM_SIDE_Y);
    let total = ((sideX + 1u) / 2u) * ((sideY + 1u) / 2u);
    if parity == 0u {
        return (total + 1u) / 2u;
    }
    return total / 2u;
}

@compute @workgroup_size(WORKGROUP)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let candidates = word(PARAM_CANDIDATES);
    let slot = wid.x;
    let candidate = slot % candidates;
    let channel = (slot / candidates) % 3u;
    let parity = (slot / candidates) / 3u;

    let modW = scalar(PARAM_MOD_W);
    let modH = scalar(PARAM_MOD_H);
    let side = u32(sqrt(f32(candidates)) + 0.5);
    let dx = scalar(PARAM_GRID + candidate % side) * modW;
    let dy = scalar(PARAM_GRID + candidate / side) * modH;

    let count = spotCount(parity);
    for (var b = lane; b < BINS; b += WORKGROUP) {
        atomicStore(&bins[b], 0u);
    }
    workgroupBarrier();

    for (var n = lane; n < count; n += WORKGROUP) {
        let v = moduleValue(n, parity, channel, dx, dy, modW, modH);
        let bin = min(u32(v * BIN_SCALE), BINS - 1u);
        atomicAdd(&bins[bin], 1u);
    }
    workgroupBarrier();

    // One lane walks the histogram for the two deciles. It is a thousand adds
    // against the thousands of footprint reads above, so splitting it would buy
    // nothing and would need a scan.
    if lane == 0u {
        let loTarget = count / 10u;
        let hiTarget = count - 1u - count / 10u;
        var seen = 0u;
        var lo = 0.0;
        var hi = 0.0;
        var haveLo = false;
        for (var b = 0u; b < BINS; b += 1u) {
            let n = atomicLoad(&bins[b]);
            if n == 0u {
                continue;
            }
            if !haveLo && seen + n > loTarget {
                lo = f32(b) / BIN_SCALE;
                haveLo = true;
            }
            if seen + n > hiTarget {
                hi = f32(b) / BIN_SCALE;
                break;
            }
            seen += n;
        }
        decile_lo = lo;
        decile_hi = hi;
        // A flat channel has no modes to sharpen, and the host reports the same
        // by scoring it out of the running entirely.
        if hi - lo < scalar(PARAM_MIN_RANGE) {
            flat = 1u;
        } else {
            flat = 0u;
        }
    }
    workgroupBarrier();

    if flat == 1u {
        if lane == 0u {
            scores[slot] = bitcast<f32>(0x7f800000u);
        }
        return;
    }

    let lo = decile_lo;
    let hi = decile_hi;
    var sum = 0.0;
    var mine = 0u;
    for (var n = lane; n < count; n += WORKGROUP) {
        let v = moduleValue(n, parity, channel, dx, dy, modW, modH);
        sum += min(abs(v - lo), abs(v - hi));
        mine += 1u;
    }
    lane_sum[lane] = sum;
    lane_count[lane] = mine;
    workgroupBarrier();

    for (var stride = WORKGROUP / 2u; stride > 0u; stride = stride >> 1u) {
        if lane < stride {
            lane_sum[lane] += lane_sum[lane + stride];
            lane_count[lane] += lane_count[lane + stride];
        }
        workgroupBarrier();
    }
    if lane == 0u {
        if lane_count[0] == 0u {
            scores[slot] = bitcast<f32>(0x7f800000u);
        } else {
            scores[slot] = lane_sum[0] / f32(lane_count[0]);
        }
    }
}
