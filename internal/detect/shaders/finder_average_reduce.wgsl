// The second stage of the finder-average reduction: the 64 lanes each of the
// four finder windows left behind, folded into the three channel averages the
// caller actually wants.
//
// It exists so those 4 KB of partials stay on the device. The fold is a few
// hundred adds - far less than the transfer it replaces - and the caller reads
// three floats instead of reducing a buffer it had to be sent first.

const LANES: u32 = 64u;
const FINDERS: u32 = 4u;

// Each lane's partial is three channel sums and the pixel count behind them.
const PARTIAL_WORDS: u32 = 4u;
const PARTIAL_COUNT: u32 = 3u;

@group(0) @binding(0) var<storage, read> partials: array<u32>;
@group(0) @binding(1) var<storage, read_write> average: array<f32>;
@group(0) @binding(2) var<storage, read_write> binarizer_params: array<u32>;

@compute @workgroup_size(3)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
    let channel = local.x;
    if channel >= PARTIAL_COUNT { return; }

    // A finder contributes only where it covered pixels, and only where its
    // mean is positive - a window that averaged to nothing is a window with no
    // signal in it, not a dark one, and counting it would drag the average
    // toward zero.
    var total = 0.0;
    var contributing = 0u;
    for (var finder = 0u; finder < FINDERS; finder += 1u) {
        var sum = 0u;
        var count = 0u;
        for (var lane = 0u; lane < LANES; lane += 1u) {
            let at = (finder * LANES + lane) * PARTIAL_WORDS;
            sum += partials[at + channel];
            count += partials[at + PARTIAL_COUNT];
        }
        if count == 0u { continue; }
        let mean = f32(sum) / f32(count);
        if mean <= 0.0 { continue; }
        total += mean;
        contributing += 1u;
    }
    if contributing == 0u {
        average[channel] = 0.0;
        binarizer_params[6u + channel] = bitcast<u32>(0.0);
        return;
    }
    let value = total / f32(contributing);
    average[channel] = value;
    binarizer_params[6u + channel] = bitcast<u32>(value);
}
