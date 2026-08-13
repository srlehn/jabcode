// Reduces the resident channel-offset score table into the three displacements
// consumed by the symbol sampler. Keeping this decision beside the scores is
// what lets a print primary remain inside the one-result-download batch.

const CANDIDATE_SIDE: u32 = 9u;
const CANDIDATES: u32 = CANDIDATE_SIDE * CANDIDATE_SIDE;
const NOMINAL: u32 = 40u;
const MIN_GAIN: f32 = 0.25;

const PARAM_MOD_W: u32 = 5u;
const PARAM_MOD_H: u32 = 6u;
const PARAM_GRID: u32 = 17u;

const SAMPLE_USE_DELTA: u32 = 7u;
const SAMPLE_DELTA: u32 = 17u;

@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;
@group(0) @binding(2) var<storage, read_write> sample: array<u32>;

fn scalar(index: u32) -> f32 {
    return bitcast<f32>(params[index]);
}

fn score(channel: u32, candidate: u32, parity: u32) -> f32 {
    return scores[(parity * 3u + channel) * CANDIDATES + candidate];
}

@compute @workgroup_size(1)
fn main() {
    let mod_w = scalar(PARAM_MOD_W);
    let mod_h = scalar(PARAM_MOD_H);
    var any = false;
    for (var channel = 0u; channel < 3u; channel += 1u) {
        var best_x = array<f32, 2>(0.0, 0.0);
        var best_y = array<f32, 2>(0.0, 0.0);
        for (var parity = 0u; parity < 2u; parity += 1u) {
            let base = score(channel, NOMINAL, parity);
            var best = base;
            for (var candidate = 0u; candidate < CANDIDATES; candidate += 1u) {
                if candidate == NOMINAL {
                    continue;
                }
                let candidate_score = score(channel, candidate, parity);
                if candidate_score < best && candidate_score < base * (1.0 - MIN_GAIN) {
                    best = candidate_score;
                    best_x[parity] = scalar(PARAM_GRID + candidate % CANDIDATE_SIDE) * mod_w;
                    best_y[parity] = scalar(PARAM_GRID + candidate / CANDIDATE_SIDE) * mod_h;
                }
            }
        }
        var dx = 0.0;
        var dy = 0.0;
        if abs(best_x[0] - best_x[1]) <= mod_w * 0.15 &&
            abs(best_y[0] - best_y[1]) <= mod_h * 0.15 {
            dx = (best_x[0] + best_x[1]) * 0.5;
            dy = (best_y[0] + best_y[1]) * 0.5;
        }
        sample[SAMPLE_DELTA + channel * 2u] = bitcast<u32>(dx);
        sample[SAMPLE_DELTA + channel * 2u + 1u] = bitcast<u32>(dy);
        any = any || dx != 0.0 || dy != 0.0;
    }
    sample[SAMPLE_USE_DELTA] = select(0u, 1u, any);
}
