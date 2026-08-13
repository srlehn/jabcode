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
//
// **A window that passes is then cross-checked before it becomes a record**, by
// re-walking through its centre off the scan line and demanding the same
// signature at a comparable module size. That is what makes the output
// candidates rather than run-length coincidences, and doing it here rather than
// on the host is the point: the rejected ones never occupy a record, an atomic
// slot or a bus.
//
// **Three directions are tried and any one of them confirms**: the
// perpendicular, then each diagonal. The host chain accepts on diagonal evidence
// when the perpendicular walk fails, so a device stage requiring the
// perpendicular would reject finders the CPU route accepts - which is the one
// trade this pipeline may not make. The record carries which walk confirmed it,
// so a later stage can rank on that rather than rediscover it.
//
// **The route does not use any of that.** It dispatches with
// FLAG_SKIP_CROSS_CHECK, because even a three-direction gate confirms on the
// seek channel while the CPU chain confirms on the other two, and measurement on
// real captures shows it rejecting candidates that chain keeps. So this stage
// ships as a candidate generator and the host chain does the rejecting it
// already does. The walks stay because deciding on the device is where this is
// going and the harness needs them to say when it is safe.
//
// Four counters. counters[0] is the record count and the base every block's
// reservation is taken from, so it stays first whatever else is added:
//
//   - counters[0] is how many records the dispatch *required*: the
//     cross-checked candidates ordinarily, every accepted window under
//     FLAG_EMIT_UNCONFIRMED or FLAG_SKIP_CROSS_CHECK. It counts reservations,
//     never verdicts, and it is
//     never clamped to the buffer - a block reserves its slots before anything
//     checks whether they fit, so a run that overflowed reports the true count
//     and writes fewer. That is deliberate: a clamped count would make a
//     truncated result indistinguishable from a complete one.
//   - counters[1] is the cross-checked candidates whose inner runs are at least
//     three samples, which is the subset the CPU sweep's run folding could also
//     have reached. Whether the rest contains anything real is a question for a
//     decode, and the split exists so it can be asked rather than assumed.
//   - counters[2] is the cross-checked candidates the perpendicular failed and a
//     diagonal rescued, which is exactly the class a perpendicular-only gate
//     would have lost.
//   - counters[3] is the windows that passed along the line *before* the
//     cross-check. It is the denominator of the only ratio this stage is really
//     about - how much the cross-check removes - and inferring that from a
//     separate run of a different kernel is how a measurement stops being one.
//
// **Only counters[0] follows the emission rule.** The other three count verdicts
// and are therefore identical with the flag set or clear, which is what lets one
// unfiltered run report both populations. It also means counters[0] equals
// counters[3] in that mode and is a subset of it otherwise; do not read the pair
// as a fixed containment.

@group(0) @binding(1) var<storage, read_write> survivors: array<u32>;
@group(0) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;

const WORKGROUP: u32 = 256u;
// One record: key, the six boundaries that define the window, and the module
// size the confirming walk measured. That last word was padding, kept only to
// make the stride a power of two; carrying the cross-check's own module estimate
// in it gives the host a second, independent measurement for free.
//
// Under FLAG_EMIT_UNCONFIRMED, or FLAG_SKIP_CROSS_CHECK where no walk runs at
// all, a record can be a window nothing confirmed. Its evidence field is zero
// and its module size negative, so a reader tells the two apart from the record
// itself rather than from how the kernel was dispatched.
const RECORD: u32 = 8u;

// Which walk confirmed the candidate, in the key word's top two bits, so the
// record keeps its power-of-two stride.
//
// **Two bits, not eight.** A key is line * 3 + channel, and a sweep's line count
// is bounded by the frame's own extent, so the key cannot exceed 3 * (width +
// height). Thirty bits therefore hold the key for any frame under about 119
// megapixels, where a byte-wide field would have started colliding at around 12
// on a pathological aspect ratio. The two diagonals are told apart because the
// joining diagonal's orientation identifies the finder type, and a stage that
// had to rediscover it would be redoing this walk.
const EVIDENCE_SHIFT: u32 = 30u;
const EVIDENCE_PERPENDICULAR: u32 = 1u;
const EVIDENCE_DIAGONAL_RIGHT: u32 = 2u;
const EVIDENCE_DIAGONAL_LEFT: u32 = 3u;

// values holds each lane's sample so its right-hand neighbour can read it
// instead of loading it again.
var<workgroup> values: array<u32, WORKGROUP>;
// bpos holds the five carried boundaries followed by this block's own.
var<workgroup> bpos: array<u32, 5u + WORKGROUP>;
var<workgroup> carry: u32;
var<workgroup> block_slot: atomic<u32>;
var<workgroup> block_strict: atomic<u32>;
var<workgroup> block_square: atomic<u32>;
var<workgroup> block_windows: atomic<u32>;
var<workgroup> block_base: u32;

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

// flush_block tests and cross-checks every window this block closed, then rolls
// the carry. The window count is exactly valid-5, and every window it covers
// ends on a boundary this block produced, so no window is tested twice and none
// is skipped at a seam.
//
// The cross-check walks sit inside the same divergent branch as the window test
// and contain no barrier or subgroup operation, so only the lanes holding a
// candidate pay for them.
fn flush_block(origin: vec2<f32>, channel: u32, key: u32, n: u32, lane: u32) {
    let c = carry;
    let valid = c + n;
    let base = 5u - c;

    if lane == 0u {
        atomicStore(&block_slot, 0u);
        atomicStore(&block_strict, 0u);
        atomicStore(&block_square, 0u);
        atomicStore(&block_windows, 0u);
    }
    workgroupBarrier();

    var hit = false;
    var confirmed = false;
    var strict = false;
    var square = false;
    var perp = 0.0;
    var evidence = 0u;
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
        if accept(s0, s1, s2, s3, s4) {
            atomicAdd(&block_windows, 1u);
            if (params.flags & FLAG_SKIP_CROSS_CHECK) != 0u {
                // Candidate generation only: no walk ran, so there is no verdict
                // and no measurement to record.
                hit = true;
                perp = -1.0;
            } else {
                let layer = f32(s1 + s2 + s3) / 3.0;
                let along = vec2<f32>(params.dx, params.dy);
                let centre_index =
                    (f32(bitcast<i32>(b[2])) + f32(bitcast<i32>(b[3]))) * 0.5;
                let centre = origin + centre_index * along;
                let normal = vec2<f32>(params.nx, params.ny);
                let unit = normalize(along);

                // The perpendicular is tried first because it is the evidence a
                // finder's own square reference carries most reliably, and because
                // stopping there costs one walk instead of three.
                perp = cross_layer(centre, cross_step(normal), channel, layer).x;
                if agrees(perp, layer) {
                    confirmed = true;
                    evidence = EVIDENCE_PERPENDICULAR;
                } else {
                    // **A failed perpendicular is not a rejection.** The host chain
                    // accepts a candidate on diagonal evidence alone when the
                    // perpendicular walk fails, and a device stage that did not
                    // would lose finders the CPU route finds. Only one diagonal can
                    // carry the signature: a JAB finder is two square references
                    // joined along one diagonal, so the other passes through the
                    // core and straight out into background.
                    //
                    // The diagonal has to confirm *twice*, from its own refined
                    // centre, which is the same bar the host sets on this branch.
                    // One unconfirmed walk is far weaker and admits most of what a
                    // dense pattern produces.
                    perp = cross_confirm(centre, cross_step(normalize(unit + normal)), channel, layer);
                    if perp >= 0.0 {
                        confirmed = true;
                        evidence = EVIDENCE_DIAGONAL_RIGHT;
                    } else {
                        perp = cross_confirm(centre, cross_step(normalize(unit - normal)), channel, layer);
                        if perp >= 0.0 {
                            confirmed = true;
                            evidence = EVIDENCE_DIAGONAL_LEFT;
                        } else {
                            // An unconfirmed record carries no module measurement.
                            // Leaving the last failed walk's return in place would
                            // make a rejected window look measured, and the reader
                            // has no way to tell the two apart.
                            perp = -1.0;
                        }
                    }
                }
                // The counters keep meaning the confirmed subsets whatever the
                // emission rule is, so an unfiltered run reports both populations at
                // once: counters[0] is every record and counters[1..2] stay the
                // cross-check's own.
                strict = confirmed && s1 >= 3u && s2 >= 3u && s3 >= 3u;
                square = confirmed && evidence != EVIDENCE_PERPENDICULAR;
                hit = confirmed || (params.flags & FLAG_EMIT_UNCONFIRMED) != 0u;
            }
        }
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
    if square {
        atomicAdd(&block_square, 1u);
    }
    workgroupBarrier();
    if lane == 0u {
        let total = atomicLoad(&block_slot);
        block_base = atomicAdd(&counters[0], total);
        atomicAdd(&counters[1], atomicLoad(&block_strict));
        atomicAdd(&counters[2], atomicLoad(&block_square));
        atomicAdd(&counters[3], atomicLoad(&block_windows));
    }
    workgroupBarrier();
    if hit {
        let index = block_base + local;
        if (index + 1u) * RECORD <= arrayLength(&survivors) {
            let at = index * RECORD;
            survivors[at] = key | (evidence << EVIDENCE_SHIFT);
            for (var k = 0u; k < 6u; k++) {
                survivors[at + 1u + k] = b[k];
            }
            survivors[at + 7u] = bitcast<u32>(perp);
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
