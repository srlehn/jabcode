// The five-run test, and the off-line walks that re-apply it through a
// candidate's centre.
//
// A window that passes along the scan line is a run-length coincidence until
// something confirms it in a direction the scan never walked. The CPU detector
// does that with crossCheckPatternAlong, on the host, per hit, after the hits
// have been counted, sorted and handed back - which is most of what makes a
// directional read expensive. Here it runs inside the kernel that produced the
// candidate, so a window that cannot be confirmed never becomes a record and
// never crosses a bus.
//
// The walk is deliberately not a port. The host version carries a slack rule
// that folds runs shorter than a few samples into their neighbours and a
// five-state machine that can step backwards when it does; both exist because it
// resumes a single serial walk across a whole line. A walk that starts at a
// known centre and only has to grow outwards needs neither, and the fused window
// stage already declines to fold for the same reason.

// accept is checkPatternCross: three inner runs within half a layer of their
// mean, two outer runs at least a quarter layer, and the first and third inner
// runs equal to within the same tolerance. Both the along-line window test and
// the off-line walks below decide with it, so a candidate meets one rule rather
// than two that drift apart.
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

// walk_side grows three runs outwards from a centre: the rest of the centre's
// own run, then the two beyond it. It stops at the third colour change, at the
// frame edge, or when a run gets long enough that the verdict is already
// settled, and returns the three counts with the number of changes it saw. Two
// changes means the outer run was entered, which is all the signature needs - a
// run truncated by the frame edge still counts, exactly as the host walk allows.
//
// **The per-run limits change no verdict.** A candidate is kept only if its
// perpendicular module size is within twice the along-line one, so a walk may
// assume that bound and reject anything needing more. Each inner run must then
// be under three times the along-line layer, since a run more than half a layer
// off its own mean fails the test; and each outer run only has to exceed a
// quarter of the perpendicular layer, which half the along-line layer already
// does. Walking further can only confirm what is decided. What this saves is the
// walk through a flat region, where nothing changes colour and the old fixed cap
// was paid in full every time.
fn walk_side(
    centre: vec2<f32>,
    step: vec2<f32>,
    channel: u32,
    mid: u32,
    inner_cap: u32,
    outer_cap: u32,
) -> vec4<u32> {
    var counts = vec3<u32>(0u, 0u, 0u);
    var stage = 0u;
    var prev = mid;
    var cap = inner_cap;
    for (var i = 1u; i <= 4096u; i++) {
        let v = mask_at(centre + f32(i) * step, channel);
        if v > 1u {
            break;
        }
        if v != prev {
            stage += 1u;
            if stage > 2u {
                break;
            }
            prev = v;
            if stage == 2u {
                cap = outer_cap;
            }
        }
        counts[stage] += 1u;
        if counts[stage] >= cap {
            if stage < 2u {
                // An inner run this long cannot be part of an accepted
                // signature, and the walk has nothing left to learn.
                return vec4<u32>(counts, 0u);
            }
            break;
        }
    }
    return vec4<u32>(counts, stage);
}

// cross_layer walks both ways from a centre and reports the module size the two
// halves imply, or a negative value when no finder signature is there. The
// centre sample belongs to the middle run and is counted once.
//
// The caps come from the module size the along-line window already implies,
// which is sound because a finder's runs are the same size in every direction
// through its centre and a candidate whose perpendicular disagrees by more than
// a factor of two is rejected anyway.
fn cross_layer(centre: vec2<f32>, step: vec2<f32>, channel: u32, layer: f32) -> f32 {
    let mid = mask_at(centre, channel);
    if mid > 1u {
        return -1.0;
    }
    let inner_cap = min(u32(layer * 3.0) + 2u, 4096u);
    let outer_cap = min(u32(layer * 0.5) + 2u, 4096u);
    let back = walk_side(centre, -step, channel, mid, inner_cap, outer_cap);
    let fwd = walk_side(centre, step, channel, mid, inner_cap, outer_cap);
    if back.w < 2u || fwd.w < 2u {
        return -1.0;
    }
    let s2 = back.x + fwd.x + 1u;
    if !accept(back.z, back.y, s2, fwd.y, fwd.z) {
        return -1.0;
    }
    return f32(back.y + s2 + fwd.y) / 3.0;
}

// cross_step scales a unit direction to the scan's own sample spacing, so runs
// counted off the line are in the same units as runs counted along it and the
// two module sizes can be compared at all.
fn cross_step(unit: vec2<f32>) -> vec2<f32> {
    return length(vec2<f32>(params.dx, params.dy)) * unit;
}

// agrees is the verdict on one off-line walk: it has to find the signature and
// imply a module size within twice the along-line one.
//
// **Every walk is held to this, not only the perpendicular**, because the bound
// is what makes walk_side's early exits sound. Those exits assume a candidate
// needing more than twice the along-line module is rejected anyway; a walk
// judged without the bound could be failed by a truncated outer run instead. The
// bound is one-sided because it is the off-line walk that can run away: a wide
// band of background crossing a narrow finder gives a perfectly proportioned
// signature at the wrong scale.
fn agrees(measured: f32, layer: f32) -> bool {
    return measured >= 0.0 && measured <= 2.0 * layer;
}
