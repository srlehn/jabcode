// Selects default-mode side versions from the two ordered explicit AP searches.
// The existing search kernel measures all candidates in parallel; this stage
// preserves only the host walk's evidence order, prepares the dependent Y-side
// search after X is known, and leaves the confirmed versions resident for grid
// preparation.

const PARAM_NAPX: u32 = 2u;
const PARAM_NAPY: u32 = 3u;
const PARAM_MODE: u32 = 4u;
const PARAM_QUAD: u32 = 11u;
const PARAM_CORNER_MODULE: u32 = 46u;
const PARAM_EXPLICIT: u32 = 50u;
const PARAM_POSITION: u32 = 51u;
const POSITION_WORDS: u32 = 8u;

const MODE_PREDICT: u32 = 0u;
const TILES: u32 = 32u;
const CELL_WORDS: u32 = 6u;
const CELL_FOUND: u32 = 0u;
const CELL_CX: u32 = 1u;
const CELL_CY: u32 = 2u;
const CELL_MODULE: u32 = 3u;
const MAX_CELLS: u32 = 81u;
const CONFIRM_CANDIDATES: u32 = 10u;

const INDIRECT_CONFIRM_Y: u32 = 12u;
const INDIRECT_CONFIRM_Y_FOLD: u32 = 15u;
const INDIRECT_STATE: u32 = 33u;
const INDIRECT_INITIAL_X: u32 = 34u;
const INDIRECT_INITIAL_Y: u32 = 35u;
const INDIRECT_CONFIRMED_X: u32 = 36u;
const INDIRECT_CONFIRMED_Y: u32 = 37u;

const STATE_CONFIRM_X: u32 = 1u;
const STATE_CONFIRM_Y: u32 = 2u;
const STATE_CONFIRMED: u32 = 3u;
const STATE_FAILED: u32 = 4u;
const AMBIGUOUS_MEASUREMENT: u32 = 33u;

const AP_SECOND: array<u32, 32> = array<u32, 32>(
    18u, 22u, 26u, 30u, 34u, 17u, 20u, 23u,
    26u, 14u, 17u, 20u, 23u, 26u, 14u, 17u,
    20u, 23u, 26u, 14u, 17u, 20u, 23u, 26u,
    14u, 17u, 20u, 23u, 26u, 14u, 17u, 20u,
);

@group(0) @binding(0) var<storage, read_write> cells: array<u32>;
@group(0) @binding(1) var<storage, read_write> params: array<u32>;
@group(0) @binding(2) var<storage, read_write> indirect: array<u32>;

fn finite(value: f32) -> bool {
    return value == value && abs(value) <= 3.402823e38;
}

fn point(corner: u32) -> vec2<f32> {
    return vec2<f32>(
        bitcast<f32>(params[PARAM_QUAD + corner * 2u]),
        bitcast<f32>(params[PARAM_QUAD + corner * 2u + 1u]),
    );
}

fn side_for_version(version: u32) -> u32 {
    return 17u + 4u * version;
}

fn walk_version(side_version: i32, target: u32) -> i32 {
    var next_version = side_version;
    var direction = 1;
    var up = 0;
    var down = 0;
    for (var at = 0u; at <= target; at += 1u) {
        let current = next_version;
        if at == target {
            return current;
        }
        direction = -direction;
        if direction == -1 {
            up += 1;
            next_version = up * direction + side_version;
            if next_version < 6 || next_version > 32 {
                direction = -direction;
                up -= 1;
                down += 1;
                next_version = down * direction + side_version;
            }
        } else {
            down += 1;
            next_version = down * direction + side_version;
            if next_version < 6 || next_version > 32 {
                direction = -direction;
                down -= 1;
                up += 1;
                next_version = up * direction + side_version;
            }
        }
    }
    return side_version;
}

fn basis_valid(u: vec2<f32>, u_modules: f32, v: vec2<f32>, v_modules: f32) -> bool {
    if u_modules <= 0.0 || v_modules <= 0.0 {
        return false;
    }
    let a = u / u_modules;
    let b = v / v_modules;
    let an = length(a);
    let bn = length(b);
    return finite(an) && finite(bn) && an > 0.0 && bn > 0.0 &&
        length(a + b) > 0.0 && length(a - b) > 0.0;
}

fn write_candidate(
    index: u32,
    version: i32,
    first_corner: u32,
    second_corner: u32,
    u: vec2<f32>,
    u_modules: f32,
    v: vec2<f32>,
    v_modules: f32,
) -> bool {
    if version < 1 || version > 32 ||
        !basis_valid(u, u_modules, v, v_modules) {
        return false;
    }
    let first = point(first_corner);
    let edge = point(second_corner) - first;
    let edge_length = length(edge);
    let module = bitcast<f32>(params[PARAM_CORNER_MODULE + first_corner]);
    if !finite(edge_length) || edge_length <= 0.0 ||
        !finite(module) || module <= 0.0 {
        return false;
    }
    let module_distance = AP_SECOND[u32(version - 1)] - 4u;
    let centre = first + edge / edge_length * (module * f32(module_distance));
    let base = PARAM_POSITION + index * POSITION_WORDS;
    params[base] = bitcast<u32>(centre.x);
    params[base + 1u] = bitcast<u32>(centre.y);
    params[base + 2u] = bitcast<u32>(module);
    params[base + 3u] = bitcast<u32>(u.x / u_modules);
    params[base + 4u] = bitcast<u32>(u.y / u_modules);
    params[base + 5u] = bitcast<u32>(v.x / v_modules);
    params[base + 6u] = bitcast<u32>(v.y / v_modules);
    params[base + 7u] = bitcast<u32>(2.0 * module);
    return finite(centre.x) && finite(centre.y);
}

fn first_ap_position(first_corner: u32, cell: u32) -> u32 {
    let base = cell * CELL_WORDS;
    let first = point(first_corner);
    let found = vec2<f32>(
        bitcast<f32>(cells[base + CELL_CX]),
        bitcast<f32>(cells[base + CELL_CY]),
    );
    let first_module = bitcast<f32>(params[PARAM_CORNER_MODULE + first_corner]);
    let found_module = bitcast<f32>(cells[base + CELL_MODULE]);
    let delta = abs(found - first);
    let distance = length(delta);
    if !finite(distance) || distance <= 0.0 ||
        !finite(first_module) || !finite(found_module) ||
        first_module <= 0.0 || found_module <= 0.0 {
        return 0u;
    }
    let cos_theta = max(delta.x, delta.y) / distance;
    let mean = (first_module + found_module) * cos_theta / 2.0;
    if !finite(mean) || mean <= 0.0 {
        return 0u;
    }
    let raw = distance / mean;
    let lower = floor(raw);
    // The staged path promotes the measured f32 centre and module to f64 before
    // rounding. Near the half-module boundary f32 cannot prove which integer it
    // would choose, so decline to that path rather than select another version.
    if abs(raw - (lower + 0.5)) <= 0.0001 {
        return AMBIGUOUS_MEASUREMENT;
    }
    var position = i32(floor(raw + 0.5)) + 4;
    let remainder = position % 3;
    if remainder == 0 {
        position -= 1;
    } else if remainder == 1 {
        position += 1;
    }
    if position < 14 || position > 26 {
        return 0u;
    }
    return u32(position);
}

fn confirm_side_version(side_version: u32, position: u32) -> u32 {
    if position == 0u || side_version < 1u || side_version > 32u {
        return 0u;
    }
    let initial = i32(side_version);
    var version = initial;
    var distance = 1;
    var sign = -1;
    loop {
        if AP_SECOND[u32(version - 1)] == position {
            return u32(version);
        }
        version = initial + sign * distance;
        if sign > 0 {
            distance += 1;
        }
        sign = -sign;
        if version < 6 || version > 32 {
            break;
        }
    }
    return 0u;
}

fn select_side_version(side_version: u32, first_a: u32, first_b: u32) -> u32 {
    for (var edge = 0u; edge < 2u; edge += 1u) {
        var position = 0u;
        for (var at = 0u; at < 5u; at += 1u) {
            let cell = edge * 5u + at;
            if cells[cell * CELL_WORDS + CELL_FOUND] == 0u {
                continue;
            }
            position = first_ap_position(select(first_a, first_b, edge == 1u), cell);
            if position == AMBIGUOUS_MEASUREMENT {
                return AMBIGUOUS_MEASUREMENT;
            }
            if position != 0u {
                break;
            }
        }
        if position != 0u {
            let confirmed = confirm_side_version(side_version, position);
            if confirmed != 0u {
                return confirmed;
            }
        }
    }
    return 0u;
}

fn prepare_y_confirmation(version_y: u32, confirmed_x: u32) -> bool {
    let span_y = f32(side_for_version(version_y) - 7u);
    let span_x = f32(side_for_version(confirmed_x) - 7u);
    let across = point(1u) - point(0u);
    let left = point(3u) - point(0u);
    let right = point(2u) - point(1u);
    if !basis_valid(left, span_y, across, span_x) ||
        !basis_valid(right, span_y, across, span_x) {
        return false;
    }
    for (var edge = 0u; edge < 2u; edge += 1u) {
        let first = select(0u, 1u, edge == 1u);
        let second = select(3u, 2u, edge == 1u);
        let along = select(left, right, edge == 1u);
        for (var at = 0u; at < 5u; at += 1u) {
            let version = walk_version(i32(version_y), at);
            if !write_candidate(
                edge * 5u + at, version, first, second,
                along, span_y, across, span_x,
            ) {
                return false;
            }
        }
    }
    params[PARAM_NAPX] = CONFIRM_CANDIDATES;
    params[PARAM_NAPY] = 1u;
    params[PARAM_MODE] = MODE_PREDICT;
    params[PARAM_EXPLICIT] = 1u;
    return true;
}

fn command(at: u32, x: u32) {
    indirect[at] = x;
    indirect[at + 1u] = 1u;
    indirect[at + 2u] = 1u;
}

var<workgroup> selected_version: u32;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
    let lane = local.x;
    let state = indirect[INDIRECT_STATE];
    if state != STATE_CONFIRM_X && state != STATE_CONFIRM_Y {
        return;
    }
    if lane == 0u {
        selected_version = select_side_version(
            select(indirect[INDIRECT_INITIAL_X], indirect[INDIRECT_INITIAL_Y], state == STATE_CONFIRM_Y),
            0u,
            select(3u, 1u, state == STATE_CONFIRM_Y),
        );
    }
    workgroupBarrier();
    if lane < MAX_CELLS {
        for (var word = 0u; word < CELL_WORDS; word += 1u) {
            cells[lane * CELL_WORDS + word] = 0u;
        }
    }
    storageBarrier();
    workgroupBarrier();
    if lane != 0u {
        return;
    }
    if selected_version == 0u || selected_version == AMBIGUOUS_MEASUREMENT {
        indirect[INDIRECT_STATE] = STATE_FAILED;
        return;
    }
    if state == STATE_CONFIRM_X {
        indirect[INDIRECT_CONFIRMED_X] = selected_version;
        if !prepare_y_confirmation(indirect[INDIRECT_INITIAL_Y], selected_version) {
            indirect[INDIRECT_STATE] = STATE_FAILED;
            return;
        }
        indirect[INDIRECT_STATE] = STATE_CONFIRM_Y;
        command(INDIRECT_CONFIRM_Y, CONFIRM_CANDIDATES * TILES);
        command(INDIRECT_CONFIRM_Y_FOLD, CONFIRM_CANDIDATES);
        return;
    }
    indirect[INDIRECT_CONFIRMED_Y] = selected_version;
    indirect[INDIRECT_STATE] = STATE_CONFIRMED;
}
