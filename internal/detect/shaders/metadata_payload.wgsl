// Publishes the interpreted metadata record directly into payload control.
// The host currently downloads that record and rebuilds the same palette and
// shape parameters before uploading them again. Keeping this conversion next to
// the record removes that round trip while preserving the metadata walk as the
// one owner of the selected colour mode, ECC pair and palette.

const WORKGROUP: u32 = 256u;

const STATUS_OK: u32 = 0u;
const STATUS_DEFAULT: u32 = 1u;
const STATUS_SIZE_MISMATCH: u32 = 4u;
const STATUS_ECC_ORDER: u32 = 5u;

const ADMISSION_OPEN: u32 = 0u;
const ADMISSION_PENDING: u32 = 1u;
const ADMISSION_REJECTED: u32 = 2u;

const PARAM_SIDE_X: u32 = 0u;
const PARAM_SIDE_Y: u32 = 1u;
const PARAM_GENERATOR: u32 = 3u;
const PARAM_DEFAULT_WC: u32 = 4u;
const PARAM_DEFAULT_WR: u32 = 5u;
const PARAM_DEFAULT_MASK: u32 = 6u;
const PARAM_MODE: u32 = 8u;
const MODE_WORDS: u32 = 5u;
const MODE_COPIES: u32 = 0u;
const PARAM_PAYLOAD_AP_NUM_X: u32 = 576u;
const PARAM_PAYLOAD_AP_NUM_Y: u32 = 577u;
const PARAM_PAYLOAD_AP_POS_X: u32 = 578u;
const PARAM_PAYLOAD_AP_POS_Y: u32 = 587u;

const RECORD_STATUS: u32 = 0u;
const RECORD_MODULES: u32 = 1u;
const RECORD_NC: u32 = 4u;
const RECORD_COLORS: u32 = 5u;
const RECORD_ECL_X: u32 = 8u;
const RECORD_ECL_Y: u32 = 9u;
const RECORD_MASK: u32 = 10u;
const RECORD_PART1_SYNDROME: u32 = 12u;
const RECORD_PART2_SYNDROME: u32 = 13u;
const RECORD_PALETTE: u32 = 16u;
const RECORD_NORMALIZED: u32 = 1552u;
const RECORD_THRESHOLDS: u32 = 1680u;

const PAYLOAD_SIDE_X: u32 = 0u;
const PAYLOAD_SIDE_Y: u32 = 1u;
const PAYLOAD_META_MODULES: u32 = 2u;
const PAYLOAD_COLORS: u32 = 3u;
const PAYLOAD_BITS: u32 = 4u;
const PAYLOAD_MASK: u32 = 5u;
const PAYLOAD_AP_NUM_X: u32 = 6u;
const PAYLOAD_AP_NUM_Y: u32 = 7u;
const PAYLOAD_GROSS_BITS: u32 = 8u;
const PAYLOAD_GENERATOR: u32 = 9u;
const PAYLOAD_COPIES: u32 = 10u;
const PAYLOAD_SYMBOL_TYPE: u32 = 11u;
const PAYLOAD_AP_POS_X: u32 = 12u;
const PAYLOAD_AP_POS_Y: u32 = 21u;
const PAYLOAD_THRESHOLDS: u32 = 30u;
const PAYLOAD_EXTREMES: u32 = 42u;
const PAYLOAD_NORMALIZED: u32 = 50u;
const PAYLOAD_PALETTE_BYTES: u32 = 178u;
const PAYLOAD_ADMISSION: u32 = 1714u;
const PAYLOAD_DATA_MODULES: u32 = 1715u;
const PAYLOAD_NET_BITS: u32 = 1716u;
const PAYLOAD_WC: u32 = 1717u;
const PAYLOAD_WR: u32 = 1718u;
const PAYLOAD_PALETTE_SEPARATION: u32 = 1719u;
const PAYLOAD_PALETTE_DISAGREEMENT: u32 = 1720u;
const PAYLOAD_EVIDENCE_FLAGS: u32 = 1723u;
const PAYLOAD_WORDS: u32 = 1724u;

const EVIDENCE_FIXED_PATTERN: u32 = 1u;

const LDPC_ADMISSION: u32 = 13u;

@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> record: array<u32>;
@group(0) @binding(2) var<storage, read_write> payload: array<u32>;
@group(0) @binding(3) var<storage, read_write> ldpc: array<u32>;

fn palette_rgb(copy: u32, color: u32, colors: u32) -> vec3<f32> {
    let at = RECORD_PALETTE + (copy * colors + color) * 3u;
    return vec3<f32>(f32(record[at]), f32(record[at + 1u]), f32(record[at + 2u]));
}

fn mean_rgb(color: u32, colors: u32, copies: u32) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    for (var copy = 0u; copy < copies; copy += 1u) {
        sum += palette_rgb(copy, color, colors);
    }
    return floor(sum / f32(copies));
}

// palette_admitted is the resident form of decode.paletteAdmitted. It keeps the
// admission decision beside the palette bytes, so a default or weak-syndrome
// observation does not need a metadata download merely to decide whether fixed
// pattern checking should run.
fn palette_evidence(colors: u32, copies: u32) -> vec2<f32> {
    if colors < 2u || copies == 0u {
        return vec2<f32>(0.0, 3.402823e38);
    }
    var disagreement = 0.0;
    var pairs = 0u;
    if copies > 1u {
        for (var color = 0u; color < colors; color += 1u) {
            for (var a = 0u; a < copies; a += 1u) {
                for (var b = a + 1u; b < copies; b += 1u) {
                    disagreement += distance(
                        palette_rgb(a, color, colors),
                        palette_rgb(b, color, colors),
                    );
                    pairs += 1u;
                }
            }
        }
        disagreement /= f32(pairs);
    }

    var separation = 3.402823e38;
    for (var a = 0u; a < colors; a += 1u) {
        let mean_a = mean_rgb(a, colors, copies);
        for (var b = a + 1u; b < colors; b += 1u) {
            separation = min(separation, distance(mean_a, mean_rgb(b, colors, copies)));
        }
    }
    return vec2<f32>(separation, disagreement);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
    let lane = local.x;
    for (var at = lane; at < PAYLOAD_WORDS; at += WORKGROUP) {
        payload[at] = 0u;
    }
    storageBarrier();
    workgroupBarrier();

    let status = record[RECORD_STATUS];
    let nc = record[RECORD_NC];
    let colors = record[RECORD_COLORS];
    if nc >= 8u || colors < 4u || colors > 256u || (colors & (colors - 1u)) != 0u {
        if lane == 0u {
            payload[PAYLOAD_ADMISSION] = ADMISSION_REJECTED;
            ldpc[LDPC_ADMISSION] = ADMISSION_REJECTED;
        }
        return;
    }
    let copies = params[PARAM_MODE + nc * MODE_WORDS + MODE_COPIES];

    for (var at = lane; at < 12u; at += WORKGROUP) {
        payload[PAYLOAD_THRESHOLDS + at] = record[RECORD_THRESHOLDS + at];
    }
    if colors <= 8u {
        let entries = colors * copies * 4u;
        for (var at = lane; at < entries; at += WORKGROUP) {
            payload[PAYLOAD_NORMALIZED + at] = record[RECORD_NORMALIZED + at];
        }
    } else {
        let bytes = colors * copies * 3u;
        for (var at = lane; at < bytes; at += WORKGROUP) {
            payload[PAYLOAD_PALETTE_BYTES + at] = record[RECORD_PALETTE + at];
        }
    }
    for (var at = lane; at < 9u; at += WORKGROUP) {
        payload[PAYLOAD_AP_POS_X + at] = params[PARAM_PAYLOAD_AP_POS_X + at];
        payload[PAYLOAD_AP_POS_Y + at] = params[PARAM_PAYLOAD_AP_POS_Y + at];
    }

    if lane != 0u {
        return;
    }
    payload[PAYLOAD_SIDE_X] = params[PARAM_SIDE_X];
    payload[PAYLOAD_SIDE_Y] = params[PARAM_SIDE_Y];
    payload[PAYLOAD_META_MODULES] = record[RECORD_MODULES];
    payload[PAYLOAD_COLORS] = colors;
    payload[PAYLOAD_BITS] = nc + 1u;
    payload[PAYLOAD_MASK] = select(record[RECORD_MASK], params[PARAM_DEFAULT_MASK], status == STATUS_DEFAULT);
    payload[PAYLOAD_AP_NUM_X] = params[PARAM_PAYLOAD_AP_NUM_X];
    payload[PAYLOAD_AP_NUM_Y] = params[PARAM_PAYLOAD_AP_NUM_Y];
    payload[PAYLOAD_GENERATOR] = params[PARAM_GENERATOR];
    payload[PAYLOAD_COPIES] = copies;
    payload[PAYLOAD_SYMBOL_TYPE] = 0u;
    payload[PAYLOAD_GROSS_BITS] = 0u;
    payload[PAYLOAD_DATA_MODULES] = 0u;
    payload[PAYLOAD_NET_BITS] = 0u;
    payload[PAYLOAD_WC] = select(record[RECORD_ECL_X], params[PARAM_DEFAULT_WC], status == STATUS_DEFAULT);
    payload[PAYLOAD_WR] = select(record[RECORD_ECL_Y], params[PARAM_DEFAULT_WR], status == STATUS_DEFAULT);

    if colors == 8u {
        for (var copy = 0u; copy < copies; copy += 1u) {
            let first = RECORD_PALETTE + copy * colors * 3u;
            let last = first + 7u * 3u;
            payload[PAYLOAD_EXTREMES + copy * 2u] =
                record[first] + record[first + 1u] + record[first + 2u];
            payload[PAYLOAD_EXTREMES + copy * 2u + 1u] =
                record[last] + record[last + 1u] + record[last + 2u];
        }
    }

    let palette = palette_evidence(colors, copies);
    payload[PAYLOAD_PALETTE_SEPARATION] = bitcast<u32>(palette.x);
    payload[PAYLOAD_PALETTE_DISAGREEMENT] = bitcast<u32>(palette.y);

    var admission = ADMISSION_REJECTED;
    if status == STATUS_OK && record[RECORD_PART1_SYNDROME] == 0u &&
        record[RECORD_PART2_SYNDROME] == 0u {
        admission = ADMISSION_OPEN;
    } else if (status == STATUS_OK || status == STATUS_DEFAULT) &&
        palette.x >= 8.0 && palette.y <= 1.7 * palette.x {
        admission = ADMISSION_PENDING;
        payload[PAYLOAD_EVIDENCE_FLAGS] = EVIDENCE_FIXED_PATTERN;
    }
    if status == STATUS_SIZE_MISMATCH || status == STATUS_ECC_ORDER {
        admission = ADMISSION_REJECTED;
    }
    payload[PAYLOAD_ADMISSION] = admission;
    ldpc[LDPC_ADMISSION] = admission;
}
