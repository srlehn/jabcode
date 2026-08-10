// Packs the host-visible primary result after metadata interpretation and
// payload correction. The correction workspace stores one u32 per bit because
// that is the useful compute shape; the transfer shape is a compact bitset plus
// only the metadata fields and palette bytes the host consumes.

const WORKGROUP: u32 = 256u;
const RESULT_MAGIC: u32 = 0x4A414252u;
const RESULT_VERSION: u32 = 1u;

const RESULT_HEADER_WORDS: u32 = 16u;
const RESULT_PALETTE_WORDS: u32 = 384u;
const RESULT_PAYLOAD: u32 = RESULT_HEADER_WORDS + RESULT_PALETTE_WORDS;
const MAX_PALETTE_BYTES: u32 = 1536u;
const MAX_PAYLOAD_BITS: u32 = 168200u;

const RESULT_META_STATUS: u32 = 2u;
const RESULT_PAYLOAD_STATUS: u32 = 3u;
const RESULT_META_MODULES: u32 = 4u;
const RESULT_NC: u32 = 5u;
const RESULT_COLORS: u32 = 6u;
const RESULT_VERSION_X: u32 = 7u;
const RESULT_VERSION_Y: u32 = 8u;
const RESULT_ECL_X: u32 = 9u;
const RESULT_ECL_Y: u32 = 10u;
const RESULT_MASK: u32 = 11u;
const RESULT_SYNDROME_1: u32 = 12u;
const RESULT_SYNDROME_2: u32 = 13u;
const RESULT_PALETTE_BYTES: u32 = 14u;
const RESULT_NET_BITS: u32 = 15u;

const PAYLOAD_OK: u32 = 0u;
const PAYLOAD_FAILED: u32 = 1u;
const PAYLOAD_REJECTED: u32 = 2u;
const PAYLOAD_INVALID: u32 = 3u;

const RECORD_STATUS: u32 = 0u;
const RECORD_MODULES: u32 = 1u;
const RECORD_NC: u32 = 4u;
const RECORD_COLORS: u32 = 5u;
const RECORD_VERSION_X: u32 = 6u;
const RECORD_VERSION_Y: u32 = 7u;
const RECORD_ECL_X: u32 = 8u;
const RECORD_ECL_Y: u32 = 9u;
const RECORD_MASK: u32 = 10u;
const RECORD_SYNDROME_1: u32 = 12u;
const RECORD_SYNDROME_2: u32 = 13u;
const RECORD_PALETTE: u32 = 16u;

const PAYLOAD_ADMISSION: u32 = 1714u;
const PAYLOAD_NET_BITS: u32 = 1716u;

const LDPC_BLOCKS: u32 = 4u;

@group(0) @binding(0) var<storage, read> metadata: array<u32>;
@group(0) @binding(1) var<storage, read> payload: array<u32>;
@group(0) @binding(2) var<storage, read> ldpc: array<u32>;
@group(0) @binding(3) var<storage, read> net: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> result: array<u32>;

var<workgroup> failed: atomic<u32>;

fn palette_bytes(colors: u32) -> u32 {
    if colors < 4u || colors > 256u || (colors & (colors - 1u)) != 0u {
        return 0u;
    }
    return colors * select(2u, 4u, colors <= 8u) * 3u;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
    let lane = local.x;
    let colors = metadata[RECORD_COLORS];
    let palette_len = palette_bytes(colors);
    let net_bits = payload[PAYLOAD_NET_BITS];
    let blocks = ldpc[LDPC_BLOCKS];
    let payload_valid = net_bits <= MAX_PAYLOAD_BITS && blocks <= 64u;

    if lane == 0u {
        atomicStore(&failed, 0u);
    }
    workgroupBarrier();
    if lane < blocks && atomicLoad(&net[lane]) != 0u {
        atomicMax(&failed, 1u);
    }

    for (var word = lane; word < (palette_len + 3u) / 4u; word += WORKGROUP) {
        var packed = 0u;
        for (var octet = 0u; octet < 4u; octet += 1u) {
            let at = word * 4u + octet;
            if at < palette_len {
                packed |= (metadata[RECORD_PALETTE + at] & 0xffu) << (octet * 8u);
            }
        }
        result[RESULT_HEADER_WORDS + word] = packed;
    }
    if payload_valid {
        for (var word = lane; word < (net_bits + 31u) / 32u; word += WORKGROUP) {
            var packed = 0u;
            for (var bit = 0u; bit < 32u; bit += 1u) {
                let at = word * 32u + bit;
                if at < net_bits {
                    packed |= (atomicLoad(&net[blocks + at]) & 1u) << bit;
                }
            }
            result[RESULT_PAYLOAD + word] = packed;
        }
    }
    workgroupBarrier();

    if lane != 0u {
        return;
    }
    result[0] = RESULT_MAGIC;
    result[1] = RESULT_VERSION;
    result[RESULT_META_STATUS] = metadata[RECORD_STATUS];
    result[RESULT_META_MODULES] = metadata[RECORD_MODULES];
    result[RESULT_NC] = metadata[RECORD_NC];
    result[RESULT_COLORS] = colors;
    result[RESULT_VERSION_X] = metadata[RECORD_VERSION_X];
    result[RESULT_VERSION_Y] = metadata[RECORD_VERSION_Y];
    result[RESULT_ECL_X] = metadata[RECORD_ECL_X];
    result[RESULT_ECL_Y] = metadata[RECORD_ECL_Y];
    result[RESULT_MASK] = metadata[RECORD_MASK];
    result[RESULT_SYNDROME_1] = metadata[RECORD_SYNDROME_1];
    result[RESULT_SYNDROME_2] = metadata[RECORD_SYNDROME_2];
    result[RESULT_PALETTE_BYTES] = palette_len;
    result[RESULT_NET_BITS] = select(0u, net_bits, payload_valid);

    var status = PAYLOAD_OK;
    if !payload_valid {
        status = PAYLOAD_INVALID;
    } else if payload[PAYLOAD_ADMISSION] != 0u {
        status = PAYLOAD_REJECTED;
    } else if atomicLoad(&failed) != 0u {
        status = PAYLOAD_FAILED;
    }
    result[RESULT_PAYLOAD_STATUS] = status;
}
