// Packs the host-visible primary result after metadata interpretation and
// payload correction. The correction workspace stores one u32 per bit because
// that is the useful compute shape; the transfer shape is a compact bitset plus
// only the metadata fields and palette bytes the host consumes.

const WORKGROUP: u32 = 256u;
const RESULT_MAGIC: u32 = 0x4A414252u;
const RESULT_VERSION: u32 = 3u;

const RESULT_HEADER_WORDS: u32 = 64u;
const RESULT_PALETTE_WORDS: u32 = 384u;
const RESULT_PAYLOAD: u32 = RESULT_HEADER_WORDS + RESULT_PALETTE_WORDS;
const RESULT_WORDS: u32 = 5705u;
const RESULT_BATCH_SLOTS: u32 = 36u;
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
const RESULT_META_PART1_INITIAL: u32 = 16u;
const RESULT_META_PART1_CORRECTIONS: u32 = 17u;
const RESULT_META_PART2_INITIAL: u32 = 18u;
const RESULT_META_PART2_CORRECTIONS: u32 = 19u;
const RESULT_PALETTE_SEPARATION: u32 = 20u;
const RESULT_PALETTE_DISAGREEMENT: u32 = 21u;
const RESULT_FIXED_AGREEMENTS: u32 = 22u;
const RESULT_FIXED_CHECKS: u32 = 23u;
const RESULT_EVIDENCE_FLAGS: u32 = 24u;
const RESULT_PAYLOAD_INITIAL: u32 = 25u;
const RESULT_PAYLOAD_HARD_RESIDUAL: u32 = 26u;
const RESULT_PAYLOAD_CORRECTIONS: u32 = 27u;
const RESULT_PAYLOAD_SOFT_USED: u32 = 28u;
const RESULT_PAYLOAD_SOFT_RESIDUAL: u32 = 29u;
const RESULT_PAYLOAD_SOFT_ITERATIONS: u32 = 30u;
const RESULT_SIDE_X: u32 = 31u;
const RESULT_SIDE_Y: u32 = 32u;
const RESULT_PROFILE: u32 = 33u;
const RESULT_SLOT: u32 = 34u;
const RESULT_GEOMETRY: u32 = 35u;
const RESULT_CORNER: u32 = 36u;
const RESULT_DEGREES: u32 = 37u;
const RESULT_PRINT: u32 = 38u;
const RESULT_PATTERNS: u32 = 40u;

const SAMPLE_DEST_WIDTH: u32 = 25u;
const SAMPLE_DEST_HEIGHT: u32 = 26u;
const SAMPLE_METADATA_CONTROL: u32 = 27u;
const METADATA_PROFILE_MASK: u32 = 0xffu;
const METADATA_SLOT_SHIFT: u32 = 8u;
const METADATA_BATCH_FLAG: u32 = 0x10000u;

const CONTROL_GEOMETRY: u32 = 0u;
const CONTROL_CORNER: u32 = 1u;
const CONTROL_DEGREES: u32 = 2u;
const CONTROL_VALID: u32 = 3u;
const CONTROL_PATTERNS: u32 = 4u;
const PATTERN_WORDS: u32 = 24u;
const CONTROL_PRINT: u32 = 28u;

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
const RECORD_PART1_INITIAL: u32 = 14u;
const RECORD_PART1_CORRECTIONS: u32 = 15u;
const RECORD_PALETTE: u32 = 16u;
const RECORD_PART2_INITIAL: u32 = 1692u;
const RECORD_PART2_CORRECTIONS: u32 = 1693u;

const PAYLOAD_ADMISSION: u32 = 1714u;
const PAYLOAD_NET_BITS: u32 = 1716u;
const PAYLOAD_PALETTE_SEPARATION: u32 = 1719u;
const PAYLOAD_PALETTE_DISAGREEMENT: u32 = 1720u;
const PAYLOAD_FIXED_AGREEMENTS: u32 = 1721u;
const PAYLOAD_FIXED_CHECKS: u32 = 1722u;
const PAYLOAD_EVIDENCE_FLAGS: u32 = 1723u;

const EVIDENCE_BASE: u32 = 180288u;
const EVIDENCE_INITIAL: u32 = EVIDENCE_BASE;
const EVIDENCE_HARD_RESIDUAL: u32 = EVIDENCE_BASE + 1u;
const EVIDENCE_CORRECTIONS: u32 = EVIDENCE_BASE + 2u;
const EVIDENCE_SOFT_USED: u32 = EVIDENCE_BASE + 3u;
const EVIDENCE_SOFT_RESIDUAL: u32 = EVIDENCE_BASE + 4u;
const EVIDENCE_SOFT_ITERATIONS: u32 = EVIDENCE_BASE + 5u;

const LDPC_BLOCKS: u32 = 4u;

@group(0) @binding(0) var<storage, read> metadata: array<u32>;
@group(0) @binding(1) var<storage, read> payload: array<u32>;
@group(0) @binding(2) var<storage, read> ldpc: array<u32>;
@group(0) @binding(3) var<storage, read> net: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> result: array<u32>;
@group(0) @binding(5) var<storage, read> sample: array<u32>;
@group(0) @binding(6) var<storage, read> control: array<u32>;

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
	let metadata_control = sample[SAMPLE_METADATA_CONTROL];
	let slot = min(
		(metadata_control >> METADATA_SLOT_SHIFT) & 0xffu,
		RESULT_BATCH_SLOTS - 1u,
	);
	let base = slot * RESULT_WORDS;
	let batch = (metadata_control & METADATA_BATCH_FLAG) != 0u;
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
		result[base + RESULT_HEADER_WORDS + word] = packed;
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
			result[base + RESULT_PAYLOAD + word] = packed;
        }
    }
    workgroupBarrier();

    if lane != 0u {
        return;
    }
	result[base + 0u] = RESULT_MAGIC;
	result[base + 1u] = RESULT_VERSION;
	result[base + RESULT_META_STATUS] = metadata[RECORD_STATUS];
	result[base + RESULT_META_MODULES] = metadata[RECORD_MODULES];
	result[base + RESULT_NC] = metadata[RECORD_NC];
	result[base + RESULT_COLORS] = colors;
	result[base + RESULT_VERSION_X] = metadata[RECORD_VERSION_X];
	result[base + RESULT_VERSION_Y] = metadata[RECORD_VERSION_Y];
	result[base + RESULT_ECL_X] = metadata[RECORD_ECL_X];
	result[base + RESULT_ECL_Y] = metadata[RECORD_ECL_Y];
	result[base + RESULT_MASK] = metadata[RECORD_MASK];
	result[base + RESULT_SYNDROME_1] = metadata[RECORD_SYNDROME_1];
	result[base + RESULT_SYNDROME_2] = metadata[RECORD_SYNDROME_2];
	result[base + RESULT_PALETTE_BYTES] = palette_len;
	result[base + RESULT_NET_BITS] = select(0u, net_bits, payload_valid);
	result[base + RESULT_META_PART1_INITIAL] = metadata[RECORD_PART1_INITIAL];
	result[base + RESULT_META_PART1_CORRECTIONS] = metadata[RECORD_PART1_CORRECTIONS];
	result[base + RESULT_META_PART2_INITIAL] = metadata[RECORD_PART2_INITIAL];
	result[base + RESULT_META_PART2_CORRECTIONS] = metadata[RECORD_PART2_CORRECTIONS];
	result[base + RESULT_PALETTE_SEPARATION] = payload[PAYLOAD_PALETTE_SEPARATION];
	result[base + RESULT_PALETTE_DISAGREEMENT] = payload[PAYLOAD_PALETTE_DISAGREEMENT];
	result[base + RESULT_FIXED_AGREEMENTS] = payload[PAYLOAD_FIXED_AGREEMENTS];
	result[base + RESULT_FIXED_CHECKS] = payload[PAYLOAD_FIXED_CHECKS];
	result[base + RESULT_EVIDENCE_FLAGS] = payload[PAYLOAD_EVIDENCE_FLAGS];
	result[base + RESULT_PAYLOAD_INITIAL] = atomicLoad(&net[EVIDENCE_INITIAL]);
	result[base + RESULT_PAYLOAD_HARD_RESIDUAL] = atomicLoad(&net[EVIDENCE_HARD_RESIDUAL]);
	result[base + RESULT_PAYLOAD_CORRECTIONS] = atomicLoad(&net[EVIDENCE_CORRECTIONS]);
	result[base + RESULT_PAYLOAD_SOFT_USED] = atomicLoad(&net[EVIDENCE_SOFT_USED]);
	result[base + RESULT_PAYLOAD_SOFT_RESIDUAL] = atomicLoad(&net[EVIDENCE_SOFT_RESIDUAL]);
	result[base + RESULT_PAYLOAD_SOFT_ITERATIONS] = atomicLoad(&net[EVIDENCE_SOFT_ITERATIONS]);
	let batch_valid = batch && control[CONTROL_VALID] != 0u;
	result[base + RESULT_SIDE_X] = select(0u, sample[SAMPLE_DEST_WIDTH], batch_valid);
	result[base + RESULT_SIDE_Y] = select(0u, sample[SAMPLE_DEST_HEIGHT], batch_valid);
	result[base + RESULT_PROFILE] = metadata_control & METADATA_PROFILE_MASK;
	result[base + RESULT_SLOT] = slot;
	result[base + RESULT_GEOMETRY] = select(0u, control[CONTROL_GEOMETRY], batch_valid);
	result[base + RESULT_CORNER] = select(0u, control[CONTROL_CORNER], batch_valid);
	result[base + RESULT_DEGREES] = select(0u, control[CONTROL_DEGREES], batch_valid);
	result[base + RESULT_PRINT] = select(0u, control[CONTROL_PRINT], batch_valid);
	if batch_valid {
		for (var word = 0u; word < PATTERN_WORDS; word += 1u) {
			result[base + RESULT_PATTERNS + word] = control[CONTROL_PATTERNS + word];
		}
	}

    var status = PAYLOAD_OK;
    if !payload_valid {
        status = PAYLOAD_INVALID;
    } else if payload[PAYLOAD_ADMISSION] != 0u {
        status = PAYLOAD_REJECTED;
    } else if atomicLoad(&failed) != 0u {
        status = PAYLOAD_FAILED;
    }
	result[base + RESULT_PAYLOAD_STATUS] = status;
}
