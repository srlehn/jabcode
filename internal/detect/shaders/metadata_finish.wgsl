// Turns Part II's corrected bits into the symbol's declared shape: the two side
// versions, the two ECC weights and the mask reference.
//
// It exists as its own dispatch because the fields are only readable after the
// correction, and the correction is a dispatch of its own. Reading them on the
// host instead would be a second download, and it would have to happen before
// the payload correction rather than with the result, so no later fusion could
// absorb it.

const STATUS_OK: u32 = 0u;
// The declared side does not match the grid the sampler produced, or the ECC
// weights are not ordered. The host treats both as metadata failures with
// ladders of their own rather than as a lost read.
const STATUS_SIZE_MISMATCH: u32 = 4u;
const STATUS_ECC_ORDER: u32 = 5u;

const VERSION_BITS: u32 = 10u;
const ECC_BITS: u32 = 6u;

const PARAM_SIDE_X: u32 = 0u;
const PARAM_SIDE_Y: u32 = 1u;

const RECORD_STATUS: u32 = 0u;
const RECORD_SIDE_VERSION_X: u32 = 6u;
const RECORD_SIDE_VERSION_Y: u32 = 7u;
const RECORD_ECL_X: u32 = 8u;
const RECORD_ECL_Y: u32 = 9u;
const RECORD_MASK: u32 = 10u;
const RECORD_PART2_SYNDROME: u32 = 13u;

@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> net: array<u32>;
@group(0) @binding(2) var<storage, read_write> record: array<u32>;

// message_bit reads one corrected bit. The corrector writes a status word per
// block ahead of the message, and Part II is a single block.
fn message_bit(index: u32) -> u32 {
    return net[1u + index] & 1u;
}

fn message_field(offset: u32, width: u32) -> u32 {
    var value = 0u;
    for (var i = 0u; i < width; i += 1u) {
        value += message_bit(offset + i) << (width - 1u - i);
    }
    return value;
}

@compute @workgroup_size(1)
fn main() {
    if record[RECORD_STATUS] != STATUS_OK {
        return;
    }
    record[RECORD_PART2_SYNDROME] = net[0];

    let version_x = message_field(0u, VERSION_BITS / 2u) + 1u;
    let version_y = message_field(VERSION_BITS / 2u, VERSION_BITS / 2u) + 1u;
    let ecl_x = message_field(VERSION_BITS, ECC_BITS / 2u) + 3u;
    let ecl_y = message_field(VERSION_BITS + ECC_BITS / 2u, ECC_BITS / 2u) + 4u;
    let mask = message_field(VERSION_BITS + ECC_BITS, 3u);

    record[RECORD_SIDE_VERSION_X] = version_x;
    record[RECORD_SIDE_VERSION_Y] = version_y;
    record[RECORD_ECL_X] = ecl_x;
    record[RECORD_ECL_Y] = ecl_y;
    record[RECORD_MASK] = mask;

    if version_x * 4u + 17u != params[PARAM_SIDE_X] ||
        version_y * 4u + 17u != params[PARAM_SIDE_Y] {
        record[RECORD_STATUS] = STATUS_SIZE_MISMATCH;
        return;
    }
    if ecl_x >= ecl_y {
        record[RECORD_STATUS] = STATUS_ECC_ORDER;
    }
}
