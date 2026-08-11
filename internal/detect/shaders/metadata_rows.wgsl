// Expands the four immutable metadata parity graphs into retained sparse rows.
// Each graph is stored here as one 38-bit mask per parity row; lanes own rows
// and emit their ascending column indexes in parallel. This avoids both a host
// row upload and the payload constructor's much larger elimination workspace.

const ROW_SET_WORDS: u32 = 512u;
const ROW_PAD: u32 = 0xffffffffu;

const ISO_PART1 = array<u32, 6>(
    0x0000002du, 0x00000000u,
    0x0000001bu, 0x00000000u,
    0x00000039u, 0x00000000u,
);
const ISO_PART2 = array<u32, 38>(
    0x53218f87u, 0x00000035u, 0x19b733a9u, 0x00000021u,
    0xcca689fdu, 0x00000010u, 0x39aa17a5u, 0x00000023u,
    0x56e6a69eu, 0x00000010u, 0x89cbcbd0u, 0x00000023u,
    0xb9b46505u, 0x0000002eu, 0xaed551d4u, 0x00000024u,
    0x3f7da442u, 0x00000014u, 0x6d23b1b7u, 0x00000010u,
    0xa70607f7u, 0x00000021u, 0x19990738u, 0x0000003fu,
    0x93c5ec1bu, 0x00000018u, 0xa0f956adu, 0x0000000cu,
    0x8cf017e3u, 0x0000001au, 0xb3d88f00u, 0x0000003eu,
    0x4cbfe5c1u, 0x00000010u, 0x9b138e5cu, 0x00000032u,
    0xce5417d6u, 0x00000021u,
);
const CURRENT_PART1 = array<u32, 6>(
    0x0000002du, 0x00000000u,
    0x0000003au, 0x00000000u,
    0x0000003cu, 0x00000000u,
);
const CURRENT_PART2 = array<u32, 38>(
    0x0d2d31d7u, 0x00000038u, 0x1e87379du, 0x00000010u,
    0xe500f4e4u, 0x0000003du, 0xdfa7cb28u, 0x00000000u,
    0x3a9ce9d4u, 0x00000011u, 0x9f3b80a3u, 0x00000032u,
    0x622a1ce8u, 0x0000003fu, 0x659692c9u, 0x0000003au,
    0x2c3f12d7u, 0x00000012u, 0xcc8a2369u, 0x0000003eu,
    0xf2620d95u, 0x00000027u, 0x65de2636u, 0x0000000au,
    0x5899d751u, 0x00000029u, 0xb1cd84f3u, 0x00000009u,
    0x6bc80ad4u, 0x00000037u, 0x97adb418u, 0x00000038u,
    0xa5f8f44du, 0x00000004u, 0xe0fcad11u, 0x00000015u,
    0xb186a2f1u, 0x00000035u,
);

@group(0) @binding(0) var<storage, read_write> rows: array<u32>;

fn bit(mask: vec2<u32>, column: u32) -> bool {
    let word = select(mask.x, mask.y, column >= 32u);
    return (word & (1u << (column % 32u))) != 0u;
}

fn emit(base: u32, row: u32, degree: u32, mask: vec2<u32>) {
    var at = 0u;
    for (var column = 0u; column < 38u; column += 1u) {
        if bit(mask, column) {
            rows[base + row * degree + at] = column;
            at += 1u;
        }
    }
    for (; at < degree; at += 1u) {
        rows[base + row * degree + at] = ROW_PAD;
    }
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
    let lane = local.x;
    if lane < 3u {
        emit(0u, lane, 4u, vec2<u32>(ISO_PART1[lane * 2u], ISO_PART1[lane * 2u + 1u]));
        return;
    }
    if lane < 22u {
        let row = lane - 3u;
        emit(
            ROW_SET_WORDS,
            row,
            19u,
            vec2<u32>(ISO_PART2[row * 2u], ISO_PART2[row * 2u + 1u]),
        );
        return;
    }
    if lane < 25u {
        let row = lane - 22u;
        emit(
            2u * ROW_SET_WORDS,
            row,
            4u,
            vec2<u32>(CURRENT_PART1[row * 2u], CURRENT_PART1[row * 2u + 1u]),
        );
        return;
    }
    if lane < 44u {
        let row = lane - 25u;
        emit(
            3u * ROW_SET_WORDS,
            row,
            19u,
            vec2<u32>(CURRENT_PART2[row * 2u], CURRENT_PART2[row * 2u + 1u]),
        );
    }
}
