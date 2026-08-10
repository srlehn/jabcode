// Builds the byte-deinterleaving permutation for a codeword of a given length.
//
// The permutation is a back-to-front Fisher-Yates shuffle driven by the format's
// seeded generator, so it is serial: each swap decides which element the next
// one may still move. It is also independent of the image - only the codeword
// length and the wire variant select it - which is what makes it worth building
// here at all. One lane runs the shuffle once and the table stays resident, so
// every later correction of the same length reuses it instead of moving the
// permuted bits across the bus.
//
// Both of the format's generators are reproduced. The ISO one is a 32-bit
// congruential sequence; the C-family one is a 64-bit congruential sequence
// whose high word is tempered, and its state is carried as two 32-bit halves
// because the device has no 64-bit integer.

const WORKGROUP: u32 = 256u;
const INTERLEAVE_SEED: u32 = 226759u;

// The C-family multiplier, 0x5851F42D4C957F2D, split into halves.
const LCG_MULTIPLIER_HIGH: u32 = 0x5851F42Du;
const LCG_MULTIPLIER_LOW: u32 = 0x4C957F2Du;

const GENERATOR_LCG: u32 = 1u;

const PARAM_GROSS_BITS: u32 = 8u;
const PARAM_GENERATOR: u32 = 9u;
const PARAM_ADMISSION: u32 = 1714u;

@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read_write> permutation: array<u32>;

// wide_product is a 32 by 32 bit multiply kept whole, as (high, low).
fn wide_product(a: u32, b: u32) -> vec2<u32> {
    let a_low = a & 0xffffu;
    let a_high = a >> 16u;
    let b_low = b & 0xffffu;
    let b_high = b >> 16u;
    let low_low = a_low * b_low;
    let cross_a = a_high * b_low;
    let cross_b = a_low * b_high;
    let middle = (low_low >> 16u) + (cross_a & 0xffffu) + (cross_b & 0xffffu);
    return vec2<u32>(
        a_high * b_high + (cross_a >> 16u) + (cross_b >> 16u) + (middle >> 16u),
        (low_low & 0xffffu) | (middle << 16u),
    );
}

// lcg_step advances the 64-bit state by one multiply-add, keeping only the low
// 64 bits exactly as the reference generator's unsigned arithmetic does.
fn lcg_step(state: vec2<u32>) -> vec2<u32> {
    var product = wide_product(state.y, LCG_MULTIPLIER_LOW);
    product.x += state.y * LCG_MULTIPLIER_HIGH + state.x * LCG_MULTIPLIER_LOW;
    let low = product.y + 1u;
    if low < product.y {
        product.x += 1u;
    }
    return vec2<u32>(product.x, low);
}

// temper is the MT-style output transform the C-family generator applies to the
// state's high word.
fn temper(value: u32) -> u32 {
    var x = value;
    x ^= x >> 11u;
    x ^= (x << 7u) & 0x9D2C5680u;
    x ^= (x << 15u) & 0xEFC60000u;
    x ^= x >> 18u;
    return x;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
    if params[PARAM_ADMISSION] != 0u {
        return;
    }
    let length = params[PARAM_GROSS_BITS];
    for (var at = local.x; at < length; at += WORKGROUP) {
        permutation[at] = at;
    }
    workgroupBarrier();
    if local.x != 0u {
        return;
    }
    let lcg = params[PARAM_GENERATOR] == GENERATOR_LCG;
    var iso_state = INTERLEAVE_SEED;
    var lcg_state = vec2<u32>(0u, INTERLEAVE_SEED);
    for (var taken = 0u; taken < length; taken += 1u) {
        let remaining = f32(length - taken);
        var pos = 0u;
        if lcg {
            lcg_state = lcg_step(lcg_state);
            pos = u32(f32(temper(lcg_state.x)) / 4294967296.0 * remaining);
        } else {
            iso_state = iso_state * 1103515245u + 12345u;
            pos = u32(f32((iso_state / 65536u) % 32768u) / 32768.0 * remaining);
        }
        let last = length - 1u - taken;
        let held = permutation[last];
        permutation[last] = permutation[pos];
        permutation[pos] = held;
    }
}
