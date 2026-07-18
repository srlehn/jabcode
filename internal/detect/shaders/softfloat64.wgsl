// Integer softfloat64 routines shared by every kernel module that must
// reproduce CPU float64 arithmetic bit for bit (round to nearest even,
// jammed sticky bits, following runtime/softfloat64.go's structure). The
// module source is assembled by prepending this file to the consumer's
// fragment. The domain is finite normal values and signed zero inputs with
// positive-zero results except where noted; no NaN, infinity or denormal
// handling exists. Every sf_* function has a Go twin in
// gpu_softfloat_test.go; keep the arithmetic in sync.

// A float64 as raw IEEE 754 binary64 bits split into u32 halves.
struct F64 { hi: u32, lo: u32 }

// An unpacked working mantissa of up to 64 bits.
struct Mant { hi: u32, lo: u32 }

struct ShiftSticky { m: Mant, sticky: u32 }

struct DivSmall { q: Mant, rem: u32 }

struct Unpacked { mant: Mant, exp: i32, sign: u32, zero: bool }

fn mant_zero(m: Mant) -> bool { return m.hi == 0u && m.lo == 0u; }

fn mant_less(a: Mant, b: Mant) -> bool {
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}

fn mant_shl(m: Mant, k: u32) -> Mant {
    if k == 0u { return m; }
    return Mant((m.hi << k) | (m.lo >> (32u - k)), m.lo << k);
}

fn mant_shr(m: Mant, k: u32) -> Mant {
    if k == 0u { return m; }
    return Mant(m.hi >> k, (m.lo >> k) | (m.hi << (32u - k)));
}

fn mant_shr_sticky(m0: Mant, k0: u32) -> ShiftSticky {
    var m = m0;
    var k = k0;
    if k == 0u { return ShiftSticky(m, 0u); }
    if k >= 64u {
        if mant_zero(m) { return ShiftSticky(Mant(0u, 0u), 0u); }
        return ShiftSticky(Mant(0u, 0u), 1u);
    }
    var sticky = 0u;
    if k >= 32u {
        if m.lo != 0u { sticky = 1u; }
        m = Mant(0u, m.hi);
        k = k - 32u;
        if k == 0u { return ShiftSticky(m, sticky); }
    }
    if (m.lo << (32u - k)) != 0u { sticky = 1u; }
    return ShiftSticky(mant_shr(m, k), sticky);
}

fn mant_add(a: Mant, b: Mant) -> Mant {
    let lo = a.lo + b.lo;
    var carry = 0u;
    if lo < a.lo { carry = 1u; }
    return Mant(a.hi + b.hi + carry, lo);
}

// mant_sub computes a - b for a >= b.
fn mant_sub(a: Mant, b: Mant) -> Mant {
    let lo = a.lo - b.lo;
    var borrow = 0u;
    if a.lo < b.lo { borrow = 1u; }
    return Mant(a.hi - b.hi - borrow, lo);
}

// mant_div_small divides a mantissa of up to 57 bits by a small constant
// divisor via base-2^16 long division.
fn mant_div_small(m: Mant, d: u32) -> DivSmall {
    var rem = 0u;
    var digits = array<u32, 4>(m.hi >> 16u, m.hi & 0xffffu, m.lo >> 16u, m.lo & 0xffffu);
    var q = array<u32, 4>(0u, 0u, 0u, 0u);
    for (var i = 0; i < 4; i++) {
        let cur = (rem << 16u) | digits[i];
        q[i] = cur / d;
        rem = cur % d;
    }
    return DivSmall(Mant((q[0] << 16u) | q[1], (q[2] << 16u) | q[3]), rem);
}

// sf_unpack splits raw bits into sign, 53-bit mantissa with the implicit top
// bit, and the exponent in the convention value = mant * 2^(exp - 52).
fn sf_unpack(a: F64) -> Unpacked {
    let sign = a.hi & 0x80000000u;
    let e = i32((a.hi >> 20u) & 0x7ffu);
    if e == 0 { return Unpacked(Mant(0u, 0u), 0, sign, true); }
    return Unpacked(Mant((a.hi & 0xfffffu) | 0x100000u, a.lo), e - 1023, sign, false);
}

// sf_pack normalizes and rounds a working mantissa to nearest-even. trunc
// carries sticky bits already shifted out of mant.
fn sf_pack(sign: u32, mant0: Mant, exp0: i32, trunc0: u32) -> F64 {
    var mant = mant0;
    var exp = exp0;
    var trunc = trunc0;
    if mant_zero(mant) { return F64(sign, 0u); }
    loop {
        if mant.hi >= 0x100000u { break; } // reached 2^52
        mant = mant_shl(mant, 1u);
        exp = exp - 1;
    }
    loop {
        if mant.hi < 0x400000u { break; } // below 2^54
        trunc = trunc | (mant.lo & 1u);
        mant = mant_shr(mant, 1u);
        exp = exp + 1;
    }
    if mant.hi >= 0x200000u { // at least 2^53: one round bit left
        if (mant.lo & 1u) != 0u && (trunc != 0u || (mant.lo & 2u) != 0u) {
            mant = mant_add(mant, Mant(0u, 1u));
            if mant.hi >= 0x400000u {
                mant = mant_shr(mant, 1u);
                exp = exp + 1;
            }
        }
        mant = mant_shr(mant, 1u);
        exp = exp + 1;
    }
    return F64(sign | (u32(exp + 1023) << 20u) | (mant.hi & 0xfffffu), mant.lo);
}

fn sf_add(a: F64, b: F64) -> F64 {
    var f = sf_unpack(a);
    var g = sf_unpack(b);
    if f.zero && g.zero { return F64(f.sign & g.sign, 0u); }
    if f.zero { return b; }
    if g.zero { return a; }
    if f.exp < g.exp || (f.exp == g.exp && mant_less(f.mant, g.mant)) {
        let t = f;
        f = g;
        g = t;
    }
    let shift = u32(f.exp - g.exp);
    var fm = mant_shl(f.mant, 2u);
    let gs = mant_shr_sticky(mant_shl(g.mant, 2u), shift);
    var fs = f.sign;
    if f.sign == g.sign {
        fm = mant_add(fm, gs.m);
    } else {
        fm = mant_sub(fm, gs.m);
        if gs.sticky != 0u { fm = mant_sub(fm, Mant(0u, 1u)); }
    }
    if mant_zero(fm) { fs = 0u; }
    return sf_pack(fs, fm, f.exp - 2, gs.sticky);
}

fn sf_neg(a: F64) -> F64 { return F64(a.hi ^ 0x80000000u, a.lo); }

fn sf_abs(a: F64) -> F64 { return F64(a.hi & 0x7fffffffu, a.lo); }

fn sf_sub(a: F64, b: F64) -> F64 { return sf_add(a, sf_neg(b)); }

fn sf_scale_pow2(a: F64, k: i32) -> F64 {
    if (a.hi & 0x7ff00000u) == 0u { return a; }
    return F64(a.hi + (u32(k) << 20u), a.lo);
}

fn sf_div_small(a: F64, d: u32) -> F64 {
    let u = sf_unpack(a);
    if u.zero { return F64(u.sign, 0u); }
    let dv = mant_div_small(mant_shl(u.mant, 4u), d);
    var trunc = 0u;
    if dv.rem != 0u { trunc = 1u; }
    return sf_pack(u.sign, dv.q, u.exp - 4, trunc);
}

// sf_mul_u16 multiplies a non-negative integer below 2^16 by a positive
// constant via 16-bit limb products, exactly rounded.
fn sf_mul_u16(m: u32, c: F64) -> F64 {
    if m == 0u { return F64(0u, 0u); }
    let u = sf_unpack(c);
    let t0 = (u.mant.lo & 0xffffu) * m;
    let t1 = (u.mant.lo >> 16u) * m + (t0 >> 16u);
    let t2 = (u.mant.hi & 0xffffu) * m + (t1 >> 16u);
    let t3 = (u.mant.hi >> 16u) * m + (t2 >> 16u);
    let p0 = (t0 & 0xffffu) | (t1 << 16u);
    let p1 = (t2 & 0xffffu) | ((t3 & 0xffffu) << 16u);
    let p2 = t3 >> 16u;
    if p2 == 0u {
        return sf_pack(0u, Mant(p1, p0), u.exp, 0u);
    }
    // The product exceeds 64 bits; shift right by the minimal amount that
    // fits the pair, jamming shifted-out bits into the sticky flag.
    let s = firstLeadingBit(p2) + 1u;
    var trunc = 0u;
    if (p0 & ((1u << s) - 1u)) != 0u { trunc = 1u; }
    return sf_pack(
        0u,
        Mant((p2 << (32u - s)) | (p1 >> s), (p1 << (32u - s)) | (p0 >> s)),
        u.exp + i32(s),
        trunc,
    );
}

// mul32_wide computes the full 64-bit product of two u32 values via 16-bit
// limb products.
fn mul32_wide(x: u32, y: u32) -> Mant {
    let xl = x & 0xffffu;
    let xh = x >> 16u;
    let yl = y & 0xffffu;
    let yh = y >> 16u;
    let ll = xl * yl;
    let lh = xl * yh;
    let hl = xh * yl;
    let hh = xh * yh;
    let mid = lh + hl;
    var hi = hh + (mid >> 16u);
    if mid < lh { hi = hi + 0x10000u; }
    let lo = ll + (mid << 16u);
    if lo < ll { hi = hi + 1u; }
    return Mant(hi, lo);
}

// sf_mul is IEEE float64 multiplication over finite normals and zeros; a
// zero operand returns the sign-correct zero. The up-to-106-bit mantissa
// product is assembled in four u32 words, then reduced to a 56-58 bit
// working mantissa with the low 48 bits jammed into the sticky flag.
fn sf_mul(a: F64, b: F64) -> F64 {
    let f = sf_unpack(a);
    let g = sf_unpack(b);
    let sign = f.sign ^ g.sign;
    if f.zero || g.zero { return F64(sign, 0u); }
    let low = mul32_wide(f.mant.lo, g.mant.lo);
    // The two cross products are below 2^53 each (the mantissa high words
    // carry at most 21 bits), so their sum fits a Mant without overflow.
    let cross = mant_add(
        mul32_wide(f.mant.lo, g.mant.hi),
        mul32_wide(f.mant.hi, g.mant.lo),
    );
    let high = mul32_wide(f.mant.hi, g.mant.hi);
    let w0 = low.lo;
    let w1 = low.hi + cross.lo;
    var c1 = 0u;
    if w1 < low.hi { c1 = 1u; }
    let t2 = cross.hi + c1;
    let w2 = t2 + high.lo;
    var c2 = 0u;
    if w2 < t2 { c2 = 1u; }
    let w3 = high.hi + c2;
    var trunc = 0u;
    if w0 != 0u || (w1 & 0xffffu) != 0u { trunc = 1u; }
    let mant = Mant((w2 >> 16u) | (w3 << 16u), (w1 >> 16u) | (w2 << 16u));
    return sf_pack(sign, mant, f.exp + g.exp - 4, trunc);
}

fn sf_from_u32(v: u32) -> F64 {
    if v == 0u { return F64(0u, 0u); }
    return sf_pack(0u, Mant(0u, v), 52, 0u);
}

fn sf_from_i32(v: i32) -> F64 {
    if v < 0 { return sf_neg(sf_from_u32(u32(-v))); }
    return sf_from_u32(u32(v));
}

// Ordered-key comparisons (sign-magnitude flip); the chain produces no
// negative zeros, so the +0/-0 equality corner never arises.
fn sf_key_hi(a: F64) -> u32 {
    if (a.hi & 0x80000000u) != 0u { return ~a.hi; }
    return a.hi | 0x80000000u;
}

fn sf_key_lo(a: F64) -> u32 {
    if (a.hi & 0x80000000u) != 0u { return ~a.lo; }
    return a.lo;
}

fn sf_less(a: F64, b: F64) -> bool {
    let ah = sf_key_hi(a);
    let bh = sf_key_hi(b);
    return ah < bh || (ah == bh && sf_key_lo(a) < sf_key_lo(b));
}

fn sf_less_eq(a: F64, b: F64) -> bool {
    let ah = sf_key_hi(a);
    let bh = sf_key_hi(b);
    return ah < bh || (ah == bh && sf_key_lo(a) <= sf_key_lo(b));
}

// sf_trunc_i32 is Go's int(x) truncation toward zero for |x| < 2^31.
fn sf_trunc_i32(a: F64) -> i32 {
    let u = sf_unpack(a);
    if u.zero || u.exp < 0 { return 0; }
    let sh = u32(52 - u.exp);
    var v: i32;
    if sh >= 32u {
        v = i32(u.mant.hi >> (sh - 32u));
    } else {
        v = i32(mant_shr(u.mant, sh).lo);
    }
    if u.sign != 0u { return -v; }
    return v;
}
