package detect

import (
	"math"
	"math/bits"
	"math/rand"
	"testing"
)

// This file pins the integer softfloat64 algorithms the finder chain
// kernels use for the float64 arithmetic that has no exact rational
// shortcut: the center and module-size averages, the mean/2.5 consistency
// tolerances and the diagonal length constant. Each sf* function mirrors its
// WGSL twin in shaders/finder_chain_prelude.wgsl operation for operation
// using only u32-pair arithmetic. The tests below check the algorithms
// against hardware float64 over the chain's domain (finite normal values
// and positive zero) with deterministic sampled fuzzing plus exhaustive
// sweeps where feasible (the 16-bit constant multiply, the small-integer
// division inputs); the end-to-end backstops are the bit-exact chain
// equivalence and device parity tests. The chain never produces negative
// zero, infinity, NaN or denormals; the packing helper mirrors
// runtime/softfloat64.go's rounding structure (jammed sticky bits, round to
// nearest even).

// sf64 is a float64 as raw IEEE 754 binary64 bits split into u32 halves.
type sf64 struct{ hi, lo uint32 }

// sfMant is an unpacked working mantissa of up to 64 bits.
type sfMant struct{ hi, lo uint32 }

const sfBias = 1023

// Test-side conversions between hardware float64 and the split form. These
// two helpers have no WGSL twin; the kernel only ever sees the split bits.
func sfFromFloat(f float64) sf64 {
	bits := math.Float64bits(f)
	return sf64{uint32(bits >> 32), uint32(bits)}
}

func (a sf64) float() float64 {
	return math.Float64frombits(uint64(a.hi)<<32 | uint64(a.lo))
}

func mantZero(m sfMant) bool { return m.hi == 0 && m.lo == 0 }

func mantLess(a, b sfMant) bool {
	return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo)
}

// mantShl shifts left by k < 32 bits.
func mantShl(m sfMant, k uint32) sfMant {
	if k == 0 {
		return m
	}
	return sfMant{m.hi<<k | m.lo>>(32-k), m.lo << k}
}

// mantShr shifts right by k < 32 bits, dropping shifted-out bits.
func mantShr(m sfMant, k uint32) sfMant {
	if k == 0 {
		return m
	}
	return sfMant{m.hi >> k, m.lo>>k | m.hi<<(32-k)}
}

// mantShrSticky shifts right by any k, returning 1 when nonzero bits were
// shifted out.
func mantShrSticky(m sfMant, k uint32) (sfMant, uint32) {
	if k == 0 {
		return m, 0
	}
	if k >= 64 {
		if mantZero(m) {
			return sfMant{}, 0
		}
		return sfMant{}, 1
	}
	var sticky uint32
	if k >= 32 {
		if m.lo != 0 {
			sticky = 1
		}
		m = sfMant{0, m.hi}
		k -= 32
		if k == 0 {
			return m, sticky
		}
	}
	if m.lo<<(32-k) != 0 {
		sticky = 1
	}
	return mantShr(m, k), sticky
}

func mantAdd(a, b sfMant) sfMant {
	lo := a.lo + b.lo
	carry := uint32(0)
	if lo < a.lo {
		carry = 1
	}
	return sfMant{a.hi + b.hi + carry, lo}
}

// mantSub computes a - b for a >= b.
func mantSub(a, b sfMant) sfMant {
	lo := a.lo - b.lo
	borrow := uint32(0)
	if a.lo < b.lo {
		borrow = 1
	}
	return sfMant{a.hi - b.hi - borrow, lo}
}

// mantDivSmall divides a mantissa of up to 57 bits by a small constant
// divisor via base-2^16 long division, returning quotient and remainder.
func mantDivSmall(m sfMant, d uint32) (sfMant, uint32) {
	rem := uint32(0)
	var q [4]uint32
	digits := [4]uint32{m.hi >> 16, m.hi & 0xffff, m.lo >> 16, m.lo & 0xffff}
	for i := range digits {
		cur := rem<<16 | digits[i]
		q[i] = cur / d
		rem = cur % d
	}
	return sfMant{q[0]<<16 | q[1], q[2]<<16 | q[3]}, rem
}

// sfUnpack splits raw bits into sign, 53-bit mantissa with the implicit top
// bit, and the exponent in the convention value = mant * 2^(exp - 52).
// A zero exponent means zero in the chain's domain (no denormals).
func sfUnpack(a sf64) (sign uint32, mant sfMant, exp int32, zero bool) {
	sign = a.hi & 0x8000_0000
	exp = int32((a.hi >> 20) & 0x7ff)
	if exp == 0 {
		return sign, sfMant{}, 0, true
	}
	return sign, sfMant{a.hi&0xf_ffff | 0x10_0000, a.lo}, exp - sfBias, false
}

// sfPack normalizes and rounds a working mantissa to nearest-even and packs
// it. trunc carries sticky bits already shifted out of mant.
func sfPack(sign uint32, mant sfMant, exp int32, trunc uint32) sf64 {
	if mantZero(mant) {
		return sf64{sign, 0}
	}
	for mant.hi < 0x10_0000 { // below 2^52
		mant = mantShl(mant, 1)
		exp--
	}
	for mant.hi >= 0x40_0000 { // at least 2^54
		trunc |= mant.lo & 1
		mant = mantShr(mant, 1)
		exp++
	}
	if mant.hi >= 0x20_0000 { // at least 2^53: one round bit left
		if mant.lo&1 != 0 && (trunc != 0 || mant.lo&2 != 0) {
			mant = mantAdd(mant, sfMant{0, 1})
			if mant.hi >= 0x40_0000 {
				mant = mantShr(mant, 1)
				exp++
			}
		}
		mant = mantShr(mant, 1)
		exp++
	}
	return sf64{sign | uint32(exp+sfBias)<<20 | mant.hi&0xf_ffff, mant.lo}
}

// sfAdd is IEEE float64 addition over the chain's domain.
func sfAdd(a, b sf64) sf64 {
	fs, fm, fe, fz := sfUnpack(a)
	gs, gm, ge, gz := sfUnpack(b)
	if fz && gz {
		return sf64{fs & gs, 0}
	}
	if fz {
		return b
	}
	if gz {
		return a
	}
	if fe < ge || (fe == ge && mantLess(fm, gm)) {
		fs, fm, fe, gs, gm, ge = gs, gm, ge, fs, fm, fe
	}
	shift := uint32(fe - ge)
	fm = mantShl(fm, 2)
	gm = mantShl(gm, 2)
	gm, sticky := mantShrSticky(gm, shift)
	if fs == gs {
		fm = mantAdd(fm, gm)
	} else {
		fm = mantSub(fm, gm)
		if sticky != 0 {
			fm = mantSub(fm, sfMant{0, 1})
		}
	}
	if mantZero(fm) {
		fs = 0
	}
	return sfPack(fs, fm, fe-2, sticky)
}

func sfNeg(a sf64) sf64 { return sf64{a.hi ^ 0x8000_0000, a.lo} }

func sfAbs(a sf64) sf64 { return sf64{a.hi &^ 0x8000_0000, a.lo} }

func sfSub(a, b sf64) sf64 { return sfAdd(a, sfNeg(b)) }

// sfScalePow2 multiplies by 2^k; the chain's values never leave the normal
// exponent range.
func sfScalePow2(a sf64, k int32) sf64 {
	if a.hi&0x7ff0_0000 == 0 {
		return a
	}
	return sf64{a.hi + uint32(k)<<20, a.lo}
}

// sfDivSmall divides by a small positive integer constant (3 or 5 in the
// chain), exactly rounded.
func sfDivSmall(a sf64, d uint32) sf64 {
	sign, mant, exp, zero := sfUnpack(a)
	if zero {
		return sf64{sign, 0}
	}
	q, rem := mantDivSmall(mantShl(mant, 4), d)
	trunc := uint32(0)
	if rem != 0 {
		trunc = 1
	}
	return sfPack(sign, q, exp-4, trunc)
}

// sfMulU16 multiplies a non-negative integer below 2^16 by a positive
// constant, exactly rounded, via 16-bit limb products.
func sfMulU16(m uint32, c sf64) sf64 {
	if m == 0 {
		return sf64{}
	}
	_, cm, ce, _ := sfUnpack(c)
	t0 := (cm.lo & 0xffff) * m
	t1 := (cm.lo>>16)*m + t0>>16
	t2 := (cm.hi&0xffff)*m + t1>>16
	t3 := (cm.hi>>16)*m + t2>>16
	p0 := t0&0xffff | t1<<16
	p1 := t2&0xffff | (t3&0xffff)<<16
	p2 := t3 >> 16
	if p2 == 0 {
		return sfPack(0, sfMant{p1, p0}, ce, 0)
	}
	// The product exceeds 64 bits; shift right by the minimal amount that
	// fits the pair, jamming shifted-out bits into the sticky flag.
	s := uint32(bits.Len32(p2))
	trunc := uint32(0)
	if p0&(1<<s-1) != 0 {
		trunc = 1
	}
	return sfPack(0, sfMant{p2<<(32-s) | p1>>s, p1<<(32-s) | p0>>s}, ce+int32(s), trunc)
}

// sfFromU32 converts exactly.
func sfFromU32(v uint32) sf64 {
	if v == 0 {
		return sf64{}
	}
	return sfPack(0, sfMant{0, v}, 52, 0)
}

// sfFromI32 converts exactly.
func sfFromI32(v int32) sf64 {
	if v < 0 {
		return sfNeg(sfFromU32(uint32(-v)))
	}
	return sfFromU32(uint32(v))
}

// sfKey maps raw bits to a totally ordered unsigned key (sign-magnitude
// flip). Positive and negative zero compare unequal, which the chain never
// observes since it produces no negative zeros.
func sfKey(a sf64) (uint32, uint32) {
	if a.hi&0x8000_0000 != 0 {
		return ^a.hi, ^a.lo
	}
	return a.hi | 0x8000_0000, a.lo
}

func sfLess(a, b sf64) bool {
	ah, al := sfKey(a)
	bh, bl := sfKey(b)
	return ah < bh || (ah == bh && al < bl)
}

func sfLessEq(a, b sf64) bool {
	ah, al := sfKey(a)
	bh, bl := sfKey(b)
	return ah < bh || (ah == bh && al <= bl)
}

// sfTruncI32 is Go's int(x) truncation toward zero for |x| < 2^31.
func sfTruncI32(a sf64) int32 {
	sign, mant, exp, zero := sfUnpack(a)
	if zero || exp < 0 {
		return 0
	}
	sh := uint32(52 - exp)
	var v int32
	if sh >= 32 {
		v = int32(mant.hi >> (sh - 32))
	} else {
		v = int32(mantShr(mant, sh).lo)
	}
	if sign != 0 {
		return -v
	}
	return v
}

// sfDomainPool builds a deterministic pool of chain-domain values: exact
// integers and halves, rounded thirds and fifths, and random normal bit
// patterns across the exponent spread the chain produces.
func sfDomainPool(rng *rand.Rand) []float64 {
	pool := []float64{0, 0.5, 1, 1.5, 2, 2.5, 3, 4.75, 100.5, 1024, 16777215}
	for k := 1; k <= 4096; k++ {
		pool = append(pool, float64(k)/3.0, float64(k)/5.0, float64(k))
	}
	for range 4096 {
		mant := rng.Uint64() & (1<<52 - 1)
		exp := uint64(sfBias - 40 + rng.Intn(80))
		pool = append(pool, math.Float64frombits(exp<<52|mant))
	}
	// Halfway-forcing neighbors: all-ones and single-bit mantissas.
	for _, exp := range []uint64{sfBias - 3, sfBias, sfBias + 7, sfBias + 30} {
		pool = append(pool,
			math.Float64frombits(exp<<52|(1<<52-1)),
			math.Float64frombits(exp<<52|1),
			math.Float64frombits(exp<<52),
		)
	}
	return pool
}

// TestGPUFinderChainSoftfloatAdd proves sfAdd bit-identical to hardware
// float64 addition and subtraction over the domain pool.
func TestGPUFinderChainSoftfloatAdd(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	pool := sfDomainPool(rng)
	check := func(a, b float64) {
		t.Helper()
		got := sfAdd(sfFromFloat(a), sfFromFloat(b)).float()
		if math.Float64bits(got) != math.Float64bits(a+b) {
			t.Fatalf("sfAdd(%x, %x) = %x, float64 add = %x",
				math.Float64bits(a), math.Float64bits(b),
				math.Float64bits(got), math.Float64bits(a+b))
		}
		gotSub := sfSub(sfFromFloat(a), sfFromFloat(b)).float()
		if math.Float64bits(gotSub) != math.Float64bits(a-b) {
			t.Fatalf("sfSub(%x, %x) = %x, float64 sub = %x",
				math.Float64bits(a), math.Float64bits(b),
				math.Float64bits(gotSub), math.Float64bits(a-b))
		}
	}
	for range 2_000_000 {
		check(pool[rng.Intn(len(pool))], pool[rng.Intn(len(pool))])
	}
	for _, a := range pool[:600] {
		for _, b := range pool[:600] {
			check(a, b)
			check(-a, b)
			check(a, -b)
		}
	}
}

// TestGPUFinderChainSoftfloatDivScale proves the small-divisor division, the
// 2.5 tolerance divisor route, and the power-of-two scaling bit-identical to
// hardware float64.
func TestGPUFinderChainSoftfloatDivScale(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	pool := sfDomainPool(rng)
	for _, a := range pool {
		if bits := math.Float64bits(sfDivSmall(sfFromFloat(a), 3).float()); bits != math.Float64bits(a/3.0) {
			t.Fatalf("sfDivSmall(%x, 3) = %x, float64 = %x", math.Float64bits(a), bits, math.Float64bits(a/3.0))
		}
		if bits := math.Float64bits(sfDivSmall(sfFromFloat(a), 5).float()); bits != math.Float64bits(a/5.0) {
			t.Fatalf("sfDivSmall(%x, 5) = %x, float64 = %x", math.Float64bits(a), bits, math.Float64bits(a/5.0))
		}
		mean25 := sfDivSmall(sfScalePow2(sfFromFloat(a), 1), 5).float()
		if math.Float64bits(mean25) != math.Float64bits(a/2.5) {
			t.Fatalf("2.5 route for %x = %x, float64 = %x", math.Float64bits(a), math.Float64bits(mean25), math.Float64bits(a/2.5))
		}
		for _, k := range []int32{-2, -1, 1, 2} {
			want := a * math.Ldexp(1, int(k))
			if bits := math.Float64bits(sfScalePow2(sfFromFloat(a), k).float()); bits != math.Float64bits(want) {
				t.Fatalf("sfScalePow2(%x, %d) = %x, float64 = %x", math.Float64bits(a), k, bits, math.Float64bits(want))
			}
		}
	}
}

// TestGPUFinderChainSoftfloatMulTruncFrom proves the diagonal-length constant
// multiply, truncation and integer conversion bit-identical to float64. The
// constant is crossCheckColor's 5.0/(2.0*1.41421) with moduleNumber 5.
func TestGPUFinderChainSoftfloatMulTruncFrom(t *testing.T) {
	k := 5.0 / (2.0 * 1.41421)
	kBits := sfFromFloat(k)
	for m := uint32(0); m < 1<<16; m++ {
		want := float64(m) * k
		if bits := math.Float64bits(sfMulU16(m, kBits).float()); bits != math.Float64bits(want) {
			t.Fatalf("sfMulU16(%d) = %x, float64 = %x", m, bits, math.Float64bits(want))
		}
		if got := sfTruncI32(sfMulU16(m, kBits)); got != int32(want) {
			t.Fatalf("trunc(sfMulU16(%d)) = %d, int(float64) = %d", m, got, int32(want))
		}
		if bits := math.Float64bits(sfFromU32(m).float()); bits != math.Float64bits(float64(m)) {
			t.Fatalf("sfFromU32(%d) = %x, float64 = %x", m, bits, math.Float64bits(float64(m)))
		}
	}
	rng := rand.New(rand.NewSource(3))
	pool := sfDomainPool(rng)
	for _, a := range pool {
		if a >= 1<<31 {
			continue
		}
		for _, v := range []float64{a, -a} {
			if got := sfTruncI32(sfFromFloat(v)); got != int32(v) {
				t.Fatalf("sfTruncI32(%x) = %d, int(float64) = %d", math.Float64bits(v), got, int32(v))
			}
		}
	}
	for v := int32(-70000); v <= 70000; v += 7 {
		if bits := math.Float64bits(sfFromI32(v).float()); bits != math.Float64bits(float64(v)) {
			t.Fatalf("sfFromI32(%d) = %x, float64 = %x", v, bits, math.Float64bits(float64(v)))
		}
	}
}

// TestGPUFinderChainSoftfloatCompare proves the ordered-key comparisons match
// hardware float64 comparisons over the domain (no negative zeros).
func TestGPUFinderChainSoftfloatCompare(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	pool := sfDomainPool(rng)
	for range 2_000_000 {
		a := pool[rng.Intn(len(pool))]
		b := pool[rng.Intn(len(pool))]
		if rng.Intn(2) == 0 {
			a = -a
		}
		if rng.Intn(4) == 0 {
			b = a
		}
		if a == 0 {
			a = 0 // normalize any -0 from negation out of the domain
		}
		if b == 0 {
			b = 0
		}
		sa, sb := sfFromFloat(a), sfFromFloat(b)
		if got := sfLess(sa, sb); got != (a < b) {
			t.Fatalf("sfLess(%v, %v) = %v", a, b, got)
		}
		if got := sfLessEq(sa, sb); got != (a <= b) {
			t.Fatalf("sfLessEq(%v, %v) = %v", a, b, got)
		}
		if got := math.Float64bits(sfAbs(sa).float()); got != math.Float64bits(math.Abs(a)) {
			t.Fatalf("sfAbs(%v) mismatch", a)
		}
	}
}
