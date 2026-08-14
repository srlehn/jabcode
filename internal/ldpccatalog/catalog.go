// Package ldpccatalog carries the precomputed pivot transcripts of every
// message parity-check code the decoder can select.
//
// Building a decoder parity-check matrix is a Gallager construction followed by
// a systematic reduction. The construction is cheap and the reduction is not,
// yet the reduction discovers only one number per parity row: the column that
// row pivots at. Everything after it - the rank, the dependent rows, the column
// arrangement and both kinds of column swap - is a deterministic function of
// that sequence. So the sequence is precomputed here and the reduction is
// deleted from the decode path rather than made faster.
//
// The catalog covers the complete legal key space rather than a reachable
// subset. A capacity is always a multiple of the row weight and never exceeds
// MaxCapacity, which bounds the space at a size worth storing outright; total
// coverage then discharges the completeness question by construction instead of
// by an enumeration that would have to be re-proved whenever a symbol shape
// changes.
package ldpccatalog

import (
	"encoding/binary"

	"github.com/srlehn/jabcode/internal/wire"
)

// The bounds of a selectable message code. A parity-check matrix outside them
// is refused before it reaches the catalog, so a missing slot is a defect and
// never an ordinary outcome.
const (
	// MinColWeight and MaxRowWeight bound the ECC parameters a symbol may name.
	MinColWeight = 3
	MinRowWeight = 4
	MaxRowWeight = 11

	// MaxCapacity is the longest sub-block a correction may hold. It matches
	// MAX_SUB in the device shaders, which
	// detect.TestGPULDPCMatrixConstantsMatchShader holds together.
	MaxCapacity = 2816
)

// PivotNone marks a dependent parity row inside a transcript. It is stored in
// the same twelve bits as a real pivot column, which reach 4095 while the
// largest column a legal capacity can name is MaxCapacity-1.
//
// SweepNone is how ecc.PivotSweep spells the same thing. The two differ because
// the sweep is not bounded by the catalog's twelve-bit field, and Pivots
// translates between them rather than narrowing the sweep's sentinel.
const (
	PivotNone = 0xFFF
	SweepNone = 0xFFFF
)

// Header words and the packing width of the record area.
const (
	magic       = 0x4A424C50 // "JBLP"
	formatMajor = 1

	headerWords = 4
	pivotBits   = 12
)

// Slot numbers a key within one generator's directory. Keys are dense by
// construction - every multiple of wr up to MaxCapacity is legal for every
// admissible column weight - so the directory is a perfect hash with no
// probing, no collision handling and no stored key to compare against.
//
// The device computes the same number from the payload shape it already holds,
// which is why the arithmetic is kept trivial.
func Slot(wc, wr, capacity int) (int, bool) {
	if wr < MinRowWeight || wr > MaxRowWeight || wc < MinColWeight || wc >= wr {
		return 0, false
	}
	if capacity <= 0 || capacity > MaxCapacity || capacity%wr != 0 {
		return 0, false
	}
	return rowWeightBase(wr) + (wc-MinColWeight)*capacityCount(wr) + capacity/wr - 1, true
}

// capacityCount is how many capacities one row weight admits.
func capacityCount(wr int) int { return MaxCapacity / wr }

// rowWeightBase is where one row weight's block of slots starts.
func rowWeightBase(wr int) int {
	base := 0
	for w := MinRowWeight; w < wr; w++ {
		base += (w - MinColWeight) * capacityCount(w)
	}
	return base
}

// SlotCount is the size of one generator's directory.
func SlotCount() int { return rowWeightBase(MaxRowWeight + 1) }

// StoredRows is how many pivots a key contributes to the record area. The
// leading Gallager block is omitted because ecc.LeadingPivot already gives it.
func StoredRows(wc, wr, capacity int) int { return (wc - 1) * (capacity / wr) }

// Generator names the pseudo-random column permutation a wire variant builds
// its Gallager copies with. The two families need separate catalogs because a
// permutation change moves every pivot.
type Generator uint8

const (
	// GeneratorISO is the ISO/IEC 23634 permutation.
	GeneratorISO Generator = 0
	// GeneratorLCG is the C-family 64-bit permutation.
	GeneratorLCG Generator = 1
)

// GeneratorOf reports which catalog a wire variant reads.
func GeneratorOf(variant wire.Variant) Generator {
	if variant.UsesISO23634Base() {
		return GeneratorISO
	}
	return GeneratorLCG
}

// Variant returns a wire variant that builds its codes with g, for a consumer
// that has a generator and needs to call the matrix builder.
func (g Generator) Variant() wire.Variant {
	if g == GeneratorISO {
		return wire.ISO23634
	}
	return wire.CurrentC
}

// Blob is one generator's catalog as the device receives it: a little-endian
// u32 stream of the header, the directory and the packed record area. Returns
// nil when the build does not compile that generator's wire family.
func Blob(g Generator) []byte { return blob(g) }

func blob(g Generator) []byte {
	if g == GeneratorISO {
		return isoCatalog
	}
	return lcgCatalog
}

// Combined is every compiled catalog behind one prefix, as the device receives
// it: magic, format, then one base word per generator giving the word offset of
// that generator's catalog, or zero where the build does not carry it.
//
// One buffer rather than one per generator is what lets the device select a
// generator by indexing rather than by binding a second resource, and it is what
// lets the whole catalog ride a single device allocation.
func Combined() []byte {
	const generators = 2
	const prefix = (2 + generators) * 4
	parts := [generators][]byte{blob(GeneratorISO), blob(GeneratorLCG)}
	bases := [generators]uint32{}
	at := prefix
	for g, part := range parts {
		if len(part) == 0 || !Wellformed(Generator(g)) {
			continue
		}
		bases[g] = uint32(at / 4)
		at += len(part)
	}

	out := make([]byte, at)
	binary.LittleEndian.PutUint32(out[0:], magic)
	binary.LittleEndian.PutUint32(out[4:], formatMajor)
	for g, base := range bases {
		binary.LittleEndian.PutUint32(out[(2+g)*4:], base)
		if base != 0 {
			copy(out[base*4:], parts[g])
		}
	}
	return out
}

// Wellformed reports whether a catalog is present and its header describes the
// directory and record area this build expects. It is what a consumer checks
// before uploading, so a catalog regenerated under a changed format is caught
// at the boundary rather than as a wrong parity matrix.
func Wellformed(g Generator) bool {
	data := blob(g)
	if len(data) < headerWords*4 ||
		binary.LittleEndian.Uint32(data[0:]) != magic ||
		binary.LittleEndian.Uint32(data[4:]) != formatMajor ||
		int(binary.LittleEndian.Uint32(data[8:])) != SlotCount() {
		return false
	}
	pivots := binary.LittleEndian.Uint32(data[12:])
	want := recordBase(data) + int((uint64(pivots)*pivotBits+31)/32)*4
	return len(data) == want
}

// pivotStart reads the record-area index where a slot's transcript begins.
func pivotStart(data []byte, slot int) uint32 {
	return binary.LittleEndian.Uint32(data[(headerWords+slot)*4:])
}

// recordBase is the byte offset of the packed record area.
func recordBase(data []byte) int {
	return (headerWords + int(binary.LittleEndian.Uint32(data[2*4:]))) * 4
}

// pivotAt reads one twelve-bit field from the packed record area. The stream is
// filled least-significant bits first, so a field that straddles a word
// boundary continues in the low bits of the next one.
func pivotAt(data []byte, base int, index uint32) uint16 {
	bit := uint64(index) * pivotBits
	at := base + int(bit/32)*4
	value := uint64(binary.LittleEndian.Uint32(data[at:])) >> (bit % 32)
	if bit%32 > 32-pivotBits {
		value |= uint64(binary.LittleEndian.Uint32(data[at+4:])) << (32 - bit%32)
	}
	return uint16(value & (1<<pivotBits - 1))
}

// Pivots reconstructs one code's full pivot sequence, leading block included,
// in the form ecc.PivotSweep would have returned. It is the host mirror of what
// the device reconstruction does, and exists so the catalog can be checked
// against the builder it replaces.
func Pivots(g Generator, wc, wr, capacity int) ([]uint16, bool) {
	data := blob(g)
	slot, ok := Slot(wc, wr, capacity)
	if !ok || len(data) == 0 {
		return nil, false
	}
	blockRows := capacity / wr
	pivots := make([]uint16, blockRows*wc)
	for i := range blockRows {
		pivots[i] = uint16(i * wr)
	}
	base, start := recordBase(data), pivotStart(data, slot)
	for i := blockRows; i < len(pivots); i++ {
		value := pivotAt(data, base, start+uint32(i-blockRows))
		if value == PivotNone {
			pivots[i] = SweepNone
			continue
		}
		pivots[i] = value
	}
	return pivots, true
}
