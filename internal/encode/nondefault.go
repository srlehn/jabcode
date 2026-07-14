package encode

import (
	"github.com/srlehn/jabcode/internal/ecc"
	"github.com/srlehn/jabcode/internal/spec"
)

// optimalECC chooses the (wc, wr) code-rate weights that best fit the net
// data length into the given capacity. wcwr is updated only if a better fit is
// found.
func optimalECC(capacity, netDataLength int, wcwr *[2]int) {
	// Ports getOptimalECC in encoder.c.
	best := float64(capacity)
	for k := 3; k <= 8; k++ {
		for j := k + 1; j <= 9; j++ {
			dist := (capacity/j)*j - (capacity/j)*k - netDataLength
			if float64(dist) < best && dist >= 0 {
				wcwr[1] = j
				wcwr[0] = k
				best = float64(dist)
			}
		}
	}
}

// encodePrimaryMetadata builds and LDPC-encodes the primary symbol's metadata
// (Part I: Nc; Part II: version, ECC level, mask reference) into symbol.metadata.
func (e *encoder) encodePrimaryMetadata() {
	// Ports encodePrimaryMetadata in encoder.c.
	s := &e.symbols[0]
	const vLen, eLen, mskLen = 10, 6, 3
	nc := spec.Log2Int(e.colors) - 1
	v := ((spec.SizeToVersion(s.sideSize.X) - 1) << 5) + (spec.SizeToVersion(s.sideSize.Y) - 1)
	e1 := s.wcwr[0] - 3
	e2 := s.wcwr[1] - 4

	partI := make([]byte, spec.PrimaryMetadataPart1Length/2)
	writeBits(partI, nc, 0, len(partI))

	partII := make([]byte, spec.PrimaryMetadataPart2Length/2)
	writeBits(partII, v, 0, vLen)
	writeBits(partII, e1, vLen, 3)
	writeBits(partII, e2, vLen+3, 3)
	writeBits(partII, spec.DefaultMaskingReference, vLen+eLen, mskLen)

	encI := ecc.EncodeLDPCVariant(partI, 2, -1, e.format.Variant())
	encII := ecc.EncodeLDPCVariant(partII, 2, -1, e.format.Variant())
	s.metadata = append(append(make([]byte, 0, len(encI)+len(encII)), encI...), encII...)
}

// updatePrimaryMetadataPartII re-encodes Part II with a new mask reference,
// replacing the Part II portion of symbol.metadata.
func (e *encoder) updatePrimaryMetadataPartII(maskRef int) {
	// Ports updatePrimaryMetadataPartII in encoder.c.
	s := &e.symbols[0]
	const vLen, eLen, mskLen = 10, 6, 3
	v := ((spec.SizeToVersion(s.sideSize.X) - 1) << 5) + (spec.SizeToVersion(s.sideSize.Y) - 1)

	partII := make([]byte, spec.PrimaryMetadataPart2Length/2)
	writeBits(partII, v, 0, vLen)
	writeBits(partII, s.wcwr[0]-3, vLen, 3)
	writeBits(partII, s.wcwr[1]-4, vLen+3, 3)
	writeBits(partII, maskRef, vLen+eLen, mskLen)

	encII := ecc.EncodeLDPCVariant(partII, 2, -1, e.format.Variant())
	copy(s.metadata[spec.PrimaryMetadataPart1Length:], encII)
}

// placePrimaryMetadataPartII rewrites the Part II metadata modules in the symbol
// matrix after the mask reference has changed.
func (e *encoder) placePrimaryMetadataPartII() {
	// Ports placePrimaryMetadataPartII in encoder.c.
	s := &e.symbols[0]
	w, h := s.sideSize.X, s.sideSize.Y
	bpm := spec.Log2Int(e.colors)

	x, y := spec.PrimaryMetadataX, spec.PrimaryMetadataY
	count := 0
	colorPaletteSize := min(e.colors, 64) - spec.PaletteFinderColors(e.colors)
	moduleOffset := spec.PrimaryMetadataPart1ModuleNumber + colorPaletteSize*spec.PaletteCopies(e.colors)
	for range moduleOffset {
		count++
		spec.NextMetadataModuleInPrimary(h, w, count, &x, &y)
	}

	bitStart := spec.PrimaryMetadataPart1Length
	bitEnd := spec.PrimaryMetadataPart1Length + spec.PrimaryMetadataPart2Length
	mi := bitStart
	for mi <= bitEnd {
		ci := int(s.matrix[y*w+x])
		for j := 0; j < bpm && mi <= bitEnd; j++ {
			if mi < len(s.metadata) {
				if s.metadata[mi] == 0 {
					ci &^= 1 << (bpm - 1 - j)
				} else {
					ci |= 1 << (bpm - 1 - j)
				}
			}
			mi++
		}
		s.matrix[y*w+x] = byte(ci)
		count++
		spec.NextMetadataModuleInPrimary(h, w, count, &x, &y)
	}
}
