package jabcode

// getOptimalECC chooses the (wc, wr) code-rate weights that best fit the net
// data length into the given capacity (getOptimalECC in encoder.c). wcwr is
// updated only if a better fit is found.
func getOptimalECC(capacity, netDataLength int, wcwr *[2]int) {
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
// (Part I: Nc; Part II: version, ECC level, mask reference) into symbol.metadata
// (encodePrimaryMetadata in encoder.c).
func (e *Encoder) encodePrimaryMetadata() {
	s := &e.symbols[0]
	const vLen, eLen, mskLen = 10, 6, 3
	nc := log2int(e.colors) - 1
	v := ((size2version(s.sideSize.X) - 1) << 5) + (size2version(s.sideSize.Y) - 1)
	e1 := s.wcwr[0] - 3
	e2 := s.wcwr[1] - 4

	partI := make([]byte, primaryMetadataPart1Length/2)
	writeBits(partI, nc, 0, len(partI))

	partII := make([]byte, primaryMetadataPart2Length/2)
	writeBits(partII, v, 0, vLen)
	writeBits(partII, e1, vLen, 3)
	writeBits(partII, e2, vLen+3, 3)
	writeBits(partII, defaultMaskingReference, vLen+eLen, mskLen)

	encI := encodeLDPC(partI, 2, -1)
	encII := encodeLDPC(partII, 2, -1)
	s.metadata = append(append(make([]byte, 0, len(encI)+len(encII)), encI...), encII...)
}

// updatePrimaryMetadataPartII re-encodes Part II with a new mask reference,
// replacing the Part II portion of symbol.metadata (updatePrimaryMetadataPartII).
func (e *Encoder) updatePrimaryMetadataPartII(maskRef int) {
	s := &e.symbols[0]
	const vLen, eLen, mskLen = 10, 6, 3
	v := ((size2version(s.sideSize.X) - 1) << 5) + (size2version(s.sideSize.Y) - 1)

	partII := make([]byte, primaryMetadataPart2Length/2)
	writeBits(partII, v, 0, vLen)
	writeBits(partII, s.wcwr[0]-3, vLen, 3)
	writeBits(partII, s.wcwr[1]-4, vLen+3, 3)
	writeBits(partII, maskRef, vLen+eLen, mskLen)

	encII := encodeLDPC(partII, 2, -1)
	copy(s.metadata[primaryMetadataPart1Length:], encII)
}

// placePrimaryMetadataPartII rewrites the Part II metadata modules in the symbol
// matrix after the mask reference has changed (placePrimaryMetadataPartII).
func (e *Encoder) placePrimaryMetadataPartII() {
	s := &e.symbols[0]
	w, h := s.sideSize.X, s.sideSize.Y
	bpm := log2int(e.colors)

	x, y := primaryMetadataX, primaryMetadataY
	count := 0
	colorPaletteSize := min(e.colors-2, 62)
	moduleOffset := primaryMetadataPart1ModuleNumber + colorPaletteSize*colorPaletteNumber
	for range moduleOffset {
		count++
		getNextMetadataModuleInPrimary(h, w, count, &x, &y)
	}

	bitStart := primaryMetadataPart1Length
	bitEnd := primaryMetadataPart1Length + primaryMetadataPart2Length
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
		getNextMetadataModuleInPrimary(h, w, count, &x, &y)
	}
}
