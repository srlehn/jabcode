package core

// PayloadRequest describes one symbol's payload correction in the terms a
// device stage can answer: which sampled grid, the metadata already interpreted
// from it, and how much of the grid that interpretation consumed.
//
// It carries no module data. A device corrector is only usable when the grid it
// would read is still the one it produced, which is why Matrix identifies the
// sample rather than supplying it.
type PayloadRequest struct {
	Matrix *Bitmap
	Symbol *DecodedSymbol

	// MetadataModules is how many modules the metadata walk consumed, which is
	// all a device stage needs to replay that walk and reserve the same modules
	// the host reserved.
	MetadataModules int

	// DataModules is how many modules carry payload, so the caller and the
	// device agree on the codeword length before either builds it.
	DataModules int

	// RequireFixedPatternAgreement defers the module-grid half of admission to
	// the resident correction submission. The host has already accepted the
	// palette-coherence half before setting it.
	RequireFixedPatternAgreement bool

	NormalizedPalette []float64
	PaletteThresholds []float64
}

// PayloadDevice corrects a symbol's payload where its module grid already is:
// classification, unmasking, deinterleaving, hard error correction, and its
// soft-decision retry, with the corrected message bits as the only result that
// crosses back.
//
// ok reports the post-retry syndrome; a false ok means the device answered but
// both correction stages gave up and the bits are unreliable. A corrector that
// cannot answer - an unsupported symbol shape, or a grid it no longer owns -
// returns an error, which is a decline rather than a failed read.
type PayloadDevice interface {
	// SupportsFixedPatternAdmission reports whether CorrectSymbolPayload can
	// gate correction on the format-fixed modules without materializing the
	// sampled grid.
	SupportsFixedPatternAdmission() bool
	CorrectSymbolPayload(request PayloadRequest) (dec []byte, ok bool, err error)
}

// GridDevice fills a sampled grid's module data from the device that produced
// it, for the host stages that have to read modules.
//
// A device-route sample is a shape-only bitmap until something asks, so that a
// read which never leaves the device never pays for the grid. Materializing is
// deliberately a call and not a side effect of reading Pix: the stages that
// need modules are all fallbacks, and they should be countable.
//
// It reports false when the grid can no longer be produced - the device has
// moved on to another sample, or the route context is gone - and the caller
// then fails the way it fails for any matrix it cannot read.
type GridDevice interface {
	MaterializeGrid(matrix *Bitmap) bool
}
