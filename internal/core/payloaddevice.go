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

	NormalizedPalette []float64
	PaletteThresholds []float64
}

// PayloadDevice corrects a symbol's payload where its module grid already is:
// classification, unmasking, deinterleaving and hard error correction, with the
// corrected message bits as the only result that crosses back.
//
// ok reports the post-correction syndrome, exactly as the host decoder's does;
// a false ok means the correction gave up and the bits are unreliable. A
// corrector that cannot answer - an unsupported symbol shape, or a grid it no
// longer owns - returns an error, which is a decline rather than a failed read.
type PayloadDevice interface {
	CorrectSymbolPayload(request PayloadRequest) (dec []byte, ok bool, err error)
}
