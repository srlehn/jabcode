package core

import "image"

// PrimaryMetadata is a primary symbol's metadata as a device stage resolves it:
// the colour mode, the embedded palette and the shape the symbol declares.
//
// It stops where geometry takes over. The reserved-module map and the
// normalized palette are both derivable from these fields alone - the first
// from the walk length, the second from the palette bytes - so a device that
// already holds them has no reason to ship them, and a host that rederives them
// gets the values its own arithmetic would have produced rather than a
// narrower float's.
type PrimaryMetadata struct {
	// Defaulted reports that part I did not resolve a colour mode and the
	// symbol is to be read under default metadata, which is a ladder the host
	// owns rather than a failure.
	Defaulted bool

	// Rejected reports that the symbol declared a shape its own sample
	// contradicts: a side version that is not the sampled side, or ECC weights
	// out of order. The fields below still carry what was read, because a
	// caller's alignment-pattern retry uses the declared version.
	Rejected bool

	NC          int
	Colors      int
	SideVersion image.Point
	ECL         image.Point
	MaskType    int
	Palette     []byte

	// MetadataModules is how many modules the walk consumed, which is what lets
	// the host reserve exactly the modules the walk reserved.
	MetadataModules int

	PartISyndromeOK  bool
	PartIISyndromeOK bool
}

// MetadataDevice interprets a primary symbol's metadata where its module grid
// already is: part I and its correction, the embedded palette, part II and the
// fields it declares.
//
// As with PayloadDevice, Matrix identifies the sample rather than supplying it,
// because a device walk is only usable on the grid the device itself produced.
// A device that cannot answer - a grid it no longer owns, a colour mode outside
// what it implements, a walk that left the symbol - returns an error, which is
// a decline the host answers by walking the metadata itself.
type MetadataDevice interface {
	WalkPrimaryMetadata(matrix *Bitmap, symbol *DecodedSymbol) (PrimaryMetadata, error)
}
