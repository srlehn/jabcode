package core

// PrimaryDeviceResult is the only host-visible output of a resident primary
// decode: interpreted metadata and corrected message bits. Locate, sampling,
// classification and correction workspaces remain device-owned.
type PrimaryDeviceResult struct {
	Metadata  PrimaryMetadata
	Payload   []byte
	PayloadOK bool
	Evidence  PrimaryEvidence
}

// PrimaryDevice decodes one primary symbol from a module grid that already
// resides on the device. An error is a decline, so the host may materialize the
// same grid and run its full path. PayloadOK false is an answered decode failure
// and must not repeat correction on the host.
type PrimaryDevice interface {
	DecodePrimary(matrix *Bitmap, symbol *DecodedSymbol) (PrimaryDeviceResult, error)
}
