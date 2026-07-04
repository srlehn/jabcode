// Package decode turns a sampled symbol matrix into message bits: metadata
// and palette decoding, module colour classification, and the
// LDPC/demask/deinterleave message decode, for primary and docked secondary
// symbols. Detection lives in the sibling detect package; the read package
// coordinates the two, and the public Decode API in the parent jabcode
// package wraps read.Decode.
package decode
