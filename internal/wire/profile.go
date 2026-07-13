// Package wire defines the internal wire-format profiles shared by encoding,
// error correction and decoding.
package wire

// Profile selects behavior where the C reference and ISO/IEC 23634 differ.
type Profile uint8

const (
	// CReference preserves compatibility with the reference C implementation.
	CReference Profile = iota
	// ISO23634 selects the experimental ISO/IEC 23634:2022 target behavior.
	ISO23634
)

// Valid reports whether p names a supported wire-format profile.
func (p Profile) Valid() bool { return p == CReference || p == ISO23634 }
