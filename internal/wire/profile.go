// Package wire defines the internal wire-format profiles shared by encoding,
// error correction and decoding.
package wire

// Profile selects one complete JAB Code wire format.
type Profile uint8

const (
	// ISO23634 selects the experimental ISO/IEC 23634:2022 target behavior.
	ISO23634 Profile = iota
	// HighColor extends the ISO23634 base with the reserved color modes through
	// 256 colors.
	HighColor
	// BSI selects the BSI TR-03137 wire format.
	BSI
	// Legacy selects historical C-reference wire formats.
	Legacy

	// CReference is the old internal name for Legacy.
	CReference = Legacy
)

// Valid reports whether p names a supported wire-format profile.
func (p Profile) Valid() bool {
	return p == ISO23634 || p == HighColor || p == BSI || p == Legacy
}

// UsesISO23634Base reports whether the profile uses the ISO palette, PRNG,
// interleaving, LDPC and message-control rules. HighColor changes only the
// ISO-reserved color-mode range.
func (p Profile) UsesISO23634Base() bool { return p == ISO23634 || p == HighColor }

// Profiles is a bitmask of decoder wire formats. Encoding and a decoded
// symbol always use one Profile; a reader can accept any compiled member of a
// Profiles mask without making those formats mutually exclusive.
type Profiles uint8

// Mask returns the one-bit decoder mask for p, or zero for an invalid profile.
func (p Profile) Mask() Profiles {
	if !p.Valid() {
		return 0
	}
	return 1 << p
}

// Has reports whether p is enabled in the decoder mask.
func (profiles Profiles) Has(p Profile) bool {
	return profiles&p.Mask() != 0
}

// Valid reports whether the mask is nonempty and contains only known profiles.
func (profiles Profiles) Valid() bool {
	const all = Profiles(1<<ISO23634 | 1<<HighColor | 1<<BSI | 1<<Legacy)
	return profiles != 0 && profiles&^all == 0
}
