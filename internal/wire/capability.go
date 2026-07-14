// Package wire defines internal JAB Code wire variants, decoder capability
// sets and encoder format choices.
package wire

// Variant identifies the wire interpretation of one decoded symbol. CurrentC
// and PreV2C intentionally remain distinct even where they share PRNG, ECC or
// message rules because their finder, metadata, palette and secondary layouts
// differ.
type Variant uint8

const (
	// ISO23634 is the experimental ISO/IEC 23634:2022 target.
	ISO23634 Variant = iota
	// ISOHighColor extends the ISO physical family through 256 module colors.
	ISOHighColor
	// CurrentC is the current C-reference format under the current finder family.
	CurrentC
	// BSI is the BSI TR-03137 format.
	BSI
	// PreV2C is the pre-v2.0 C-reference format under the BSI-era finder family.
	PreV2C
)

// Valid reports whether v names one decoder wire variant.
func (v Variant) Valid() bool {
	return v == ISO23634 || v == ISOHighColor || v == CurrentC || v == BSI || v == PreV2C
}

// UsesISO23634Base reports whether the variant uses the ISO palette, PRNG,
// interleaving, LDPC and message-control rules. ISOHighColor changes only the
// ISO-reserved color-mode range.
func (v Variant) UsesISO23634Base() bool { return v == ISO23634 || v == ISOHighColor }

// Capabilities is an additive bitmask of decoder variants.
type Capabilities uint8

// Mask returns the one-bit decoder capability for v, or zero when v is invalid.
func (v Variant) Mask() Capabilities {
	if !v.Valid() {
		return 0
	}
	return 1 << v
}

// Has reports whether v is enabled in the decoder capability set.
func (capabilities Capabilities) Has(v Variant) bool {
	return capabilities&v.Mask() != 0
}

// Valid reports whether the set is nonempty and contains only known variants.
func (capabilities Capabilities) Valid() bool {
	const all = Capabilities(1<<ISO23634 | 1<<ISOHighColor | 1<<CurrentC | 1<<BSI | 1<<PreV2C)
	return capabilities != 0 && capabilities&^all == 0
}

// Encoding identifies one output format. It is deliberately not a decoder
// capability set: encoding selects exactly one format and performs no probing.
type Encoding uint8

const (
	EncodeISO23634 Encoding = iota
	EncodeISOHighColor
	EncodeCurrentC
	EncodeBSI
)

// Valid reports whether e names one encoder output format.
func (e Encoding) Valid() bool {
	return e == EncodeISO23634 || e == EncodeISOHighColor || e == EncodeCurrentC || e == EncodeBSI
}

// Variant returns the decoder variant whose wire rules define e.
func (e Encoding) Variant() Variant {
	switch e {
	case EncodeISO23634:
		return ISO23634
	case EncodeISOHighColor:
		return ISOHighColor
	case EncodeCurrentC:
		return CurrentC
	case EncodeBSI:
		return BSI
	default:
		return Variant(255)
	}
}
