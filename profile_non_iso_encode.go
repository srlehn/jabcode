//go:build jabcode_non_iso_encode

package jabcode

import (
	"fmt"

	"github.com/srlehn/jabcode/internal/wire"
)

// Profile selects one encoder output format. Decoding is always additive and
// has no corresponding public profile selector.
type Profile uint8

const (
	// ProfileISO23634 selects the experimental ISO/IEC 23634:2022 target.
	// Annex F range reduction still lacks an independent wire oracle.
	ProfileISO23634 Profile = iota
	// ProfileHighColor extends the ISO format through 256 module colors. It is
	// non-standard and intended primarily for lossless digital use.
	ProfileHighColor
	// ProfileBSI selects the BSI TR-03137-2 format, including its primary and
	// docked-secondary metadata, palette and finder-pattern layouts.
	ProfileBSI
)

func (p Profile) String() string {
	switch p {
	case ProfileISO23634:
		return "iso"
	case ProfileHighColor:
		return "hc"
	case ProfileBSI:
		return "bsi"
	default:
		return fmt.Sprintf("profile(%d)", p)
	}
}

func (p Profile) encoding() wire.Encoding {
	switch p {
	case ProfileISO23634:
		return wire.EncodeISO23634
	case ProfileHighColor:
		return wire.EncodeISOHighColor
	case ProfileBSI:
		return wire.EncodeBSI
	default:
		return wire.Encoding(255)
	}
}

// WithProfile selects an optional encoder output format. The untagged encoder
// is implicitly ISO and does not export this option or the Profile type.
func WithProfile(profile Profile) Option {
	return func(e *Encoder) { e.format = profile.encoding() }
}
