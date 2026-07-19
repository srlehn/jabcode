//go:build jabcode_non_iso_encode

package jabcode

import publicencoder "github.com/srlehn/jabcode/encoder"

// Profile selects one encoder output format. Decoding is always additive and
// has no corresponding public profile selector.
type Profile = publicencoder.Profile

const (
	// ProfileISO23634 selects the experimental ISO/IEC 23634:2022 target.
	ProfileISO23634 = publicencoder.ProfileISO23634
	// ProfileHighColor extends the ISO format through 256 module colors.
	ProfileHighColor = publicencoder.ProfileHighColor
	// ProfileBSI selects the BSI TR-03137-2 format.
	ProfileBSI = publicencoder.ProfileBSI
)

// WithProfile selects an optional encoder output format. The untagged encoder
// is implicitly ISO and does not export this option or the Profile type.
func WithProfile(profile Profile) Option {
	return publicencoder.WithProfile(profile)
}
