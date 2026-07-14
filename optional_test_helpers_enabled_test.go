//go:build jabcode_non_iso_encode && jabcode_high_color

package jabcode

const highColorRoundTripEnabled = true

func highColorOption() Option { return WithProfile(ProfileHighColor) }
