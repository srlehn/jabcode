package jabcode

import (
	"image"

	"github.com/srlehn/jabcode/internal/read"
)

// Decode decodes the data of a JAB Code from img: the primary symbol and any
// docked secondary symbols. The untagged build accepts ISO/IEC 23634; optional
// decoder build tags add their wire families to the same automatic read. They
// never replace the ISO decoder. Reading a JAB Code from a file is stdlib
// decoding (e.g. png.Decode) followed by Decode.
func Decode(img image.Image) ([]byte, error) {
	return read.Decode(img)
}

// DecodeWithProfile decodes img under the selected compiled wire-format
// profile. Unlike Decode's additive compiled-profile search, this function
// forces one format. ProfileISO23634 selects the experimental ISO/IEC
// 23634:2022 target: its palette, interleaving, LDPC and message-control
// behavior, with reserved color modes rejected. Its returned bytes are the
// ECI-capable reader transmission rather than the raw encoded payload: every
// message starts with ]j1, ]j4 or ]j5, literal data backslashes are doubled,
// ECI assignments are escaped, and the JAB ISO/IEC 15434 switch expands its
// message envelope. That expansion validates the JAB macro controls, not the
// application data inside the format envelope. Annex F range reduction has not
// been independently validated.
func DecodeWithProfile(img image.Image, profile Profile) ([]byte, error) {
	if err := profile.validateAvailable(); err != nil {
		return nil, err
	}
	return read.DecodeProfile(img, profile.profile())
}
