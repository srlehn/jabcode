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
//
// When the ISO variant succeeds, Decode returns the ECI-capable reader
// transmission rather than the raw encoded payload: every message starts with
// ]j1, ]j4 or ]j5, literal data backslashes are doubled, ECI assignments are
// escaped, and the JAB ISO/IEC 15434 switch expands its message envelope. That
// expansion validates the JAB macro controls, not the application data inside
// the format envelope. The ISO variant rejects reserved color modes. Its Annex
// F range reduction has not been independently validated.
func Decode(img image.Image) ([]byte, error) {
	return read.Decode(img)
}
