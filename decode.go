package jabcode

import (
	"fmt"
	"image"

	"github.com/srlehn/jabcode/internal/read"
)

// Decode decodes the data of a JAB Code from img: the primary symbol and any
// docked secondary symbols. Reading a JAB Code from a file is stdlib decoding
// (e.g. png.Decode) followed by Decode.
func Decode(img image.Image) ([]byte, error) {
	return read.Decode(img)
}

// DecodeWithConformance decodes img under the selected wire-format profile.
// ConformanceCReference preserves compatibility with the reference C tools and
// is what Decode uses. ConformanceISO23634 selects the experimental ISO/IEC
// 23634:2022 target: its palette, interleaving, LDPC and message-control
// behavior, with reserved color modes rejected. Its returned bytes are the
// ECI-capable reader transmission rather than the raw encoded payload: every
// message starts with ]j1, ]j4 or ]j5, literal data backslashes are doubled,
// ECI assignments are escaped, and the JAB ISO/IEC 15434 switch expands its
// message envelope. That expansion validates the JAB macro controls, not the
// application data inside the format envelope. Annex F range reduction has not
// been independently validated.
func DecodeWithConformance(img image.Image, mode ConformanceMode) ([]byte, error) {
	if !mode.valid() {
		return nil, fmt.Errorf("jabcode: invalid conformance mode %d", mode)
	}
	return read.DecodeProfile(img, mode.profile())
}
