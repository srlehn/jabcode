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
// is what Decode uses. ConformanceISO23634 selects the ISO/IEC 23634:2022
// palette, interleaving and LDPC behavior and rejects reserved color modes.
// ECI and FNC1 controls currently end decoding as they do in the C profile.
func DecodeWithConformance(img image.Image, mode ConformanceMode) ([]byte, error) {
	if !mode.valid() {
		return nil, fmt.Errorf("jabcode: invalid conformance mode %d", mode)
	}
	return read.DecodeProfile(img, mode.profile())
}
