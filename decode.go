package jabcode

import (
	"image"

	"github.com/srlehn/jabcode/internal/decode"
)

// Decode decodes the data of a JAB Code from img: the primary symbol and any
// docked secondary symbols. Reading a JAB Code from a file is stdlib decoding
// (e.g. png.Decode) followed by Decode.
func Decode(img image.Image) ([]byte, error) {
	return decode.Decode(img)
}
