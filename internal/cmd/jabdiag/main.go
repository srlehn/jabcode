// Command jabdiag reports where JAB Code primary-symbol detection dies on a
// capture, for debugging the detector. The image path is taken from the
// JABDIAG_IMG environment variable (PNG or JPEG); the report is written to
// stdout. When JABDIAG_OUT names a directory, each stage additionally writes
// an annotated image there (region boxes, finder candidates and quad, warped
// sampling grid, upscaled sampled matrix, palette swatches), numbered in
// report order. It never decodes for output, only diagnoses, and is not part
// of the public API.
package main

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"

	"github.com/srlehn/jabcode/internal/decode"
)

func main() {
	path := os.Getenv("JABDIAG_IMG")
	if path == "" {
		fmt.Fprintln(os.Stderr, "jabdiag: set JABDIAG_IMG to the capture image path (PNG or JPEG)")
		os.Exit(2)
	}
	f, err := os.Open(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "jabdiag: open:", err)
		os.Exit(1)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		fmt.Fprintln(os.Stderr, "jabdiag: decode image:", err)
		os.Exit(1)
	}
	decode.Diagnose(img, os.Stdout, os.Getenv("JABDIAG_OUT"))
}
