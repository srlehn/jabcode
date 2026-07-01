// Command jabdecode decodes a JAB Code from an image file and writes the decoded
// payload to stdout. It exits non-zero if the image cannot be read or no JAB Code
// decodes.
package main

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"

	"github.com/srlehn/jabcode"

	_ "github.com/gen2brain/heic"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: jabdecode <image>")
		os.Exit(2)
	}
	if err := run(os.Args[1]); err != nil {
		fmt.Fprintln(os.Stderr, "jabdecode:", err)
		os.Exit(1)
	}
}

func run(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		return fmt.Errorf("decode image %s: %w", path, err)
	}

	data, err := jabcode.Decode(img)
	if err != nil {
		return err
	}
	_, err = os.Stdout.Write(data)
	return err
}
