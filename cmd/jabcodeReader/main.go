// Command jabcodeReader decodes a JAB Code from an image (PNG or JPEG).
package main

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"

	"github.com/srlehn/jabcode"
)

func main() {
	var output string
	args := os.Args[1:]
	var input string
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--output", "-o":
			if i+1 < len(args) {
				i++
				output = args[i]
			}
		default:
			input = args[i]
		}
	}
	if input == "" {
		fmt.Fprintln(os.Stderr, "usage: jabcodeReader IMAGE.png [--output FILE]")
		os.Exit(2)
	}

	f, err := os.Open(input)
	if err != nil {
		fatal(err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		fatal(err)
	}
	data, err := jabcode.Decode(img)
	if err != nil {
		fatal(err)
	}

	if output != "" {
		if err := os.WriteFile(output, data, 0o644); err != nil {
			fatal(err)
		}
	} else {
		os.Stdout.Write(data)
		fmt.Println()
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "jabcodeReader:", err)
	os.Exit(1)
}
