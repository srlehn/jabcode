// Command jabcodeWriter encodes text into a JAB Code PNG image.
package main

import (
	"flag"
	"fmt"
	"image"
	"image/png"
	"os"
	"strconv"
	"strings"

	"github.com/srlehn/jabcode"
)

func main() {
	input := flag.String("input", "", "input text to encode")
	output := flag.String("output", "", "output PNG file path")
	colors := flag.Int("colors", 8, "number of module colors (4, 8, 16, 32 or 64; above 8 is non-interoperable)")
	moduleSize := flag.Int("module-size", 12, "module size in pixels")
	eccLevel := flag.Int("ecc-level", 0, "error correction level (0 = default)")
	positions := flag.String("symbol-positions", "", "multi-symbol positions, comma-separated (e.g. 0,2)")
	versions := flag.String("symbol-versions", "", "multi-symbol side-versions, comma-separated WxH (e.g. 4x4,4x4)")
	eccLevels := flag.String("symbol-ecc", "", "multi-symbol ECC levels, comma-separated (e.g. 0,0)")
	flag.Parse()

	if *input == "" || *output == "" {
		fmt.Fprintln(os.Stderr, "usage: jabcodeWriter --input TEXT --output FILE.png [--colors N] [--module-size PX] [--ecc-level L]")
		fmt.Fprintln(os.Stderr, "  multi-symbol: --symbol-positions 0,2 --symbol-versions 4x4,4x4 --symbol-ecc 0,0")
		os.Exit(2)
	}

	if *colors > 8 {
		fmt.Fprintf(os.Stderr, "warning: %d-color symbols are non-interoperable; only this library reads them, no other JAB Code decoder does. Use 4 or 8 for portable codes.\n", *colors)
	}

	opts := []jabcode.Option{
		jabcode.WithColors(*colors),
		jabcode.WithModuleSize(*moduleSize),
		jabcode.WithECCLevel(*eccLevel),
	}
	if *positions != "" {
		pos, vers, ecc, err := parseSymbols(*positions, *versions, *eccLevels)
		if err != nil {
			fmt.Fprintln(os.Stderr, "symbols:", err)
			os.Exit(2)
		}
		opts = append(opts, jabcode.WithSymbols(pos, vers, ecc))
	}

	img, err := jabcode.NewEncoder(opts...).Encode([]byte(*input))
	if err != nil {
		fmt.Fprintln(os.Stderr, "encode:", err)
		os.Exit(1)
	}

	f, err := os.Create(*output)
	if err != nil {
		fmt.Fprintln(os.Stderr, "create:", err)
		os.Exit(1)
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		fmt.Fprintln(os.Stderr, "encode png:", err)
		os.Exit(1)
	}
}

// parseSymbols parses the comma-separated multi-symbol flags.
func parseSymbols(positions, versions, eccLevels string) ([]int, []image.Point, []int, error) {
	pos, err := parseInts(positions)
	if err != nil {
		return nil, nil, nil, err
	}
	vers := make([]image.Point, len(pos))
	for i, s := range strings.Split(versions, ",") {
		wh := strings.SplitN(s, "x", 2)
		if len(wh) != 2 {
			return nil, nil, nil, fmt.Errorf("invalid version %q", s)
		}
		w, _ := strconv.Atoi(wh[0])
		h, _ := strconv.Atoi(wh[1])
		if i < len(vers) {
			vers[i] = image.Pt(w, h)
		}
	}
	ecc := make([]int, len(pos))
	if eccLevels != "" {
		parsed, err := parseInts(eccLevels)
		if err != nil {
			return nil, nil, nil, err
		}
		copy(ecc, parsed)
	}
	return pos, vers, ecc, nil
}

func parseInts(s string) ([]int, error) {
	parts := strings.Split(s, ",")
	out := make([]int, len(parts))
	for i, p := range parts {
		v, err := strconv.Atoi(strings.TrimSpace(p))
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}
