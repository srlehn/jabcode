// Command jabcode encodes and decodes JAB Code symbols.
package main

import (
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/spf13/pflag"
	"github.com/srlehn/jabcode"
	"github.com/srlehn/jabcode/internal/diag"
	"github.com/srlehn/jabcode/internal/wire"

	_ "github.com/gen2brain/avif"
	_ "github.com/gen2brain/heic"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/vp8"
	_ "golang.org/x/image/vp8l"
	_ "golang.org/x/image/webp"
)

type usageError string

func (e usageError) Error() string { return string(e) }

const encodeInputFlagName = "input"

func main() {
	if err := run(os.Args[1:]); err != nil {
		if _, ok := err.(usageError); ok {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		fmt.Fprintln(os.Stderr, "jabcode:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		rootUsage(os.Stderr)
		return usageError("missing command")
	}
	switch args[0] {
	case "-h", "--help", "help":
		rootUsage(os.Stdout)
		return nil
	case "encode":
		return runEncode(args[1:])
	case "decode":
		return runDecode(args[1:])
	default:
		rootUsage(os.Stderr)
		return usageError(fmt.Sprintf("unknown command %q", args[0]))
	}
}

func rootUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: jabcode <command> [flags]")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "commands:")
	fmt.Fprintln(w, "  encode   encode stdin or literal text to a PNG JAB Code")
	fmt.Fprintln(w, "  decode   decode an image to bytes")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "run \"jabcode <command> --help\" for command flags")
}

func runEncode(args []string) error {
	var literal string
	var output string
	var colors int
	var moduleSize int
	var eccLevel int
	var symbols string
	var profileName string

	fs := pflag.NewFlagSet("encode", pflag.ContinueOnError)
	fs.SetOutput(io.Discard)
	fs.StringVarP(&literal, encodeInputFlagName, "i", "", "literal input text instead of stdin")
	fs.StringVarP(&output, "output", "o", "", "output PNG file, or stdout when empty or -")
	fs.IntVarP(&colors, "colors", "c", 8, "number of module colors")
	fs.IntVarP(&moduleSize, "module-size", "m", 12, "module size in pixels")
	fs.IntVarP(&eccLevel, "ecc-level", "e", 0, "error correction level, 0 selects the default")
	fs.StringVarP(&symbols, "symbols", "s", "", "multi-symbol spec: pos:WxH:ecc[,pos:WxH:ecc...]")
	fs.StringVar(&profileName, "profile", "iso", "wire profile: iso, hc, bsi, or legacy")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, pflag.ErrHelp) {
			encodeUsage(os.Stdout)
			return nil
		}
		encodeUsage(os.Stderr)
		return usageError(fmt.Sprintf("encode: %v", err))
	}
	if fs.NArg() != 0 {
		encodeUsage(os.Stderr)
		return usageError("encode takes no positional arguments")
	}

	data, err := encodeInput(literal, fs.Changed(encodeInputFlagName))
	if err != nil {
		return err
	}
	profile, _, err := parseProfile(profileName)
	if err != nil {
		return usageError(fmt.Sprintf("encode: %v", err))
	}
	opts := []jabcode.Option{
		jabcode.WithColors(colors),
		jabcode.WithModuleSize(moduleSize),
		jabcode.WithECCLevel(eccLevel),
		jabcode.WithProfile(profile),
	}
	if symbols != "" {
		pos, vers, ecc, err := parseSymbols(symbols)
		if err != nil {
			return usageError(fmt.Sprintf("encode: %v", err))
		}
		opts = append(opts, jabcode.WithSymbols(pos, vers, ecc))
	}
	if colors > 8 && profile == jabcode.ProfileHighColor {
		fmt.Fprintf(os.Stderr, "warning: %d-color symbols use the non-standard high-color extension; use 4 or 8 for ISO/IEC 23634 interoperability.\n", colors)
	}
	img, err := jabcode.NewEncoder(opts...).Encode(data)
	if err != nil {
		return fmt.Errorf("encode: %w", err)
	}
	return writePNG(output, img)
}

func encodeUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: jabcode encode [flags]")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "reads stdin or -i text; writes PNG to stdout or -o file")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "examples:")
	fmt.Fprintln(w, "  printf hello | jabcode encode -o hello.png")
	fmt.Fprintln(w, "  jabcode encode -i hello -o hello.png")
	fmt.Fprintln(w, "  jabcode encode -s 0:4x4:0,2:4x4:0 -o cascade.png < payload.bin")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "flags:")
	fmt.Fprintln(w, "  -i, --input string        literal input text; omit it to read stdin")
	fmt.Fprintln(w, "  -o, --output file         output PNG file, or - for stdout")
	fmt.Fprintln(w, "  -c, --colors n            module colors: 4, 8, 16, 32, 64, 128, 256")
	fmt.Fprintln(w, "  -m, --module-size px      module size in pixels, default 12")
	fmt.Fprintln(w, "  -e, --ecc-level n         error correction level, 0 selects the default")
	fmt.Fprintln(w, "  -s, --symbols spec        pos:WxH:ecc[,pos:WxH:ecc...]")
	fmt.Fprintln(w, "      --profile mode        wire profile: iso (default, experimental), hc, bsi, or legacy")
	fmt.Fprintln(w, "  -h, --help                show help")
}

func encodeInput(literal string, hasLiteral bool) ([]byte, error) {
	if hasLiteral {
		return []byte(literal), nil
	}
	data, err := io.ReadAll(os.Stdin)
	if err != nil {
		return nil, fmt.Errorf("read stdin: %w", err)
	}
	return data, nil
}

func writePNG(path string, img image.Image) error {
	if path == "" || path == "-" {
		if err := png.Encode(os.Stdout, img); err != nil {
			return fmt.Errorf("write png: %w", err)
		}
		return nil
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create %s: %w", path, err)
	}
	return writePNGFile(f, img, path)
}

func writePNGFile(w io.WriteCloser, img image.Image, path string) error {
	if err := png.Encode(w, img); err != nil {
		_ = w.Close()
		return fmt.Errorf("write png %s: %w", path, err)
	}
	if err := w.Close(); err != nil {
		return fmt.Errorf("close png %s: %w", path, err)
	}
	return nil
}

func parseSymbols(spec string) ([]int, []image.Point, []int, error) {
	entries := strings.Split(spec, ",")
	pos := make([]int, len(entries))
	vers := make([]image.Point, len(entries))
	ecc := make([]int, len(entries))
	for i, entry := range entries {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			return nil, nil, nil, fmt.Errorf("empty symbol entry")
		}
		fields := strings.Split(entry, ":")
		if len(fields) != 3 {
			return nil, nil, nil, fmt.Errorf("invalid symbol %q, want pos:WxH:ecc", entry)
		}
		var err error
		pos[i], err = parseIntField(fields[0], "position")
		if err != nil {
			return nil, nil, nil, err
		}
		wh := strings.SplitN(strings.ToLower(strings.TrimSpace(fields[1])), "x", 2)
		if len(wh) != 2 {
			return nil, nil, nil, fmt.Errorf("invalid version %q, want WxH", fields[1])
		}
		w, err := parseIntField(wh[0], "version width")
		if err != nil {
			return nil, nil, nil, err
		}
		h, err := parseIntField(wh[1], "version height")
		if err != nil {
			return nil, nil, nil, err
		}
		vers[i] = image.Pt(w, h)
		ecc[i], err = parseIntField(fields[2], "ecc")
		if err != nil {
			return nil, nil, nil, err
		}
	}
	return pos, vers, ecc, nil
}

func parseIntField(s, name string) (int, error) {
	v, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil {
		return 0, fmt.Errorf("invalid %s %q", name, s)
	}
	return v, nil
}

func runDecode(args []string) error {
	var output string
	var wantDiag bool
	var diagOut string
	var profileName string

	fs := pflag.NewFlagSet("decode", pflag.ContinueOnError)
	fs.SetOutput(io.Discard)
	fs.StringVarP(&output, "output", "o", "", "output payload file, or stdout when empty or -")
	fs.BoolVarP(&wantDiag, "diag", "d", false, "write diagnostics to stderr")
	fs.StringVarP(&diagOut, "diag-out", "D", "", "diagnostic image output directory, implies --diag")
	fs.StringVar(&profileName, "profile", "iso", "wire profile: iso, hc, bsi, or legacy")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, pflag.ErrHelp) {
			decodeUsage(os.Stdout)
			return nil
		}
		decodeUsage(os.Stderr)
		return usageError(fmt.Sprintf("decode: %v", err))
	}
	if fs.NArg() != 1 {
		decodeUsage(os.Stderr)
		return usageError("decode needs exactly one image path")
	}
	if diagOut != "" {
		wantDiag = true
	}
	explicitProfile := fs.Changed("profile")
	var selected jabcode.Profile
	var profile wire.Profile
	var err error
	if explicitProfile {
		selected, profile, err = parseProfile(profileName)
		if err != nil {
			return usageError(fmt.Sprintf("decode: %v", err))
		}
		if !selected.Available() {
			return fmt.Errorf("decode: %w: %s", jabcode.ErrProfileUnavailable, selected)
		}
	}

	img, err := readImage(fs.Arg(0))
	if err != nil {
		return err
	}
	var data []byte
	if wantDiag {
		if explicitProfile {
			data, err = diag.DiagnoseProfile(img, os.Stderr, diagOut, fs.Arg(0), profile)
		} else {
			data, err = diag.Diagnose(img, os.Stderr, diagOut, fs.Arg(0))
		}
	} else if explicitProfile {
		data, err = jabcode.DecodeWithProfile(img, selected)
	} else {
		data, err = jabcode.Decode(img)
	}
	if err != nil {
		return fmt.Errorf("decode: %w", err)
	}
	return writePayload(output, data)
}

func decodeUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: jabcode decode [flags] IMAGE")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "writes payload to stdout or -o file")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "examples:")
	fmt.Fprintln(w, "  jabcode decode code.png")
	fmt.Fprintln(w, "  jabcode decode -o payload.bin code.png")
	fmt.Fprintln(w, "  jabcode decode -d -D ./diag-images code.png")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "flags:")
	fmt.Fprintln(w, "  -o, --output file       output payload file, or - for stdout")
	fmt.Fprintln(w, "  -d, --diag              write diagnostics to stderr")
	fmt.Fprintln(w, "  -D, --diag-out dir      write diagnostic images, implies --diag")
	fmt.Fprintln(w, "      --profile mode      force one profile: iso (experimental), hc, bsi, or legacy")
	fmt.Fprintln(w, "                            default: try every compiled decoder")
	fmt.Fprintln(w, "  -h, --help              show help")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "image formats: PNG, JPEG, HEIC, AVIF, TIFF, WebP VP8 and WebP VP8L")
}

func parseProfile(value string) (jabcode.Profile, wire.Profile, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "iso", "iso-23634", "iso23634":
		return jabcode.ProfileISO23634, wire.ISO23634, nil
	case "hc", "high-color", "high_color":
		return jabcode.ProfileHighColor, wire.HighColor, nil
	case "bsi":
		return jabcode.ProfileBSI, wire.BSI, nil
	case "legacy", "c", "c-reference", "compat":
		return jabcode.ProfileLegacy, wire.Legacy, nil
	default:
		return 0, 0, fmt.Errorf("invalid profile %q (want iso, hc, bsi, or legacy)", value)
	}
}

func readImage(path string) (image.Image, error) {
	if path == "-" {
		img, _, err := image.Decode(os.Stdin)
		if err != nil {
			return nil, fmt.Errorf("decode image stdin: %w", err)
		}
		return img, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("decode image %s: %w", path, err)
	}
	return img, nil
}

func writePayload(path string, data []byte) error {
	if path == "" || path == "-" {
		_, err := os.Stdout.Write(data)
		return err
	}
	return os.WriteFile(path, data, 0o644)
}
