//go:build jabcode_non_iso_encode

package main

import (
	"fmt"
	"io"
	"strings"

	"github.com/spf13/pflag"

	"github.com/srlehn/jabcode"
)

func addEncodeProfileFlag(fs *pflag.FlagSet, value *string) {
	fs.StringVar(value, "profile", "iso", "encoder profile: iso or hc")
}

func encodeProfileOption(value string) (jabcode.Option, bool, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "iso", "iso-23634", "iso23634":
		return jabcode.WithProfile(jabcode.ProfileISO23634), false, nil
	case "hc", "high-color", "high_color":
		return jabcode.WithProfile(jabcode.ProfileHighColor), true, nil
	default:
		return nil, false, fmt.Errorf("invalid encoder profile %q (want iso or hc)", value)
	}
}

func encodeColorsUsage(w io.Writer) {
	fmt.Fprintln(w, "  -c, --colors n            module colors: 4, 8, 16, 32, 64, 128, 256")
}

func encodeProfileUsage(w io.Writer) {
	fmt.Fprintln(w, "      --profile mode        encoder profile: iso (default, experimental) or hc")
}
