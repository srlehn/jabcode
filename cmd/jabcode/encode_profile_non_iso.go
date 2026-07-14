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
	fs.StringVar(value, "profile", "iso", "encoder profile: iso, hc or bsi")
}

func encodeProfileOption(value string) (jabcode.Option, string, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "iso", "iso-23634", "iso23634":
		return jabcode.WithProfile(jabcode.ProfileISO23634), "iso", nil
	case "hc", "high-color", "high_color":
		return jabcode.WithProfile(jabcode.ProfileHighColor), "hc", nil
	case "bsi":
		return jabcode.WithProfile(jabcode.ProfileBSI), "bsi", nil
	default:
		return nil, "", fmt.Errorf("invalid encoder profile %q (want iso, hc or bsi)", value)
	}
}

func encodeColorsUsage(w io.Writer) {
	fmt.Fprintln(w, "  -c, --colors n            module colors: 4, 8, 16, 32, 64, 128, 256")
}

func encodeProfileUsage(w io.Writer) {
	fmt.Fprintln(w, "      --profile mode        encoder profile: iso (default, experimental), hc or bsi")
}
