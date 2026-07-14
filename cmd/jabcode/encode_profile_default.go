//go:build !jabcode_non_iso_encode

package main

import (
	"fmt"
	"io"

	"github.com/spf13/pflag"

	"github.com/srlehn/jabcode"
)

func addEncodeProfileFlag(*pflag.FlagSet, *string) {}

func encodeProfileOption(string) (jabcode.Option, string, error) {
	return nil, "", nil
}

func encodeColorsUsage(w io.Writer) {
	fmt.Fprintln(w, "  -c, --colors n            module colors: 4 or 8")
}

func encodeProfileUsage(io.Writer) {}
