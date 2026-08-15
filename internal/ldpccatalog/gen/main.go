// Command gen writes the precomputed pivot transcripts package ldpccatalog
// embeds. It is invoked through the go:generate directive beside the artifacts
// and should otherwise be left alone: what it produces is checked in, and a
// rebuild is only interesting when the Gallager construction or the sweep
// changes, in which case a byte difference is the signal that it did.
//
// The generation itself lives in ldpccatalog.Generate, so this command and a
// build that computes its catalog at startup cannot drift apart.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/srlehn/jabcode/internal/ldpccatalog"
)

func main() {
	dir := flag.String("out", ".", "directory to write the catalog artifacts into")
	flag.Parse()

	for _, g := range []struct {
		generator ldpccatalog.Generator
		name      string
	}{
		{ldpccatalog.GeneratorISO, "iso.bin"},
		{ldpccatalog.GeneratorLCG, "lcg.bin"},
	} {
		blob, rows, err := ldpccatalog.Generate(g.generator)
		if err != nil {
			fmt.Fprintln(os.Stderr, "jabcode: "+err.Error())
			os.Exit(1)
		}
		at := filepath.Join(*dir, g.name)
		if err := os.WriteFile(at, blob, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "jabcode: "+err.Error())
			os.Exit(1)
		}
		fmt.Printf("%s: %d slots, %d stored pivots, %d bytes\n",
			g.name, ldpccatalog.SlotCount(), rows, len(blob))
	}
}
