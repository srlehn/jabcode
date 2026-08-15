//go:build jabcode_ldpc_catalog_runtime

package ldpccatalog

import "sync"

// This build carries no catalog artifact and computes each one on first use
// instead, trading roughly 13 MB of embedded data per generator for minutes of
// startup elimination. Generate produces the same bytes the artifact holds, so
// nothing downstream can tell the two builds apart.
//
// Laziness alone does not keep the C-family catalog out of a build that cannot
// select it. Only a non-ISO wire variant ever asks for that catalog through a
// decode, but Combined assembles both generators for the device upload and asks
// unconditionally, so an ISO-only build was computing a catalog it can never
// read - about a minute of startup for nothing. The compiled families decide
// it instead.
var (
	catalogOnce [2]sync.Once
	catalogs    [2][]byte
)

func catalogBytes(g Generator) []byte {
	if g != GeneratorISO && g != GeneratorLCG {
		return nil
	}
	if g == GeneratorLCG && !lcgSelectable {
		return nil
	}
	catalogOnce[g].Do(func() {
		if blob, _, err := Generate(g); err == nil {
			catalogs[g] = blob
		}
	})
	return catalogs[g]
}
