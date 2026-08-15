//go:build jabcode_ldpc_catalog_runtime

package ldpccatalog

import "sync"

// This build carries no catalog artifact and computes each one on first use
// instead, trading roughly 13 MB of embedded data per generator for minutes of
// startup elimination. Generate produces the same bytes the artifact holds, so
// nothing downstream can tell the two builds apart.
//
// Laziness is also what keeps the C-family catalog out of a build that cannot
// select it: only a non-ISO wire variant ever asks for it, and those exist only
// where the pre-ISO families are compiled. No build tag is needed to express
// that, because nothing requests what it cannot reach.
var (
	catalogOnce [2]sync.Once
	catalogs    [2][]byte
)

func catalogBytes(g Generator) []byte {
	if g != GeneratorISO && g != GeneratorLCG {
		return nil
	}
	catalogOnce[g].Do(func() {
		if blob, _, err := Generate(g); err == nil {
			catalogs[g] = blob
		}
	})
	return catalogs[g]
}
