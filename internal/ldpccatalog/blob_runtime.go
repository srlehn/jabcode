//go:build !jabcode_ldpc_catalog_blob

package ldpccatalog

import "sync"

// An ordinary build carries no catalog artifact and computes each one on first
// use instead, because roughly 13 MB of embedded data per generator is ballast
// every dependant would link whether or not it ever runs a device decode. The
// jabcode_ldpc_catalog_blob build trades that back, paying the size to skip the
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
