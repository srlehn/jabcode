//go:build (jabcode_bsi || jabcode_legacy) && jabcode_ldpc_catalog_blob

package ldpccatalog

import _ "embed"

// The C-family permutation only ever builds codes for the pre-ISO wire
// families, so its catalog is embedded exactly where those families are
// compiled. A build without them never selects this generator.
//
//go:embed lcg.bin
var lcgCatalog []byte
