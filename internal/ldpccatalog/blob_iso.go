//go:build jabcode_ldpc_catalog_blob

package ldpccatalog

import _ "embed"

//go:embed iso.bin
var isoCatalog []byte
