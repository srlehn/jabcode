//go:build jabcode_ldpc_catalog_blob

package ldpccatalog

func catalogBytes(g Generator) []byte {
	if g == GeneratorISO {
		return isoCatalog
	}
	return lcgCatalog
}
