//go:build !jabcode_ldpc_catalog_runtime

package ldpccatalog

func catalogBytes(g Generator) []byte {
	if g == GeneratorISO {
		return isoCatalog
	}
	return lcgCatalog
}
