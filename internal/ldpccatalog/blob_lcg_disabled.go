//go:build !jabcode_bsi && !jabcode_legacy

package ldpccatalog

// Without the pre-ISO wire families there is no variant that reads the C-family
// catalog, so it stays out of the binary.
var lcgCatalog []byte
