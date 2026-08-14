package ldpccatalog

import _ "embed"

//go:generate go run ./gen -out .

//go:embed iso.bin
var isoCatalog []byte
