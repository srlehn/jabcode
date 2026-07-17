//go:build jabcode_bsi || jabcode_legacy

package detect

import _ "embed"

// The BSI chain fragment is embedded only in builds whose decoder compiles
// the BSI family in, so an untagged binary carries no trace of the BSI chain
// and its kernel module never reaches a driver's pipeline compiler.
//
//go:embed shaders/finder_chain_bsi.wgsl
var finderChainBSIWGSL string
