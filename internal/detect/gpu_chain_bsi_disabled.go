//go:build !jabcode_bsi && !jabcode_legacy

package detect

// An untagged build compiles the BSI chain out entirely; the empty source is
// never compiled because bsiFamilyFinderEnabled gates every use.
var finderChainBSIWGSL string
