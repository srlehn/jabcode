//go:build !jabcode_bsi && !jabcode_legacy

package detect

// An untagged build compiles the BSI chains out entirely; the empty sources are
// never compiled because bsiFamilyFinderEnabled gates every use.
var (
	finderChainBSIWGSL            string
	finderChainDirectionalBSIWGSL string
)
