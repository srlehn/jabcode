//go:build jabcode_bsi || jabcode_legacy

package ldpccatalog

// lcgSelectable reports whether this build compiles a wire variant that reads
// the C-family catalog. It is the same condition the embedded artifact is
// tagged with, named so the runtime build can apply it too: computing a catalog
// no variant can select costs about a minute and produces nothing.
const lcgSelectable = true
