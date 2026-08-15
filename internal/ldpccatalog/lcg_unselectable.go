//go:build !jabcode_bsi && !jabcode_legacy

package ldpccatalog

// lcgSelectable is false where no compiled wire variant reads the C-family
// catalog. See its counterpart for why the runtime build needs to know.
const lcgSelectable = false
