//go:build !jabcode_high_color && !jabcode_legacy

package read

import "github.com/srlehn/jabcode/internal/wire"

const currentFamilyCapabilities = wire.Capabilities(1 << wire.ISO23634)

var currentFamilyVariants = [...]wire.Variant{wire.ISO23634}
