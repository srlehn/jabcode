//go:build jabcode_high_color && jabcode_legacy

package read

import "github.com/srlehn/jabcode/internal/wire"

const currentFamilyCapabilities = wire.Capabilities(1<<wire.ISO23634 | 1<<wire.ISOHighColor | 1<<wire.CurrentC)
const shareCurrentFamilyEvidence = true

func currentObservationVariants(capabilities wire.Capabilities) ([2]wire.Variant, int) {
	var variants [2]wire.Variant
	n := 0
	if capabilities.Has(wire.ISOHighColor) {
		variants[n] = wire.ISOHighColor
		n++
	} else if capabilities.Has(wire.ISO23634) {
		variants[n] = wire.ISO23634
		n++
	}
	if capabilities.Has(wire.CurrentC) {
		variants[n] = wire.CurrentC
		n++
	}
	return variants, n
}
