//go:build !jabcode_high_color && !jabcode_legacy

package read

import "github.com/srlehn/jabcode/internal/wire"

const currentFamilyCapabilities = wire.Capabilities(1 << wire.ISO23634)
const shareCurrentFamilyEvidence = false

func currentObservationVariants(capabilities wire.Capabilities) ([2]wire.Variant, int) {
	var variants [2]wire.Variant
	if capabilities.Has(wire.ISO23634) {
		variants[0] = wire.ISO23634
		return variants, 1
	}
	return variants, 0
}
