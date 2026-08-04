//go:build (jabcode_bsi || jabcode_legacy) && !js

package detect

import (
	"fmt"

	"github.com/srlehn/vulki"
)

// bindDirectionalBSIChain binds the BSI-era directional chain over the state
// the current family's chain already owns. The two never sweep at the same
// time, so sharing the records, outcomes, summary and dispatch arguments costs
// nothing and keeps one overflow contract rather than two.
func (b *gpuBinarizer) bindDirectionalBSIChain(
	outcomes, params, summary, args, colorSource *vulki.Buffer,
) (*vulki.BindingSet, error) {
	kernel, err := b.kernels.finderChainDirectionalBSI()
	if err != nil {
		return nil, err
	}
	bindings, err := kernel.NewBindings(
		vulki.BindBuffer(0, b.packedMasks),
		vulki.BindBuffer(1, b.dirRecords),
		vulki.BindBuffer(2, outcomes),
		vulki.BindBuffer(3, params),
		vulki.BindBuffer(4, colorSource),
		vulki.BindBuffer(5, summary),
		vulki.BindBuffer(6, args),
	)
	if err != nil {
		return nil, fmt.Errorf("jabcode: bind GPU BSI directional chain: %w", err)
	}
	return bindings, nil
}

// directionalChainFor selects the chain that seeks on the given channel. A
// channel no compiled family seeks on returns nil bindings and the caller
// reports it rather than sweeping with the wrong signature's kernel.
func (b *gpuBinarizer) directionalChainFor(channel int) (*vulki.Kernel, *vulki.BindingSet) {
	if channel == bsiFamilySeekChannel {
		kernel, err := b.kernels.finderChainDirectionalBSI()
		if err != nil {
			return nil, nil
		}
		return kernel, b.dirChainBSIBindings
	}
	kernel, err := b.kernels.finderChainDirectional()
	if err != nil {
		return nil, nil
	}
	return kernel, b.dirChainBindings
}
