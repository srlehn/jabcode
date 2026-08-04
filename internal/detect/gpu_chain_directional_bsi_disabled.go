//go:build !jabcode_bsi && !jabcode_legacy && !js

package detect

import "github.com/srlehn/vulki"

// An untagged build has no BSI-era family to sweep, so there is nothing to bind
// and the current family's chain serves every channel that reaches here.
func (b *gpuBinarizer) bindDirectionalBSIChain(
	_, _, _, _, _ *vulki.Buffer,
) (*vulki.BindingSet, error) {
	return nil, nil
}

func (b *gpuBinarizer) directionalChainFor(int) (*vulki.Kernel, *vulki.BindingSet) {
	kernel, err := b.kernels.finderChainDirectional()
	if err != nil {
		return nil, nil
	}
	return kernel, b.dirChainBindings
}
