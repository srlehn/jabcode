//go:build !jabcode_bsi

package encode

import "errors"

func (e *encoder) generateBSI(_ []byte) error {
	return errors.New("jabcode: BSI profile was not compiled into this build")
}
