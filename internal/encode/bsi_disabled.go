//go:build !jabcode_non_iso_encode

package encode

import "errors"

func (*encoder) generateBSI([]byte) error {
	return errors.New("jabcode: BSI encoder was not compiled into this build")
}
