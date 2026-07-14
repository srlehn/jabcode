//go:build !jabcode_bsi

package read

import "github.com/srlehn/jabcode/internal/core"

const bsiReadEnabled = false

func decodeBSISampled(*core.Bitmap, core.DecodedSymbol) ([]byte, bool) {
	return nil, false
}
