//go:build !jabcode_bsi

package read

import "github.com/srlehn/jabcode/internal/core"

const bsiReadEnabled = false

func decodeBSISampled(_ *core.Bitmap, _ core.DecodedSymbol) ([]byte, bool) {
	return nil, false
}
