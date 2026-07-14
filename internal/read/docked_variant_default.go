//go:build !jabcode_bsi && !jabcode_legacy

package read

import (
	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
)

func decodeVariantDockedSecondary(bm *core.Bitmap, ch [3]*core.Bitmap, host, secondary *core.DecodedSymbol, dockedPosition int, trace *decode.ModuleClassificationTrace) (*core.Bitmap, *core.Bitmap, int) {
	matrix, result := decodeCurrentDockedSecondary(bm, ch, host, secondary, dockedPosition, trace)
	return matrix, nil, result
}
