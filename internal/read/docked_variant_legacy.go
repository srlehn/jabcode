//go:build jabcode_legacy

package read

import (
	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/wire"
)

func decodeVariantDockedSecondary(bm *core.Bitmap, ch [3]*core.Bitmap, host, secondary *core.DecodedSymbol, dockedPosition int, trace *decode.ModuleClassificationTrace) (*core.Bitmap, *core.Bitmap, int) {
	switch secondary.WireVariant {
	case wire.PreV2C:
		matrix := detect.DetectPreV2CSecondary(bm, ch, host, secondary, dockedPosition)
		if matrix == nil {
			return nil, nil, core.Failure
		}
		if trace != nil {
			return matrix, nil, decode.DecodePreV2CSecondaryTraced(matrix, secondary, trace)
		}
		return matrix, nil, decode.DecodePreV2CSecondary(matrix, secondary)
	case wire.BSI:
		return decodeBSIDockedSecondary(bm, ch, host, secondary, dockedPosition, trace)
	default:
		matrix, result := decodeCurrentDockedSecondary(bm, ch, host, secondary, dockedPosition, trace)
		return matrix, nil, result
	}
}
