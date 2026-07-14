//go:build jabcode_bsi

package read

import (
	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/detect"
)

func decodeBSIDockedSecondary(bm *core.Bitmap, ch [3]*core.Bitmap, host, secondary *core.DecodedSymbol, dockedPosition int, trace *decode.ModuleClassificationTrace) (*core.Bitmap, *core.Bitmap, int) {
	seed, metadata := detect.PrepareBSISecondary(bm, ch, host, secondary, dockedPosition)
	if metadata == nil || decode.DecodeBSISecondaryMetadata(metadata, host, secondary) != core.Success {
		return nil, metadata, core.Failure
	}
	matrix := detect.FinishBSISecondary(bm, ch, secondary, seed)
	if matrix == nil {
		return nil, metadata, core.Failure
	}
	if trace != nil {
		return matrix, metadata, decode.DecodeBSISecondaryTraced(matrix, secondary, trace)
	}
	return matrix, metadata, decode.DecodeBSISecondary(matrix, secondary)
}
