//go:build !jabcode_bsi

package read

import (
	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
)

func decodeBSIDockedSecondary(*core.Bitmap, [3]*core.Bitmap, *core.DecodedSymbol, *core.DecodedSymbol, int, *decode.ModuleClassificationTrace) (*core.Bitmap, *core.Bitmap, int) {
	return nil, nil, core.Failure
}
