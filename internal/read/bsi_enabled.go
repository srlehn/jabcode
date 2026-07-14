//go:build jabcode_bsi

package read

import (
	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/wire"
)

const bsiReadEnabled = true

func decodeBSISampled(bm, matrix *core.Bitmap, base core.DecodedSymbol, detail *DiagnosticAttempt) ([]byte, bool) {
	symbols := make([]core.DecodedSymbol, maxSymbolNumber)
	symbols[0] = base
	symbols[0].WireVariant = wire.BSI
	if decode.DecodeBSIPrimary(matrix, &symbols[0]) != core.Success {
		return nil, false
	}
	return decodeSymbolsTraced(bm, [3]*core.Bitmap{}, symbols, 1, detail)
}
