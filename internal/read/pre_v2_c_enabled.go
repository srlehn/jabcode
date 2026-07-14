//go:build jabcode_legacy

package read

import (
	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/wire"
)

const (
	currentCReadEnabled = true
	preV2CReadEnabled   = true
)

func decodePreV2CSampled(bm *core.Bitmap, ch [3]*core.Bitmap, matrix *core.Bitmap, base core.DecodedSymbol, detail *DiagnosticAttempt) ([]byte, bool) {
	symbols := make([]core.DecodedSymbol, maxSymbolNumber)
	symbols[0] = base
	symbols[0].WireVariant = wire.PreV2C
	if decode.DecodePreV2CPrimary(matrix, &symbols[0]) != core.Success {
		return nil, false
	}
	return decodeSymbolsTraced(bm, ch, symbols, 1, detail)
}
