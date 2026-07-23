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

func decodePreV2CSampled(bm, matrix *core.Bitmap, base core.DecodedSymbol, detail *DiagnosticAttempt,
	channels func() ([3]*core.Bitmap, bool)) (*Message, bool) {
	symbols, correction, ok, _ := observePreV2CStreamSampled(matrix, base)
	if !ok || correction.CorrectPayload() != core.Success {
		return nil, false
	}
	var ch [3]*core.Bitmap
	if symbols[0].Meta.DockedPosition != 0 {
		var ok bool
		ch, ok = channels()
		if !ok {
			return nil, false
		}
	}
	return decodeSymbolsTraced(bm, ch, symbols, 1, detail)
}

func observePreV2CStreamSampled(matrix *core.Bitmap, base core.DecodedSymbol) ([]core.DecodedSymbol, primaryCorrection, bool, bool) {
	symbols := make([]core.DecodedSymbol, maxSymbolNumber)
	symbols[0] = base
	symbols[0].WireVariant = wire.PreV2C
	observation, result := decode.ObservePreV2CPrimary(matrix, &symbols[0])
	if result != core.Success {
		return nil, nil, false, false
	}
	return symbols, observation, true, observation.ExplicitMetadata()
}
