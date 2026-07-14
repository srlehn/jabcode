//go:build jabcode_bsi

package read

import (
	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/wire"
)

const bsiReadEnabled = true

func decodeBSISampled(matrix *core.Bitmap, base core.DecodedSymbol) ([]byte, bool) {
	symbol := base
	symbol.WireProfile = wire.BSI
	if decode.DecodeBSIPrimary(matrix, &symbol) != core.Success {
		return nil, false
	}
	if symbol.Meta.DockedPosition != 0 {
		return nil, false
	}
	data, ok := decode.DecodeDataProfile(symbol.Data, wire.BSI)
	return data, ok
}
