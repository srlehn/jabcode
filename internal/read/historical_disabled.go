//go:build !jabcode_bsi && !jabcode_legacy

package read

import (
	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/wire"
)

func decodeHistoricalBitmap(_ *core.Bitmap, _ [3]*core.Bitmap, _ func() bool, _ *finding, _ *DiagnosticAttempt, _ wire.Capabilities) ([]byte, readStage, bool) {
	return nil, readNoFinders, false
}
