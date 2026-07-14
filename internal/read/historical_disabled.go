//go:build !jabcode_bsi && !jabcode_legacy

package read

import (
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/wire"
)

func decodeHistoricalLocated(*detect.PrimaryDetector, *finding, *DiagnosticAttempt, wire.Capabilities) ([]byte, readStage, bool) {
	return nil, readNoFinders, false
}
