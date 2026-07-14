//go:build !jabcode_bsi && !jabcode_legacy

package read

import (
	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/wire"
)

func decodeHistoricalLocated(*detect.PrimaryDetector, *finding, *DiagnosticAttempt, wire.Capabilities) ([]byte, readStage, bool) {
	return nil, readNoFinders, false
}

func decodeHistoricalSampled(*core.Bitmap, *core.Bitmap, core.DecodedSymbol, *DiagnosticAttempt, wire.Capabilities, func() ([3]*core.Bitmap, bool)) ([]byte, bool) {
	return nil, false
}

func historicalObservationVariants(wire.Capabilities) ([2]wire.Variant, int) {
	return [2]wire.Variant{}, 0
}

func observeHistoricalStreamSampled(*core.Bitmap, core.DecodedSymbol, wire.Variant) ([]core.DecodedSymbol, primaryCorrection, bool, bool) {
	return nil, nil, false, false
}
