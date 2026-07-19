//go:build !jabcode_bsi && !jabcode_legacy

package read

import "github.com/srlehn/jabcode/internal/core"

type optionalStreamObservation struct{}

func setHistoricalStreamObservation(*streamObservation, [3]*core.Bitmap, []core.DecodedSymbol,
	primaryCorrection, streamRoute, bool) bool {
	return false
}

func historicalSeedAdmitted(*streamObservation) bool { return false }

func (*Stream) finishHistoricalObservation(*core.Bitmap, func() [3]*core.Bitmap, *streamObservation) (*Message, bool) {
	return nil, false
}
