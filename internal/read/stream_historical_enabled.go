//go:build jabcode_bsi || jabcode_legacy

package read

import "github.com/srlehn/jabcode/internal/core"

type optionalStreamObservation struct {
	correction   primaryCorrection
	seedAdmitted bool
}

func setHistoricalStreamObservation(observed *streamObservation, channels [3]*core.Bitmap,
	symbols []core.DecodedSymbol, correction primaryCorrection, route streamRoute, seedAdmitted bool) bool {
	*observed = streamObservation{
		optionalStreamObservation: optionalStreamObservation{
			correction: correction, seedAdmitted: seedAdmitted,
		},
		channels: channels, symbols: symbols, route: route,
	}
	return true
}

func historicalSeedAdmitted(observed *streamObservation) bool {
	return observed.seedAdmitted
}

func (s *Stream) finishHistoricalObservation(bm *core.Bitmap, chFn func() [3]*core.Bitmap,
	observed *streamObservation) ([]byte, bool) {
	if s.work.correctionChains >= 1 || observed.correction == nil {
		return nil, false
	}
	s.work.correctionChains++
	if observed.correction.CorrectPayload() != core.Success {
		return nil, false
	}
	var ch [3]*core.Bitmap
	if observed.symbols[0].Meta.DockedPosition != 0 {
		ch = chFn()
	}
	data, ok := decodeSymbols(bm, ch, observed.symbols, 1)
	if !ok {
		return nil, false
	}
	s.group = evidenceGroup{}
	return data, true
}
