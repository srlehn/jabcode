//go:build jabcode_bsi || jabcode_legacy

package read

import (
	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/wire"
)

// decodeHistoricalLocated samples the finder family shared by BSI TR-03137 and
// the pre-v2.0 C reference from the shared detector traversal once, then tries
// every enabled interpretation requested by capabilities.
func decodeHistoricalLocated(d *detect.PrimaryDetector, f *finding, detail *DiagnosticAttempt, capabilities wire.Capabilities) (*Message, readStage, bool) {
	if !d.SelectFinderFamily(detect.FinderFamilyBSI) {
		return nil, readNoFinders, finderEvidence(d)
	}
	base := core.DecodedSymbol{}
	matrix, stage := sampleLocatedPrimaryTraced(d, detect.FinderFamilyBSI, &base, f, detail)
	if stage != readSampled {
		return nil, stage, true
	}
	bm, ch := d.BM, d.Ch
	if detail != nil {
		detail.FinalChannels = d.Ch
		detail.Detector = d.Stats
		detail.Finders = append([]detect.FinderPattern(nil), d.FPs[:4]...)
	}
	data, ok := decodeHistoricalSampled(bm, matrix, base, detail, capabilities, func() ([3]*core.Bitmap, bool) {
		// Historical wire routes read mask pixels, which a GPU-located
		// detector defers until a consumer needs them.
		return ch, d.EnsureChannels()
	})
	if ok {
		if f != nil && f.located {
			f.payload = cloneMessage(data)
		}
		return data, readDecoded, true
	}
	return nil, readSampled, true
}

func decodeHistoricalSampled(bm, matrix *core.Bitmap, base core.DecodedSymbol, detail *DiagnosticAttempt, capabilities wire.Capabilities, channels func() ([3]*core.Bitmap, bool)) (*Message, bool) {
	if capabilities.Has(wire.BSI) {
		if data, ok := decodeBSISampled(bm, matrix, base, detail, channels); ok {
			return data, true
		}
	}
	if capabilities.Has(wire.PreV2C) {
		ch, ok := channels()
		if !ok {
			return nil, false
		}
		if data, ok := decodePreV2CSampled(bm, ch, matrix, base, detail); ok {
			return data, true
		}
	}
	return nil, false
}

func historicalObservationVariants(capabilities wire.Capabilities) ([2]wire.Variant, int) {
	var variants [2]wire.Variant
	n := 0
	if capabilities.Has(wire.BSI) {
		variants[n] = wire.BSI
		n++
	}
	if capabilities.Has(wire.PreV2C) {
		variants[n] = wire.PreV2C
		n++
	}
	return variants, n
}

func observeHistoricalStreamSampled(matrix *core.Bitmap, base core.DecodedSymbol, variant wire.Variant) ([]core.DecodedSymbol, primaryCorrection, bool, bool) {
	switch variant {
	case wire.BSI:
		return observeBSIStreamSampled(matrix, base)
	case wire.PreV2C:
		return observePreV2CStreamSampled(matrix, base)
	default:
		return nil, nil, false, false
	}
}
