package read

import (
	"image"
	"math"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/spec"
	"github.com/srlehn/jabcode/internal/wire"
)

// decodeSeeded resumes a read from the coarsest level's published finding on
// the finer pyramid levels: scale the finder quad, rotate the level by the
// finding's angle, sample and decode - no finder search at fine resolution.
// The chain climbs the pyramid until one level decodes, then applies the
// commit rule:
//
//   - the finding carries a payload (the coarse route decoded): ok only if the
//     seeded re-decode agrees byte-for-byte - two scales reading the same bytes
//     through the LDPC syndrome gate is stronger evidence than either alone. A
//     disagreement returns ok=false and leaves the outcome to the blind
//     ladders.
//   - the finding is locate-only (the coarse route saw the quad but could not
//     classify, the small-module case): the seeded decode stands on its own,
//     like any other single success.
//
// side reports the level whose full read a Stream should replay: the coarsest
// level when the coarse route decoded there, otherwise the level the seeded
// decode succeeded on.
//
// The chain is sequential and sourced only from the deterministic finding, so
// its result is a pure function of the input regardless of scheduling.
func decodeSeeded(levels []*image.NRGBA, f finding, quit func() bool) (data []byte, side int, ok bool) {
	message, side, ok := decodeSeededTracedCapabilities(eagerPyramid(levels), f, quit, nil, compiledCapabilities())
	return messageTransmission(message), side, ok
}

func decodeSeededTraced(levels []*image.NRGBA, f finding, quit func() bool, tr *routeTrace) (data []byte, side int, ok bool) {
	message, side, ok := decodeSeededTracedCapabilities(eagerPyramid(levels), f, quit, tr, compiledCapabilities())
	return messageTransmission(message), side, ok
}

func decodeSeededTracedOnly(levels []*image.NRGBA, f finding, quit func() bool, tr *routeTrace, variant wire.Variant) (data []byte, side int, ok bool) {
	message, side, ok := decodeSeededTracedCapabilities(eagerPyramid(levels), f, quit, tr, variant.Mask())
	return messageTransmission(message), side, ok
}

func decodeSeededTracedCapabilities(p *pyramid, f finding, quit func() bool, tr *routeTrace, capabilities wire.Capabilities) (data *Message, side int, ok bool) {
	base := p.dim(0)
	for j := 1; j < p.count(); j++ {
		if quit() {
			return nil, 0, false
		}
		lvl := p.level(j)
		sx := float64(lvl.Rect.Dx()) / float64(base.X)
		sy := float64(lvl.Rect.Dy()) / float64(base.Y)

		var bm *core.Bitmap
		if f.deg != 0 {
			bm = detect.RotateToBitmap(lvl, f.deg)
		} else {
			bm = core.BitmapFromImage(lvl)
		}
		detect.BalanceRGB(bm)

		// Scale the quad into this level, then map it onto the rotation canvas
		// (centred on the level, rotateInto's forward mapping).
		rad := f.deg * math.Pi / 180
		cs, sn := math.Cos(rad), math.Sin(rad)
		lcx, lcy := float64(lvl.Rect.Dx())/2, float64(lvl.Rect.Dy())/2
		ccx, ccy := float64(bm.Width)/2, float64(bm.Height)/2
		var fps [4]detect.FinderPattern
		for i := range 4 {
			dx, dy := f.quad[i].X*sx-lcx, f.quad[i].Y*sy-lcy
			fps[i] = detect.FinderPattern{
				Typ:        i,
				Center:     core.PointF{X: cs*dx - sn*dy + ccx, Y: sn*dx + cs*dy + ccy},
				ModuleSize: f.sizes[i] * (sx + sy) / 2,
				FoundCount: 1,
			}
		}

		oldLevel := -1
		if tr != nil {
			oldLevel = tr.level
			tr.level = j
		}
		detail := tr.beginAttempt("seeded", f.deg, -1)
		payload, stage, okj := decodeFromQuadFamilyTracedCapabilities(bm, fps, f.side, f.family, quit, detail, capabilities)
		tr.finishAttempt(routeAttempt{deg: f.deg, roi: -1, stage: stage, side: f.side}, detail, messageTransmission(payload))
		if tr != nil {
			tr.level = oldLevel
		}
		if quit() {
			return nil, 0, false
		}
		if !okj {
			continue
		}
		if f.payload != nil {
			return payload, p.side(0), equalMessages(payload, f.payload)
		}
		return payload, shorterSide(lvl), true
	}
	return nil, 0, false
}

// decodeFromQuad is detectPrimary entered directly at a known finder quad and
// side size: rectify, sample and decode the primary, with the same
// alignment-pattern fallback, then decode any docked secondaries. The
// binarized channels are only computed when a consumer needs them (the AP
// fallback, secondary detection) - the direct sample reads the balanced
// bitmap, which is what makes the seeded path cheap.
func decodeFromQuad(bm *core.Bitmap, fps [4]detect.FinderPattern, sideSize image.Point, quit func() bool) (data []byte, ok bool) {
	message, _, ok := decodeFromQuadTracedCapabilities(bm, fps, sideSize, quit, nil, compiledCapabilities())
	return messageTransmission(message), ok
}

func decodeFromQuadTraced(bm *core.Bitmap, fps [4]detect.FinderPattern, sideSize image.Point, quit func() bool, detail *DiagnosticAttempt) (data []byte, stage readStage, ok bool) {
	message, stage, ok := decodeFromQuadTracedCapabilities(bm, fps, sideSize, quit, detail, compiledCapabilities())
	return messageTransmission(message), stage, ok
}

func decodeFromQuadTracedOnly(bm *core.Bitmap, fps [4]detect.FinderPattern, sideSize image.Point, quit func() bool, detail *DiagnosticAttempt, variant wire.Variant) (data []byte, stage readStage, ok bool) {
	message, stage, ok := decodeFromQuadTracedCapabilities(bm, fps, sideSize, quit, detail, variant.Mask())
	return messageTransmission(message), stage, ok
}

func decodeFromQuadTracedCapabilities(bm *core.Bitmap, fps [4]detect.FinderPattern, sideSize image.Point, quit func() bool, detail *DiagnosticAttempt, capabilities wire.Capabilities) (data *Message, stage readStage, ok bool) {
	return decodeFromQuadFamilyTracedCapabilities(bm, fps, sideSize, detect.FinderFamilyCurrent, quit, detail, capabilities)
}

// decodeFromQuadFamilyTracedCapabilities samples known geometry once and sends
// that matrix only to interpretations compatible with the finder signature
// that established it. It never repeats finder detection for another format.
func decodeFromQuadFamilyTracedCapabilities(bm *core.Bitmap, fps [4]detect.FinderPattern, sideSize image.Point, family detect.FinderFamily, quit func() bool, detail *DiagnosticAttempt, capabilities wire.Capabilities) (data *Message, stage readStage, ok bool) {
	if detail != nil {
		detail.Balanced = bm
		detail.Finders = append([]detect.FinderPattern(nil), fps[:]...)
		detail.Side = sideSize
	}
	pt := core.PerspectiveTransform(fps[0].Center, fps[1].Center, fps[2].Center, fps[3].Center, sideSize)
	if detail != nil {
		detail.Transform = pt
		detail.HasTransform = true
	}
	matrix := detect.SampleSymbol(bm, pt, sideSize)
	if matrix == nil {
		return nil, readNoSample, false
	}
	if detail != nil {
		detail.Sampled = matrix
	}

	base := core.DecodedSymbol{
		Index:      0,
		HostIndex:  0,
		SideSize:   sideSize,
		ModuleSize: (fps[0].ModuleSize + fps[1].ModuleSize + fps[2].ModuleSize + fps[3].ModuleSize) / 4.0,
	}
	for i := range 4 {
		base.PatternPositions[i] = fps[i].Center
	}

	var ch [3]*core.Bitmap
	haveCh := false
	ensureChannels := func() bool {
		if haveCh {
			return true
		}
		if quit() {
			return false
		}
		ch = detect.BinarizerRGB(bm, nil)
		if detail != nil {
			detail.InitialChannels = ch
			detail.FinalChannels = ch
		}
		haveCh = true
		return true
	}
	channels := func() ([3]*core.Bitmap, bool) {
		if !ensureChannels() {
			return [3]*core.Bitmap{}, false
		}
		return ch, true
	}

	if family == detect.FinderFamilyBSI {
		data, ok := decodeHistoricalSampled(bm, matrix, base, detail, capabilities, channels)
		if ok {
			return data, readDecoded, true
		}
		return nil, readSampled, false
	}
	if family != detect.FinderFamilyCurrent {
		return nil, readSampled, false
	}

	variants, variantCount := currentObservationVariants(capabilities)
	var moduleEvidence decode.ModuleEvidenceCache
	var moduleEvidenceCache *decode.ModuleEvidenceCache
	var alignmentSamples alignmentSampleCache
	var alignmentCache *alignmentSampleCache
	if shareCurrentFamilyEvidence && variantCount > 1 {
		moduleEvidenceCache = &moduleEvidence
		alignmentCache = &alignmentSamples
	}
	for _, variant := range variants[:variantCount] {
		traceStart := primaryTraceCount(detail)
		symbol := base
		symbol.WireVariant = variant
		obs, res := observePrimaryMatrix(matrix, &symbol, detail)
		primaryOK := res == core.Success && correctPrimaryPayload(obs, moduleEvidenceCache) == core.Success
		if !primaryOK && res >= 0 {
			// The finder-pattern sample failed; fall back to alignment-pattern
			// resampling with the version interpreted by this variant.
			sv := symbol.Meta.SideVersion
			if sv.X >= 1 && sv.X <= 32 && sv.Y >= 1 && sv.Y <= 32 && ensureChannels() {
				symbol.SideSize = image.Pt(spec.VersionToSize(sv.X), spec.VersionToSize(sv.Y))
				apMatrix := samplePrimaryByAlignment(bm, ch, &symbol, fps[:], detail, alignmentCache)
				if apMatrix != nil {
					apObs, apResult := observePrimaryMatrix(apMatrix, &symbol, detail)
					primaryOK = apResult == core.Success && correctPrimaryPayload(apObs, moduleEvidenceCache) == core.Success
				}
			}
		}
		normalizeCurrentVariant(&symbol, detail, capabilities, traceStart)
		if !primaryOK {
			continue
		}
		if symbol.Meta.DockedPosition != 0 && !ensureChannels() {
			return nil, readAborted, false
		}
		symbols := make([]core.DecodedSymbol, maxSymbolNumber)
		symbols[0] = symbol
		data, ok = decodeSymbolsTraced(bm, ch, symbols, 1, detail)
		if ok {
			return data, readDecoded, true
		}
	}
	return nil, readSampled, false
}
