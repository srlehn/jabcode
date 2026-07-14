package read

import (
	"bytes"
	"image"
	"math"

	"github.com/srlehn/jabcode/internal/core"
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
	return decodeSeededTraced(levels, f, quit, nil)
}

func decodeSeededTraced(levels []*image.NRGBA, f finding, quit func() bool, tr *routeTrace) (data []byte, side int, ok bool) {
	return decodeSeededTracedCapabilities(levels, f, quit, tr, compiledCapabilities())
}

func decodeSeededTracedOnly(levels []*image.NRGBA, f finding, quit func() bool, tr *routeTrace, variant wire.Variant) (data []byte, side int, ok bool) {
	return decodeSeededTracedCapabilities(levels, f, quit, tr, variant.Mask())
}

func decodeSeededTracedCapabilities(levels []*image.NRGBA, f finding, quit func() bool, tr *routeTrace, capabilities wire.Capabilities) (data []byte, side int, ok bool) {
	base := levels[0].Rect
	for j := 1; j < len(levels); j++ {
		if quit() {
			return nil, 0, false
		}
		lvl := levels[j]
		sx := float64(lvl.Rect.Dx()) / float64(base.Dx())
		sy := float64(lvl.Rect.Dy()) / float64(base.Dy())

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
		payload, stage, okj := decodeFromQuadTracedCapabilities(bm, fps, f.side, quit, detail, capabilities)
		tr.finishAttempt(routeAttempt{deg: f.deg, roi: -1, stage: stage, side: f.side}, detail, payload)
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
			return payload, shorterSide(levels[0]), bytes.Equal(payload, f.payload)
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
	data, _, ok = decodeFromQuadTraced(bm, fps, sideSize, quit, nil)
	return data, ok
}

func decodeFromQuadTraced(bm *core.Bitmap, fps [4]detect.FinderPattern, sideSize image.Point, quit func() bool, detail *DiagnosticAttempt) (data []byte, stage readStage, ok bool) {
	return decodeFromQuadTracedCapabilities(bm, fps, sideSize, quit, detail, compiledCapabilities())
}

func decodeFromQuadTracedOnly(bm *core.Bitmap, fps [4]detect.FinderPattern, sideSize image.Point, quit func() bool, detail *DiagnosticAttempt, variant wire.Variant) (data []byte, stage readStage, ok bool) {
	return decodeFromQuadTracedCapabilities(bm, fps, sideSize, quit, detail, variant.Mask())
}

func decodeFromQuadTracedCapabilities(bm *core.Bitmap, fps [4]detect.FinderPattern, sideSize image.Point, quit func() bool, detail *DiagnosticAttempt, capabilities wire.Capabilities) (data []byte, stage readStage, ok bool) {
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

	isoTried, isoNC := false, -1
	for _, variant := range currentFamilyVariants {
		if !capabilities.Has(variant) {
			continue
		}
		if variant == wire.ISOHighColor && isoTried && isoNC <= 2 {
			continue
		}

		symbol := base
		symbol.WireVariant = variant
		obs, res := observePrimaryMatrix(matrix, &symbol, detail)
		if variant == wire.ISO23634 {
			isoTried, isoNC = true, symbol.Meta.NC
		}
		primaryOK := res == core.Success && obs.CorrectPayload() == core.Success
		if !primaryOK && res >= 0 {
			// The finder-pattern sample failed; fall back to alignment-pattern
			// resampling with the version interpreted by this variant.
			sv := symbol.Meta.SideVersion
			if sv.X >= 1 && sv.X <= 32 && sv.Y >= 1 && sv.Y <= 32 && ensureChannels() {
				symbol.SideSize = image.Pt(spec.VersionToSize(sv.X), spec.VersionToSize(sv.Y))
				var apMatrix *core.Bitmap
				if detail != nil {
					detail.Alignment = &detect.AlignmentTrace{}
					apMatrix = detect.SampleSymbolByAlignmentPatternTraced(bm, ch, &symbol, fps[:], detail.Alignment)
				} else {
					apMatrix = detect.SampleSymbolByAlignmentPattern(bm, ch, &symbol, fps[:])
				}
				if apMatrix != nil {
					apObs, apResult := observePrimaryMatrix(apMatrix, &symbol, detail)
					primaryOK = apResult == core.Success && apObs.CorrectPayload() == core.Success
				}
			}
		}
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
