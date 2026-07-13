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
	return decodeSeededTracedProfile(levels, f, quit, tr, wire.CReference)
}

func decodeSeededTracedProfile(levels []*image.NRGBA, f finding, quit func() bool, tr *routeTrace, profile wire.Profile) (data []byte, side int, ok bool) {
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
		payload, stage, okj := decodeFromQuadTracedProfile(bm, fps, f.side, quit, detail, profile)
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
	return decodeFromQuadTracedProfile(bm, fps, sideSize, quit, detail, wire.CReference)
}

func decodeFromQuadTracedProfile(bm *core.Bitmap, fps [4]detect.FinderPattern, sideSize image.Point, quit func() bool, detail *DiagnosticAttempt, profile wire.Profile) (data []byte, stage readStage, ok bool) {
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

	symbols := make([]core.DecodedSymbol, maxSymbolNumber)
	symbol := &symbols[0]
	symbol.WireProfile = profile
	symbol.Index = 0
	symbol.HostIndex = 0
	symbol.SideSize = sideSize
	symbol.ModuleSize = (fps[0].ModuleSize + fps[1].ModuleSize + fps[2].ModuleSize + fps[3].ModuleSize) / 4.0
	for i := range 4 {
		symbol.PatternPositions[i] = fps[i].Center
	}

	var ch [3]*core.Bitmap
	haveCh := false
	obs, res := observePrimaryMatrix(matrix, symbol, detail)
	primaryOK := res == core.Success && obs.CorrectPayload() == core.Success
	if !primaryOK {
		if res < 0 {
			return nil, readSampled, false
		}
		// The finder-pattern sample failed; fall back to alignment-pattern
		// resampling exactly like detectPrimary, which needs the version from
		// the partially decoded metadata and the binarized channels.
		sv := symbol.Meta.SideVersion
		if sv.X < 1 || sv.X > 32 || sv.Y < 1 || sv.Y > 32 {
			return nil, readSampled, false
		}
		if quit() {
			return nil, readAborted, false
		}
		symbol.SideSize = image.Pt(spec.VersionToSize(sv.X), spec.VersionToSize(sv.Y))
		ch = detect.BinarizerRGB(bm, nil)
		if detail != nil {
			detail.InitialChannels = ch
			detail.FinalChannels = ch
			detail.Alignment = &detect.AlignmentTrace{}
		}
		haveCh = true
		var apMatrix *core.Bitmap
		if detail != nil {
			apMatrix = detect.SampleSymbolByAlignmentPatternTraced(bm, ch, symbol, fps[:], detail.Alignment)
		} else {
			apMatrix = detect.SampleSymbolByAlignmentPattern(bm, ch, symbol, fps[:])
		}
		if apMatrix == nil {
			return nil, readSampled, false
		}
		apObs, apResult := observePrimaryMatrix(apMatrix, symbol, detail)
		if apResult != core.Success || apObs.CorrectPayload() != core.Success {
			return nil, readSampled, false
		}
	}

	if symbol.Meta.DockedPosition != 0 && !haveCh {
		if quit() {
			return nil, readAborted, false
		}
		ch = detect.BinarizerRGB(bm, nil)
		if detail != nil {
			detail.InitialChannels = ch
			detail.FinalChannels = ch
		}
	}
	data, ok = decodeSymbolsTraced(bm, ch, symbols, 1, detail)
	if !ok {
		return nil, readSampled, false
	}
	return data, readDecoded, true
}
