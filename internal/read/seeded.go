package read

import (
	"bytes"
	"image"
	"math"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/decode"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/spec"
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
	frame := levels[len(levels)-1].Rect
	for j := 1; j < len(levels); j++ {
		if quit() {
			return nil, 0, false
		}
		payload, okj := seedOnLevel(levels[j], frame, f, true, quit)
		if quit() {
			return nil, 0, false
		}
		if !okj {
			continue
		}
		if f.payload != nil {
			return payload, shorterSide(levels[0]), bytes.Equal(payload, f.payload)
		}
		return payload, shorterSide(levels[j]), true
	}
	return nil, 0, false
}

// seedOnLevel decodes one pyramid level directly from a frame-coordinate
// finding: rotate the level by the finding's angle, scale the quad in and map
// it onto the rotation canvas (centred on the level, rotateInto's forward
// mapping), then decode from that geometry. refine selects whether a failed
// direct sample may escalate into the alignment-pattern fallback (the pyramid
// chain wants it; a Stream's prior replay must miss cheaply instead).
func seedOnLevel(lvl *image.NRGBA, frame image.Rectangle, f finding, refine bool, quit func() bool) (data []byte, ok bool) {
	sx := float64(lvl.Rect.Dx()) / float64(frame.Dx())
	sy := float64(lvl.Rect.Dy()) / float64(frame.Dy())

	var bm *core.Bitmap
	if f.deg != 0 {
		bm = detect.RotateToBitmap(lvl, f.deg)
	} else {
		bm = core.BitmapFromImage(lvl)
	}
	detect.BalanceRGB(bm)

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
	return decodeFromQuad(bm, fps, f.side, refine, quit)
}

// decodeFromQuad is detectPrimary entered directly at a known finder quad and
// side size: rectify, sample and decode the primary, then decode any docked
// secondaries. With refine set, a failed finder-pattern sample escalates into
// the same alignment-pattern fallback detectPrimary uses; without it the
// failure returns immediately - the cheap-miss contract of a Stream's prior
// replay. The binarized channels are only computed when a consumer needs them
// (the AP fallback, secondary detection) - the direct sample reads the
// balanced bitmap, which is what makes the seeded path cheap.
func decodeFromQuad(bm *core.Bitmap, fps [4]detect.FinderPattern, sideSize image.Point, refine bool, quit func() bool) (data []byte, ok bool) {
	pt := core.PerspectiveTransform(fps[0].Center, fps[1].Center, fps[2].Center, fps[3].Center, sideSize)
	matrix := detect.SampleSymbol(bm, pt, sideSize)
	if matrix == nil {
		return nil, false
	}

	symbols := make([]core.DecodedSymbol, maxSymbolNumber)
	symbol := &symbols[0]
	symbol.Index = 0
	symbol.HostIndex = 0
	symbol.SideSize = sideSize
	symbol.ModuleSize = (fps[0].ModuleSize + fps[1].ModuleSize + fps[2].ModuleSize + fps[3].ModuleSize) / 4.0
	for i := range 4 {
		symbol.PatternPositions[i] = fps[i].Center
	}

	var ch [3]*core.Bitmap
	haveCh := false
	switch res := decode.DecodePrimary(matrix, symbol); {
	case res == core.Success:
	case res < 0:
		return nil, false
	default:
		if !refine {
			return nil, false
		}
		// The finder-pattern sample failed; fall back to alignment-pattern
		// resampling exactly like detectPrimary, which needs the version from
		// the partially decoded metadata and the binarized channels.
		sv := symbol.Meta.SideVersion
		if sv.X < 1 || sv.X > 32 || sv.Y < 1 || sv.Y > 32 {
			return nil, false
		}
		if quit() {
			return nil, false
		}
		symbol.SideSize = image.Pt(spec.VersionToSize(sv.X), spec.VersionToSize(sv.Y))
		ch = detect.BinarizerRGB(bm, nil)
		haveCh = true
		apMatrix := detect.SampleSymbolByAlignmentPattern(bm, ch, symbol, fps[:])
		if apMatrix == nil {
			return nil, false
		}
		if decode.DecodePrimary(apMatrix, symbol) != core.Success {
			return nil, false
		}
	}

	if symbol.Meta.DockedPosition != 0 && !haveCh {
		if quit() {
			return nil, false
		}
		ch = detect.BinarizerRGB(bm, nil)
	}
	return decodeSymbols(bm, ch, symbols, 1)
}
