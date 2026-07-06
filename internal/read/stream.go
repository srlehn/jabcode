package read

import (
	"image"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/detect"
)

// Stream decodes successive camera frames of the same scene. It remembers the
// hypothesis that read the previous frame - the winning geometry (finder quad,
// side size, pre-rotation) and the pyramid level's shorter side - and spends
// the cheapest attempts on it before falling back to the full pyramid search:
// first a direct seeded decode from the remembered quad (no finder search and
// a cheap miss), then one full read at the remembered level and angle, whose
// finder search re-locates the symbol despite drift and refreshes the
// remembered geometry. A steady stream pays the search cost once and then
// reads every frame at the cost of a single decode. The zero value is ready to
// use; a Stream is not safe for concurrent use (frames of one camera arrive in
// order).
//
// Each frame's result is deterministic given the frames decoded before it: the
// prior is a pure function of the sequence, and every attempt is deterministic.
type Stream struct {
	prior *streamPrior
}

// streamPrior is the winning hypothesis of the last decoded frame.
type streamPrior struct {
	side int     // the winning pyramid level's shorter side
	deg  float64 // the winning pre-rotation, 0 for an upright read
	f    finding // the winning geometry in frame coordinates, when located
}

// Decode reads one frame. On success the winning hypothesis becomes the next
// frame's first attempt; on total failure the previous prior is kept - a
// single blurred or occluded frame should not throw away a working lock.
func (s *Stream) Decode(img image.Image) ([]byte, error) {
	levels := pyramidLevels(img)
	if s.prior != nil {
		if data, ok := s.tryPrior(img, levels); ok {
			return data, nil
		}
	}
	if levels != nil {
		if data, side, deg, f, ok := decodePyramid(levels); ok {
			s.remember(side, deg, f)
			return data, nil
		}
		return nil, errDecodeFailed
	}
	var f finding
	if data, deg, ok := decodeSearchFinding(img, nil, &f); ok {
		b := img.Bounds()
		s.remember(min(b.Dx(), b.Dy()), deg, f)
		return data, nil
	}
	return nil, errDecodeFailed
}

// remember stores the winning hypothesis for the next frame. The finding is
// the geometry prior only; a frame's decoded payload is never retained.
func (s *Stream) remember(side int, deg float64, f finding) {
	f.payload = nil
	s.prior = &streamPrior{side: side, deg: deg, f: f}
}

// tryPrior replays the previous frame's winning hypothesis on the pyramid
// level nearest the remembered scale, cheapest first. First a direct seeded
// decode from the remembered quad with no finder search and a cheap miss
// (refine false, so a drifted quad abandons immediately instead of paying the
// alignment-pattern fallback) - this tests only whether the stored geometry
// still lines up. Then, only if that misses, one full read pre-rotated by the
// remembered angle: its finder search is drift-immune and its success
// refreshes the remembered geometry. A miss on both falls through to the full
// pyramid search.
func (s *Stream) tryPrior(img image.Image, levels []*image.NRGBA) ([]byte, bool) {
	lvl := s.nearestLevel(img, levels)
	frame := img.Bounds()

	if s.prior.f.located {
		if data, ok := seedOnLevel(lvl, frame, s.prior.f, false, func() bool { return false }); ok {
			return data, true
		}
	}

	var bm *core.Bitmap
	if s.prior.deg != 0 {
		bm = detect.RotateToBitmap(lvl, s.prior.deg)
	} else {
		bm = core.BitmapFromImage(lvl)
	}
	var rf finding
	data, ok, _ := decodeBitmapFinding(bm, nil, &rf)
	if ok && rf.located {
		// Convert the fresh locate from the rotated level canvas back to frame
		// coordinates (canvas -> level, then level -> frame) and adopt it.
		rf.toImage(s.prior.deg, bm.Width, bm.Height, lvl.Rect.Dx(), lvl.Rect.Dy(), image.Point{})
		rf.scale(float64(frame.Dx())/float64(lvl.Rect.Dx()), float64(frame.Dy())/float64(lvl.Rect.Dy()))
		s.remember(shorterSide(lvl), s.prior.deg, rf)
	}
	return data, ok
}

// nearestLevel picks the pyramid level whose shorter side is closest to the
// remembered scale, or the single base frame when the image is too small to
// pyramid.
func (s *Stream) nearestLevel(img image.Image, levels []*image.NRGBA) *image.NRGBA {
	if levels == nil {
		return pyramidBase(img)
	}
	best := levels[0]
	for _, l := range levels[1:] {
		if absInt(shorterSide(l)-s.prior.side) < absInt(shorterSide(best)-s.prior.side) {
			best = l
		}
	}
	return best
}

func shorterSide(img *image.NRGBA) int { return min(img.Rect.Dx(), img.Rect.Dy()) }

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
