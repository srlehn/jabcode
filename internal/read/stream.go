package read

import (
	"image"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/detect"
)

// Stream decodes successive camera frames of the same scene. It remembers the
// hypothesis that read the previous frame - the pyramid level's shorter side
// and the pre-rotation angle - and spends one cheap decode on exactly that
// hypothesis before falling back to the full pyramid search, so a steady
// hand-held stream pays the search cost once and then reads every frame at
// the cost of a single clean decode. The zero value is ready to use; a Stream
// is not safe for concurrent use (frames of one camera arrive in order).
//
// Each frame's result is deterministic given the frames decoded before it:
// the prior is a pure function of the sequence, and both the prior attempt
// and the fallback search are deterministic.
type Stream struct {
	prior *streamPrior
}

// streamPrior is the winning hypothesis of the last decoded frame.
type streamPrior struct {
	side int     // the winning pyramid level's shorter side
	deg  float64 // the winning pre-rotation, 0 for an upright read
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
		if data, side, deg, ok := decodePyramid(levels); ok {
			s.prior = &streamPrior{side: side, deg: deg}
			return data, nil
		}
		return nil, errDecodeFailed
	}
	if data, deg, ok := decodeSearch(img, nil); ok {
		b := img.Bounds()
		s.prior = &streamPrior{side: min(b.Dx(), b.Dy()), deg: deg}
		return data, nil
	}
	return nil, errDecodeFailed
}

// tryPrior replays the previous frame's winning hypothesis: one decode at the
// pyramid level nearest the remembered scale, pre-rotated by the remembered
// angle. The frame usually moved a little between captures, but the rung
// angles tolerate several degrees of residual, so a steady stream keeps
// hitting.
func (s *Stream) tryPrior(img image.Image, levels []*image.NRGBA) ([]byte, bool) {
	var lvl image.Image = img
	if levels != nil {
		best := levels[0]
		for _, l := range levels[1:] {
			if absInt(shorterSide(l)-s.prior.side) < absInt(shorterSide(best)-s.prior.side) {
				best = l
			}
		}
		lvl = best
	}
	var bm *core.Bitmap
	if s.prior.deg != 0 {
		bm = detect.RotateToBitmap(lvl, s.prior.deg)
	} else {
		bm = core.BitmapFromImage(lvl)
	}
	data, ok, _ := decodeBitmap(bm, nil)
	return data, ok
}

func shorterSide(img *image.NRGBA) int { return min(img.Rect.Dx(), img.Rect.Dy()) }

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
