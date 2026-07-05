package jabcode

import (
	"image"

	"github.com/srlehn/jabcode/internal/read"
)

// Stream decodes successive camera frames of the same scene, such as a phone
// preview stream. It remembers the hypothesis that read the previous frame -
// which resolution and orientation succeeded - and tries exactly that first
// on the next frame, falling back to the full search only when the scene
// moved too far. A steady hand-held stream therefore pays the search cost on
// the first readable frame and decodes every following frame at the cost of
// a single clean read.
//
// The zero value is ready to use. A Stream is not safe for concurrent use;
// decode the frames of one camera in order. For isolated images use Decode.
type Stream struct {
	s read.Stream
}

// NewStream returns a Stream ready for its first frame.
func NewStream() *Stream { return &Stream{} }

// Decode reads one frame, like Decode, reusing the previous frame's winning
// hypothesis when one exists.
func (st *Stream) Decode(img image.Image) ([]byte, error) {
	return st.s.Decode(img)
}
