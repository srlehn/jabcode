package jabcode

import (
	"image"

	"github.com/srlehn/jabcode/internal/read"
)

// Stream decodes successive images from one coherent frame sequence. Frames
// may come from a live camera, network video, or a decoded recording. Each
// frame has a fixed route and correction budget: recent
// geometry is replayed first, unused search hypotheses carry forward, and the
// exhaustive single-image ladder is never entered implicitly. Four- and
// eight-colour primary-only symbols may also combine bounded, compatible
// module evidence across frames when no individual frame is sufficient.
// Consequently one frame can return a decode error even when the exhaustive
// Decode function would succeed; later frames can complete the bounded search
// or add the missing evidence.
//
// Stream automatically accepts every decoder capability compiled into the
// build. Optional finder signatures are classified inside the same image
// traversal, compatible wire variants share the physical-family sample, and
// the scheduler chooses at most one irreducible wire correction per frame.
// Disabled capabilities add no stream route.
//
// The zero value is ready to use. A Stream is not safe for concurrent use;
// decode one coherent frame sequence in order. Results are deterministic for
// a given frame sequence. For isolated images or an exhaustive attempt use
// Decode.
type Stream struct {
	s read.Stream
}

// NewStream returns a Stream ready for its first frame.
func NewStream() *Stream { return &Stream{} }

// Decode reads one frame within the stream's fixed work budget, reusing
// geometry and compatible evidence retained from earlier frames.
func (st *Stream) Decode(img image.Image) ([]byte, error) {
	return st.s.Decode(img)
}
