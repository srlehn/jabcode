package jabcode

import (
	"fmt"
	"image"

	"github.com/srlehn/jabcode/internal/read"
)

// Stream decodes successive camera frames of the same scene, such as a phone
// preview stream. The zero value uses ProfileISO23634. Each frame has a fixed
// route and correction budget: recent
// geometry is replayed first, unused search hypotheses carry forward, and the
// exhaustive single-image ladder is never entered implicitly. Four- and
// eight-colour primary-only symbols may also combine bounded, compatible
// module evidence across frames when no individual frame is sufficient.
// Consequently one frame can return a decode error even when the exhaustive
// Decode function would succeed; later frames can complete the bounded search
// or add the missing evidence.
//
// The zero value is ready to use. A Stream is not safe for concurrent use;
// decode the frames of one camera in order. Results are deterministic for a
// given frame sequence. For isolated images or an exhaustive attempt use
// Decode.
type Stream struct {
	s read.Stream
}

// NewStream returns a Stream ready for its first frame.
func NewStream() *Stream { return &Stream{} }

// NewStreamWithProfile returns a Stream using an explicitly selected compiled
// profile. ProfileLegacy is not supported here because its pre-v2.0 finder
// fallback is part of the exhaustive still-image reader, not the bounded
// per-frame stream scheduler.
func NewStreamWithProfile(profile Profile) (*Stream, error) {
	if err := profile.validateAvailable(); err != nil {
		return nil, err
	}
	if profile == ProfileLegacy {
		return nil, fmt.Errorf("jabcode: profile %s is not supported by Stream", profile)
	}
	return &Stream{s: read.NewStreamProfile(profile.profile())}, nil
}

// Decode reads one frame within the stream's fixed work budget, reusing
// geometry and compatible evidence retained from earlier frames.
func (st *Stream) Decode(img image.Image) ([]byte, error) {
	return st.s.Decode(img)
}
