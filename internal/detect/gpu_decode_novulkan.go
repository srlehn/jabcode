//go:build js

package detect

import (
	"errors"
	"image"
	"sync"

	"github.com/srlehn/jabcode/internal/core"
)

var errGPUDecodeUnavailable = errors.New("jabcode: GPU decode is unavailable on JavaScript targets")

// webgpuMinPixels is deliberately conservative: a browser must pay the async
// device and readback overhead, so tiny frames are better served by the CPU.
const webgpuMinPixels = 1024 * 1024

// GPUDecodeSession owns the browser's retained image pyramid. Detection still
// falls back to the CPU until its resident kernels are wired, but retaining the
// pyramid here already removes the most expensive repeated image work from
// every CPU route and keeps the device ownership in one place.
type GPUDecodeSession struct {
	mu      sync.Mutex
	device  *webgpuDevice
	pyramid *webgpuPyramid
	closed  bool
}

// NewAutomaticGPUDecodeSession opens WebGPU lazily for sufficiently large
// frames. Any unavailable or failed browser GPU is an ordinary CPU fallback.
func NewAutomaticGPUDecodeSession(base *core.Bitmap, levelCount int) (*GPUDecodeSession, error) {
	if base == nil || levelCount <= 0 || base.Width <= 0 || base.Height <= 0 ||
		uint64(base.Width)*uint64(base.Height) < webgpuMinPixels {
		return nil, nil
	}
	device, err := openWebGPUDevice()
	if err != nil {
		return nil, nil
	}
	pyramid, err := newWebGPUPyramid(device, base.NRGBA(), levelCount)
	if err != nil {
		device.close()
		return nil, nil
	}
	return &GPUDecodeSession{device: device, pyramid: pyramid}, nil
}

// DownloadLevel exposes one retained level to the CPU detector. The returned
// bitmap is a copy so route consumers cannot mutate shared session state.
func (session *GPUDecodeSession) DownloadLevel(level int) (*core.Bitmap, error) {
	if session == nil {
		return nil, errGPUDecodeUnavailable
	}
	session.mu.Lock()
	defer session.mu.Unlock()
	if session.closed || session.pyramid == nil {
		return nil, errGPUDecodeUnavailable
	}
	img, err := session.pyramid.download(level)
	if err != nil {
		return nil, err
	}
	bm := core.BitmapFromImage(img)
	return bm, nil
}

// LocateLevelFamilies reports that the caller should use its CPU fallback.
func (*GPUDecodeSession) LocateLevelFamilies(
	int,
	FinderFamilySet,
	int,
	func() bool,
	*DetectorTrace,
) (*PrimaryDetector, FinderFamilySet, error) {
	return nil, 0, errGPUDecodeUnavailable
}

// LocateRouteFamilies reports that the caller should use its CPU fallback.
func (*GPUDecodeSession) LocateRouteFamilies(
	int,
	image.Rectangle,
	float64,
	FinderFamilySet,
	int,
	func() bool,
	*DetectorTrace,
) (*PrimaryDetector, FinderFamilySet, image.Point, error) {
	return nil, 0, image.Point{}, errGPUDecodeUnavailable
}

// ProbeLevelFamilies reports that the coarse probe was not handled.
func (*GPUDecodeSession) ProbeLevelFamilies(int, *CoarseProbeTrace) ([]CoarseFamily, bool) {
	return nil, false
}

// Close releases the browser device and makes later downloads fall back.
func (session *GPUDecodeSession) Close() error {
	if session == nil {
		return nil
	}
	session.mu.Lock()
	defer session.mu.Unlock()
	if session.closed {
		return nil
	}
	session.closed = true
	if session.pyramid != nil {
		session.pyramid.close()
	}
	if session.device != nil {
		session.device.close()
	}
	session.device = nil
	session.pyramid = nil
	return nil
}
