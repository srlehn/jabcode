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

// LocateLevelFamilies runs the existing finder ladder over GPU-prepared masks.
// Route and probe methods remain on the CPU fallback until their shared
// resident canvases are implemented.
func (session *GPUDecodeSession) LocateLevelFamilies(
	level int,
	wanted FinderFamilySet,
	mode int,
	quit func() bool,
	trace *DetectorTrace,
) (*PrimaryDetector, FinderFamilySet, error) {
	if session == nil {
		return nil, 0, errGPUDecodeUnavailable
	}
	session.mu.Lock()
	defer session.mu.Unlock()
	if session.closed || session.pyramid == nil || session.device == nil {
		return nil, 0, errGPUDecodeUnavailable
	}
	if quit != nil && quit() {
		return nil, 0, errGPUDecodeUnavailable
	}
	img, err := session.pyramid.download(level)
	if err != nil {
		return nil, 0, err
	}
	bm := core.BitmapFromImage(img)
	if err := session.device.balanceRGB(bm); err != nil {
		return nil, 0, err
	}
	channels, err := session.device.webgpuBinarizeRGB(bm, false)
	if err != nil {
		return nil, 0, err
	}
	detector := &PrimaryDetector{BM: bm, Ch: channels, Mode: mode, Quit: quit}
	if trace != nil {
		detector.Trace = trace
	}
	found := detector.LocateFinderFamilies(wanted)
	return detector, found, nil
}

// LocateRouteFamilies keeps rotation on the CPU for now, then runs the
// balanced mask preparation and finder ladder through the same WebGPU path as
// upright levels. This makes the route seam useful without duplicating the
// detector or changing the CPU fallback contract.
func (session *GPUDecodeSession) LocateRouteFamilies(
	level int,
	crop image.Rectangle,
	angle float64,
	wanted FinderFamilySet,
	mode int,
	quit func() bool,
	trace *DetectorTrace,
) (*PrimaryDetector, FinderFamilySet, image.Point, error) {
	if session == nil {
		return nil, 0, image.Point{}, errGPUDecodeUnavailable
	}
	session.mu.Lock()
	defer session.mu.Unlock()
	if session.closed || session.pyramid == nil || session.device == nil {
		return nil, 0, image.Point{}, errGPUDecodeUnavailable
	}
	if quit != nil && quit() {
		return nil, 0, image.Point{}, errGPUDecodeUnavailable
	}
	img, err := session.pyramid.download(level)
	if err != nil {
		return nil, 0, image.Point{}, err
	}
	bounds := img.Bounds()
	if crop.Empty() || crop.Intersect(bounds) != crop {
		return nil, 0, image.Point{}, errGPUDecodeUnavailable
	}
	rotated := RotateToBitmap(img.SubImage(crop), angle)
	if err := session.device.balanceRGB(rotated); err != nil {
		return nil, 0, image.Point{}, err
	}
	channels, err := session.device.webgpuBinarizeRGB(rotated, false)
	if err != nil {
		return nil, 0, image.Point{}, err
	}
	detector := &PrimaryDetector{BM: rotated, Ch: channels, Mode: mode, Quit: quit}
	if trace != nil {
		detector.Trace = trace
	}
	found := detector.LocateFinderFamilies(wanted)
	return detector, found, image.Pt(rotated.Width, rotated.Height), nil
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
