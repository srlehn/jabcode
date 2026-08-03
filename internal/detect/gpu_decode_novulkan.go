//go:build js

package detect

import (
	"errors"
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

// WarmAutomaticGPUDecode does nothing in the browser. Adapter and device
// requests there are promises the session already awaits, and starting one
// early would have to be cancelled or awaited somewhere, which is more
// machinery than the browser route's acquisition costs.
func WarmAutomaticGPUDecode(int, int, int) {}

// NewAutomaticGPUDecodeSession opens WebGPU lazily for sufficiently large
// frames. Any unavailable or failed browser GPU is an ordinary CPU fallback.
func NewAutomaticGPUDecodeSession(base *core.Bitmap, levelCount int) (*GPUDecodeSession, error) {
	if base == nil || levelCount <= 0 || base.Width <= 0 || base.Height <= 0 ||
		gpuRoutesDisabled.Load() || uint64(base.Width)*uint64(base.Height) < webgpuMinPixels {
		return nil, nil
	}
	device, err := openWebGPUDevice()
	if err != nil {
		return nil, nil
	}
	pyramid, err := newWebGPUPyramid(device, base.NRGBA(), levelCount)
	if err != nil {
		return nil, nil
	}
	return &GPUDecodeSession{device: device, pyramid: pyramid}, nil
}

// ReplaceBase refreshes the resident pyramid when frame geometry is stable.
func (session *GPUDecodeSession) ReplaceBase(base *core.Bitmap) error {
	if session == nil {
		return errGPUDecodeUnavailable
	}
	session.mu.Lock()
	defer session.mu.Unlock()
	if session.closed || session.device == nil || session.pyramid == nil || base == nil {
		return errGPUDecodeUnavailable
	}
	if base.Width != session.pyramid.levels[0].width || base.Height != session.pyramid.levels[0].height {
		return errGPUDecodeUnavailable
	}
	if err := session.pyramid.replaceBase(base.NRGBA()); err != nil {
		return session.failLocked(err)
	}
	return nil
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
		return nil, session.failLocked(err)
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
	prepared, err := session.pyramid.prepare(level, false)
	if err != nil {
		return nil, 0, session.failLocked(err)
	}
	defer prepared.close()
	detector := &PrimaryDetector{BM: prepared.bm, Ch: prepared.channels, Mode: mode, Quit: quit}
	if trace != nil {
		detector.Trace = trace
	}
	found, err := detector.locateFinderFamilies(wanted, webgpuFinderPassPreparer{
		device: session.device,
		bm:     prepared.bm,
		input:  prepared.balanced,
		trace:  trace != nil,
	})
	if err != nil {
		return nil, 0, session.failLocked(err)
	}
	return detector, found, nil
}

// scanDirection reports no device sweep. The browser route has no directional
// kernel yet, so its directional retries walk on the host exactly as before.
func (webgpuFinderPassPreparer) scanDirection(scanDirection, int, int) ([]finderDirHit, error) {
	return nil, nil
}

func (session *GPUDecodeSession) failLocked(err error) error {
	if err != nil && session != nil && !session.closed {
		retireAutomaticWebGPUDevice(session.device)
		session.closed = true
		if session.pyramid != nil {
			session.pyramid.close()
		}
		session.device = nil
		session.pyramid = nil
	}
	return err
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
	session.device = nil
	session.pyramid = nil
	return nil
}
