package detect

import (
	"errors"
	"fmt"
	"image"
	"sync"
	"sync/atomic"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
)

var automaticGPUDecode = newGPUDecodeRuntime(automaticGPUDevices)

type gpuDecodeRuntime struct {
	devices *gpuDeviceCache

	workspaceMu sync.Mutex
	workspace   *gpuDecodeWorkspace
}

func newGPUDecodeRuntime(devices *gpuDeviceCache) *gpuDecodeRuntime {
	return &gpuDecodeRuntime{devices: devices}
}

// GPUDecodeSession leases the process-wide resident image workspace to one
// decode. Its methods may be called by concurrent pyramid routes; device work
// remains serialized because the workspace reuses scratch buffers.
type GPUDecodeSession struct {
	workspace *gpuDecodeWorkspace
	release   func() error

	operationMu sync.Mutex
	closing     atomic.Bool
	closed      bool
}

// NewAutomaticGPUDecodeSession starts a resident decode workspace when the
// image crosses the measured GPU threshold and a measured discrete Vulkan
// adapter is available. A nil session means the caller should use the CPU path.
func NewAutomaticGPUDecodeSession(base *core.Bitmap, levelCount int) (*GPUDecodeSession, error) {
	return automaticGPUDecode.begin(base, levelCount)
}

func (runtime *gpuDecodeRuntime) begin(
	base *core.Bitmap,
	levelCount int,
) (*GPUDecodeSession, error) {
	if runtime == nil || runtime.devices == nil || base == nil ||
		!automaticGPUWorkload(base.Width, base.Height) {
		return nil, nil
	}
	device, err := runtime.devices.deviceFor(base.Width, base.Height)
	if err != nil || device == nil {
		return nil, nil
	}
	if !runtime.workspaceMu.TryLock() {
		return nil, nil
	}
	keepLease := false
	defer func() {
		if !keepLease {
			runtime.workspaceMu.Unlock()
		}
	}()
	if runtime.workspace == nil || !runtime.workspace.matches(base.Width, base.Height, levelCount) {
		if runtime.workspace != nil {
			if err := runtime.workspace.Close(); err != nil {
				return nil, err
			}
		}
		runtime.workspace, err = newGPUDecodeWorkspace(device, base.Width, base.Height, levelCount)
		if err != nil {
			runtime.workspace = nil
			return nil, err
		}
	}
	if err := runtime.workspace.ladder.UploadAndBuild(base); err != nil {
		return nil, err
	}
	keepLease = true
	return &GPUDecodeSession{
		workspace: runtime.workspace,
		release: func() error {
			runtime.workspaceMu.Unlock()
			return nil
		},
	}, nil
}

// NewGPUDecodeSessionWithDevice starts a resident session on a borrowed
// device. Closing the session releases its buffers and pipelines but leaves
// the device open. It is the explicit parity and embedding seam; normal reads
// use NewAutomaticGPUDecodeSession.
func NewGPUDecodeSessionWithDevice(
	device *vulki.Device,
	base *core.Bitmap,
	levelCount int,
) (*GPUDecodeSession, error) {
	if base == nil {
		return nil, fmt.Errorf("jabcode: GPU decode base image is nil")
	}
	workspace, err := newGPUDecodeWorkspace(device, base.Width, base.Height, levelCount)
	if err != nil {
		return nil, err
	}
	if err := workspace.ladder.UploadAndBuild(base); err != nil {
		_ = workspace.Close()
		return nil, err
	}
	return &GPUDecodeSession{workspace: workspace, release: workspace.Close}, nil
}

type gpuDecodeWorkspace struct {
	width, height int
	levelCount    int
	ladder        *gpuCanvasLadder
	resident      *gpuResidentBinarizer
	preparer      *gpuFinderPassPreparer
}

func newGPUDecodeWorkspace(
	device *vulki.Device,
	width, height, levelCount int,
) (*gpuDecodeWorkspace, error) {
	ladder, err := newGPUCanvasLadderWithDevice(device, width, height, levelCount)
	if err != nil {
		return nil, err
	}
	resident, err := newGPUResidentBinarizerWithDevice(device, width, height)
	if err != nil {
		_ = ladder.Close()
		return nil, err
	}
	preparer, err := newGPUFinderPassPreparer(device, resident)
	if err != nil {
		_ = resident.Close()
		_ = ladder.Close()
		return nil, err
	}
	return &gpuDecodeWorkspace{
		width: width, height: height, levelCount: levelCount,
		ladder: ladder, resident: resident, preparer: preparer,
	}, nil
}

func (workspace *gpuDecodeWorkspace) matches(width, height, levelCount int) bool {
	return workspace != nil && workspace.width == width && workspace.height == height &&
		workspace.levelCount == levelCount
}

func (workspace *gpuDecodeWorkspace) Close() error {
	if workspace == nil {
		return nil
	}
	return errors.Join(
		workspace.resident.releasePreparedBindings(workspace.preparer.descreenFiltered),
		workspace.preparer.Close(),
		workspace.resident.Close(),
		workspace.ladder.Close(),
	)
}

// LocateInitialLevel runs the raw balanced-image finder pass on one retained
// pyramid level. Balanced pixels remain resident unless the pass locates a
// symbol, needs to confirm one missing current-family finder, or records a
// detailed trace.
func (session *GPUDecodeSession) LocateInitialLevel(
	level int,
	wanted FinderFamilySet,
	mode int,
	quit func() bool,
	trace *DetectorTrace,
) (*PrimaryDetector, FinderFamilySet, error) {
	if session == nil || session.closing.Load() {
		return nil, 0, fmt.Errorf("jabcode: GPU decode session is closed")
	}
	session.operationMu.Lock()
	defer session.operationMu.Unlock()
	if session.closing.Load() || session.closed || session.workspace == nil {
		return nil, 0, fmt.Errorf("jabcode: GPU decode session is closed")
	}
	return session.workspace.locateInitialLevel(level, wanted, mode, quit, trace)
}

func (workspace *gpuDecodeWorkspace) locateInitialLevel(
	level int,
	wanted FinderFamilySet,
	mode int,
	quit func() bool,
	trace *DetectorTrace,
) (*PrimaryDetector, FinderFamilySet, error) {
	detector, err := workspace.levelDetector(level, mode, quit, trace)
	if err != nil {
		return nil, 0, err
	}
	found := detector.LocateInitialFinderFamilies(wanted)
	return finishGPUDetector(detector, found, trace)
}

// LocateLevelFamilies runs the complete integrated finder retry ladder on one
// retained pyramid level. Every retry reuses the resident balanced pixels and
// returns only packed masks or compact reductions until pixels are genuinely
// needed downstream.
func (session *GPUDecodeSession) LocateLevelFamilies(
	level int,
	wanted FinderFamilySet,
	mode int,
	quit func() bool,
	trace *DetectorTrace,
) (*PrimaryDetector, FinderFamilySet, error) {
	if session == nil || session.closing.Load() {
		return nil, 0, fmt.Errorf("jabcode: GPU decode session is closed")
	}
	session.operationMu.Lock()
	defer session.operationMu.Unlock()
	if session.closing.Load() || session.closed || session.workspace == nil {
		return nil, 0, fmt.Errorf("jabcode: GPU decode session is closed")
	}
	return session.workspace.locateLevelFamilies(level, wanted, mode, quit, trace)
}

func (workspace *gpuDecodeWorkspace) locateLevelFamilies(
	level int,
	wanted FinderFamilySet,
	mode int,
	quit func() bool,
	trace *DetectorTrace,
) (*PrimaryDetector, FinderFamilySet, error) {
	detector, err := workspace.levelDetector(level, mode, quit, trace)
	if err != nil {
		return nil, 0, err
	}
	found, err := detector.locateFinderFamilies(wanted, workspace.preparer)
	if err != nil {
		return nil, 0, err
	}
	return finishGPUDetector(detector, found, trace)
}

func (workspace *gpuDecodeWorkspace) levelDetector(
	level int,
	mode int,
	quit func() bool,
	trace *DetectorTrace,
) (*PrimaryDetector, error) {
	if workspace == nil || workspace.ladder == nil || workspace.resident == nil {
		return nil, fmt.Errorf("jabcode: GPU decode workspace is closed")
	}
	if level < 0 || level >= len(workspace.ladder.levels) {
		return nil, fmt.Errorf("jabcode: invalid GPU decode level %d", level)
	}
	retained := workspace.ladder.levels[level]
	return workspace.bufferDetector(
		retained.buffer,
		retained.width,
		retained.height,
		mode,
		quit,
		trace,
	)
}

// LocateRouteFamilies rotates a whole retained level or one of its regions and
// runs the complete finder ladder on the resident result. The returned size is
// the rotation canvas used by finding-coordinate conversion.
func (session *GPUDecodeSession) LocateRouteFamilies(
	level int,
	crop image.Rectangle,
	angle float64,
	wanted FinderFamilySet,
	mode int,
	quit func() bool,
	trace *DetectorTrace,
) (*PrimaryDetector, FinderFamilySet, image.Point, error) {
	if session == nil || session.closing.Load() {
		return nil, 0, image.Point{}, fmt.Errorf("jabcode: GPU decode session is closed")
	}
	session.operationMu.Lock()
	defer session.operationMu.Unlock()
	if session.closing.Load() || session.closed || session.workspace == nil {
		return nil, 0, image.Point{}, fmt.Errorf("jabcode: GPU decode session is closed")
	}
	workspace := session.workspace
	size, err := workspace.ladder.Rotate(level, crop, angle)
	if err != nil {
		return nil, 0, image.Point{}, err
	}
	detector, err := workspace.bufferDetector(
		workspace.ladder.route,
		size.X,
		size.Y,
		mode,
		quit,
		trace,
	)
	if err != nil {
		return nil, 0, image.Point{}, err
	}
	found, err := detector.locateFinderFamilies(wanted, workspace.preparer)
	if err != nil {
		return nil, 0, image.Point{}, err
	}
	detector, found, err = finishGPUDetector(detector, found, trace)
	return detector, found, size, err
}

func (workspace *gpuDecodeWorkspace) bufferDetector(
	input *vulki.Buffer,
	width, height int,
	mode int,
	quit func() bool,
	trace *DetectorTrace,
) (*PrimaryDetector, error) {
	channels, err := workspace.resident.Binarize(
		input,
		width,
		height,
		nil,
		false,
	)
	if err != nil {
		return nil, err
	}
	workspace.preparer.setInput(width, height, trace != nil)
	balanced := &core.Bitmap{
		Width: width, Height: height, Channels: 4,
	}
	detector := &PrimaryDetector{
		BM: balanced, Ch: channels, Mode: mode, Quit: quit, Trace: trace,
	}
	detector.materializeBitmap = func() error {
		downloaded, err := workspace.resident.DownloadBalanced(width, height)
		if err != nil {
			return err
		}
		balanced.Pix = downloaded.Pix
		return nil
	}
	return detector, nil
}

func finishGPUDetector(
	detector *PrimaryDetector,
	found FinderFamilySet,
	trace *DetectorTrace,
) (*PrimaryDetector, FinderFamilySet, error) {
	if (found != 0 || trace != nil) && !detector.ensureBitmap() {
		if detector.materializeErr != nil {
			return nil, 0, detector.materializeErr
		}
		return nil, 0, fmt.Errorf("jabcode: materialize resident GPU balanced image")
	}
	return detector, found, nil
}

// Close releases the workspace after any in-flight GPU operation. Automatic
// sessions cache it for another same-sized decode; borrowed-device sessions
// release their buffers and pipelines.
func (session *GPUDecodeSession) Close() error {
	if session == nil || session.closing.Swap(true) {
		return nil
	}
	session.operationMu.Lock()
	defer session.operationMu.Unlock()
	if session.closed {
		return nil
	}
	session.closed = true
	if session.release != nil {
		return session.release()
	}
	return nil
}
