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
	// kernels live as long as the process-wide device: workspaces of any size
	// share the compiled pipelines instead of recompiling WGSL per resize.
	kernels *gpuDecodeKernels
}

func newGPUDecodeRuntime(devices *gpuDeviceCache) *gpuDecodeRuntime {
	return &gpuDecodeRuntime{devices: devices}
}

// GPUDecodeSession leases the process-wide resident image workspace to one
// decode. Its methods may be called by concurrent pyramid and rotation
// routes; each route leases its own context from the workspace pool, so one
// route's CPU scan overlaps another route's device kernels.
type GPUDecodeSession struct {
	workspace *gpuDecodeWorkspace
	release   func() error

	closing atomic.Bool
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
	if runtime.kernels == nil {
		runtime.kernels = newGPUDecodeKernels(device)
	}
	runtime.kernels.warmFinderChains()
	if runtime.workspace == nil || !runtime.workspace.matches(base.Width, base.Height, levelCount) {
		// Retire the cached pointer before closing: a workspace whose Close
		// failed has already released device state and must never be matched
		// and leased again.
		if retired := runtime.workspace; retired != nil {
			runtime.workspace = nil
			if err := retired.Close(); err != nil {
				return nil, err
			}
		}
		runtime.workspace, err = newGPUDecodeWorkspace(device, runtime.kernels, base.Width, base.Height, levelCount)
		if err != nil {
			runtime.workspace = nil
			return nil, err
		}
	}
	if err := runtime.workspace.ladder.UploadAndBuild(base); err != nil {
		return nil, err
	}
	runtime.workspace.contexts.reopen()
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
	kernels := newGPUDecodeKernels(device)
	kernels.warmFinderChains()
	workspace, err := newGPUDecodeWorkspace(device, kernels, base.Width, base.Height, levelCount)
	if err != nil {
		_ = kernels.Close()
		return nil, err
	}
	workspace.ownsKernels = true
	if err := workspace.ladder.UploadAndBuild(base); err != nil {
		_ = workspace.Close()
		return nil, err
	}
	return &GPUDecodeSession{workspace: workspace, release: workspace.Close}, nil
}

type gpuDecodeWorkspace struct {
	width, height int
	levelCount    int
	kernels       *gpuDecodeKernels
	ownsKernels   bool
	ladder        *gpuCanvasLadder
	contexts      *gpuRouteContextPool
}

func newGPUDecodeWorkspace(
	device *vulki.Device,
	kernels *gpuDecodeKernels,
	width, height, levelCount int,
) (*gpuDecodeWorkspace, error) {
	ladder, err := newGPUCanvasLadderWithDevice(device, kernels, width, height, levelCount)
	if err != nil {
		return nil, err
	}
	return &gpuDecodeWorkspace{
		width: width, height: height, levelCount: levelCount,
		kernels:  kernels,
		ladder:   ladder,
		contexts: newGPURouteContextPool(device, kernels, ladder),
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
	err := errors.Join(
		workspace.contexts.Close(),
		workspace.ladder.Close(),
	)
	if workspace.ownsKernels {
		err = errors.Join(err, workspace.kernels.Close())
	}
	return err
}

// gpuRouteContext owns everything one concurrent route mutates on the device:
// a rotation canvas with its own parameter buffer and binding sets, one
// resident binarizer instance and one finder-pass preparer. Routes share only
// the device, the retained pyramid levels (read-only after the build) and the
// compiled kernels. The pool hands a context to one route at a time.
type gpuRouteContext struct {
	capWidth  int
	capHeight int
	// deviceBytes is the pool's budgeted device-memory cost of this context,
	// fixed at creation (see gpuRouteContextDeviceBytes).
	deviceBytes uint64
	canvas      *gpuRouteCanvas
	resident    *gpuResidentBinarizer
	preparer    *gpuFinderPassPreparer

	// epoch counts pool releases. Detector closures that materialize resident
	// pixels capture the epoch at lease time and refuse to touch buffers a
	// later route may have overwritten.
	epoch atomic.Uint64
}

// gpuRouteContextDeviceBytes bounds the device memory one route context of
// the given capacity can ever hold: the route canvas (4 B/px), the balanced
// image (4), the raw and final masks (4+4), the packed masks (~0.5) and the
// lazy descreen pair (16+4) - budgeted even though it only materializes on
// print retries, so an admitted context never fails its retry - plus the
// block thresholds and fixed-size reductions inside the remainder, and the
// fixed-size finder scan record and chain outcome buffers. Update it when a
// per-context device buffer is added or resized.
func gpuRouteContextDeviceBytes(capWidth, capHeight int) uint64 {
	return 37*uint64(capWidth)*uint64(capHeight) +
		gpuFinderScanBufferBytes + gpuFinderChainBufferBytes
}

func newGPURouteContext(
	device *vulki.Device,
	kernels *gpuDecodeKernels,
	ladder *gpuCanvasLadder,
	capWidth, capHeight int,
) (*gpuRouteContext, error) {
	resident, err := newGPUResidentBinarizerWithKernels(device, kernels, capWidth, capHeight)
	if err != nil {
		return nil, err
	}
	preparer, err := newGPUFinderPassPreparer(device, kernels, resident)
	if err != nil {
		_ = resident.Close()
		return nil, err
	}
	canvas, err := ladder.newRouteCanvas()
	if err != nil {
		_ = preparer.Close()
		_ = resident.Close()
		return nil, err
	}
	return &gpuRouteContext{
		capWidth: capWidth, capHeight: capHeight,
		canvas: canvas, resident: resident, preparer: preparer,
	}, nil
}

// rotate renders one level crop rotation into the context's route canvas.
// When the rotation grows the route buffer, the resident binarizer's cached
// binding sets for the old buffer are released first so the buffer can close.
func (ctx *gpuRouteContext) rotate(
	levelIndex int,
	crop image.Rectangle,
	angle float64,
) (image.Point, error) {
	size, err := ctx.canvas.ladder.rotatedRouteSize(levelIndex, crop, angle)
	if err != nil {
		return image.Point{}, err
	}
	if ctx.canvas.needsGrowth(uint64(size.X) * uint64(size.Y)) {
		if err := ctx.resident.releaseInputBindings(ctx.canvas.route); err != nil {
			return image.Point{}, err
		}
	}
	return ctx.canvas.rotate(levelIndex, crop, angle)
}

func (ctx *gpuRouteContext) Close() error {
	if ctx == nil {
		return nil
	}
	var descreenFiltered *vulki.Buffer
	if ctx.preparer != nil {
		descreenFiltered = ctx.preparer.descreenFiltered
	}
	return errors.Join(
		ctx.resident.releasePreparedBindings(descreenFiltered),
		ctx.preparer.Close(),
		ctx.resident.Close(),
		ctx.canvas.Close(),
	)
}

// gpuRouteContextPad quantizes context capacities so a context is reusable
// across the similar canvas sizes neighbouring routes request, instead of one
// exact-size context per distinct rotation.
const gpuRouteContextPad = 256

// gpuRouteContextMaxLive bounds how many contexts one workspace may hold.
// Beyond it acquisition waits for a release. It bounds host-side packed-mask
// scratch; exhausted device memory already pushes back through failed context
// creation.
const gpuRouteContextMaxLive = 32

// errGPURouteAborted reports an acquisition abandoned because the route's
// quit hook fired while it waited for a context.
var errGPURouteAborted = errors.New("jabcode: GPU route aborted before acquiring a context")

// gpuRouteContextPool hands out route contexts sized for the requesting
// route's canvas. Contexts are created on demand and kept for reuse.
//
// Admission is deterministic when the device reports its memory: a request is
// admitted iff its worst-case context fits the pool budget alone, a pure
// function of the frame and the device. Admitted requests never fall back to
// the CPU for timing reasons - when the budget is full they retire idle
// contexts (smallest first) or wait for a lease to return - so which routes
// run on the GPU does not depend on allocation order. Unadmitted requests
// fail immediately to their CPU route.
//
// Only genuine device-memory exhaustion (vulki.ErrOutOfDeviceMemory, external
// pressure from other users of the adapter) becomes backpressure instead of a
// failed route: creation runs single-flight outside the pool lock so releases
// always make progress, an out-of-memory creation retires the idle contexts
// and then latches the pool as exhausted so waiters stop re-probing the
// driver, and only a request no live context could ever satisfy surfaces the
// error and takes its CPU fallback. Any other creation failure - a lost
// device, a programming error - fails its route straight to the CPU fallback
// without destroying healthy cached contexts or masquerading as memory
// pressure.
type gpuRouteContextPool struct {
	device  *vulki.Device
	kernels *gpuDecodeKernels
	ladder  *gpuCanvasLadder
	// create is the context constructor; tests inject failures through it.
	create func(capWidth, capHeight int) (*gpuRouteContext, error)

	// budget is the device memory the pool may spend on route contexts when
	// budgetKnown; admission against it is what keeps the CPU-or-GPU backend
	// choice deterministic (see acquire). Without the device's memory size
	// the pool admits everything and relies on the out-of-memory latch.
	budget      uint64
	budgetKnown bool

	mu          sync.Mutex
	cond        *sync.Cond
	free        []*gpuRouteContext
	live        []*gpuRouteContext
	planned     uint64 // deviceBytes of live contexts plus in-flight creations
	outstanding int
	creating    bool
	exhausted   bool
	draining    bool
	closed      bool
}

func newGPURouteContextPool(
	device *vulki.Device,
	kernels *gpuDecodeKernels,
	ladder *gpuCanvasLadder,
) *gpuRouteContextPool {
	pool := &gpuRouteContextPool{device: device, kernels: kernels, ladder: ladder}
	pool.budget, pool.budgetKnown = gpuRouteContextPoolBudget(device, ladder)
	pool.cond = sync.NewCond(&pool.mu)
	return pool
}

// gpuRouteContextPoolBudget derives the pool's context budget: half of the
// device's reported local memory - the other half stays with the driver, the
// display and whatever else shares the adapter - minus the ladder's retained
// levels. A device that does not report its memory returns known=false and
// the pool falls back to probe-and-latch admission.
func gpuRouteContextPoolBudget(device *vulki.Device, ladder *gpuCanvasLadder) (uint64, bool) {
	if device == nil || ladder == nil {
		return 0, false
	}
	total := device.Info().DeviceLocalMemoryBytes
	if total == 0 {
		return 0, false
	}
	usable := total / 2
	var ladderBytes uint64
	for _, level := range ladder.levels {
		area, err := gpuCanvasArea(level.width, level.height)
		if err != nil {
			return 0, false
		}
		ladderBytes += area * 4
	}
	if ladderBytes >= usable {
		return 0, true
	}
	return usable - ladderBytes, true
}

func gpuRoutePadded(dim int) int {
	return (dim + gpuRouteContextPad - 1) / gpuRouteContextPad * gpuRouteContextPad
}

// newContext builds one route context through the injected constructor when a
// test set one, and through the real device otherwise.
func (pool *gpuRouteContextPool) newContext(capWidth, capHeight int) (*gpuRouteContext, error) {
	if pool.create != nil {
		return pool.create(capWidth, capHeight)
	}
	return newGPURouteContext(pool.device, pool.kernels, pool.ladder, capWidth, capHeight)
}

func (pool *gpuRouteContextPool) acquire(
	width, height int,
	quit func() bool,
) (*gpuRouteContext, error) {
	if pool == nil {
		return nil, fmt.Errorf("jabcode: GPU route context pool is closed")
	}
	capWidth := gpuRoutePadded(width)
	capHeight := gpuRoutePadded(height)
	need := gpuRouteContextDeviceBytes(capWidth, capHeight)
	pool.mu.Lock()
	defer pool.mu.Unlock()
	for {
		if pool.closed || pool.draining {
			return nil, fmt.Errorf("jabcode: GPU route context pool is closed")
		}
		if quit != nil && quit() {
			return nil, errGPURouteAborted
		}
		if pool.budgetKnown && need > pool.budget {
			// Deterministic admission: a request whose worst-case context
			// cannot fit the device budget at all always takes its CPU
			// route, independent of allocation timing.
			return nil, fmt.Errorf(
				"jabcode: a %dx%d GPU route context exceeds the device budget", width, height,
			)
		}
		if ctx := pool.takeFreeLocked(capWidth, capHeight); ctx != nil {
			pool.outstanding++
			return ctx, nil
		}
		creatable := len(pool.live) < gpuRouteContextMaxLive && !pool.exhausted
		if creatable && !pool.creating {
			if pool.budgetKnown && pool.planned+need > pool.budget {
				// Admitted but the pool is full: retire just enough idle
				// contexts, smallest first, or wait for a lease to return.
				// Since need fits the budget alone, releases and
				// retirements always make this request creatable
				// eventually - the wait cannot deadlock.
				if retired := pool.takeIdlesForBytesLocked(need); len(retired) > 0 {
					pool.mu.Unlock()
					for _, ctx := range retired {
						_ = ctx.Close()
					}
					pool.mu.Lock()
					continue
				}
				pool.cond.Wait()
				continue
			}
			ctx, err := pool.createUnlocked(capWidth, capHeight, need)
			if err == nil {
				pool.outstanding++
				return ctx, nil
			}
			if !errors.Is(err, vulki.ErrOutOfDeviceMemory) {
				// Not memory pressure: waiting or re-probing could not help,
				// so the route fails straight to its CPU fallback.
				return nil, err
			}
			continue
		}
		if !creatable && !pool.fitsAnyLiveLocked(capWidth, capHeight) {
			// Nothing this large exists and nothing this large can be
			// created: waiting could never succeed, so the route takes its
			// CPU fallback instead of deadlocking the search.
			return nil, fmt.Errorf(
				"jabcode: no GPU route context can hold a %dx%d canvas", width, height,
			)
		}
		pool.cond.Wait()
	}
}

// takeIdlesForBytesLocked removes idle contexts, smallest capacity first,
// until the freed budget can hold need more bytes, and returns them for the
// caller to close outside the pool lock. An empty result means the free list
// had nothing left to give; whatever was removed still frees real memory
// either way.
func (pool *gpuRouteContextPool) takeIdlesForBytesLocked(need uint64) []*gpuRouteContext {
	var retired []*gpuRouteContext
	for pool.planned+need > pool.budget && len(pool.free) > 0 {
		smallest := 0
		for index, ctx := range pool.free {
			if uint64(ctx.capWidth)*uint64(ctx.capHeight) <
				uint64(pool.free[smallest].capWidth)*uint64(pool.free[smallest].capHeight) {
				smallest = index
			}
		}
		ctx := pool.free[smallest]
		pool.free = append(pool.free[:smallest], pool.free[smallest+1:]...)
		pool.dropLiveLocked([]*gpuRouteContext{ctx})
		retired = append(retired, ctx)
	}
	return retired
}

// fitsAnyLiveLocked reports whether some existing context, free or leased,
// could serve the requested capacity once released.
func (pool *gpuRouteContextPool) fitsAnyLiveLocked(capWidth, capHeight int) bool {
	for _, ctx := range pool.live {
		if ctx.capWidth >= capWidth && ctx.capHeight >= capHeight {
			return true
		}
	}
	return false
}

// takeFreeLocked pops the smallest idle context whose capacity covers the
// request, keeping larger contexts free for the routes that need them.
func (pool *gpuRouteContextPool) takeFreeLocked(capWidth, capHeight int) *gpuRouteContext {
	best := -1
	var bestArea uint64
	for index, ctx := range pool.free {
		if ctx.capWidth < capWidth || ctx.capHeight < capHeight {
			continue
		}
		area := uint64(ctx.capWidth) * uint64(ctx.capHeight)
		if best < 0 || area < bestArea {
			best, bestArea = index, area
		}
	}
	if best < 0 {
		return nil
	}
	ctx := pool.free[best]
	pool.free = append(pool.free[:best], pool.free[best+1:]...)
	return ctx
}

// createUnlocked creates one context while temporarily dropping the pool
// lock, so in-flight releases and free-list reuse keep making progress during
// slow device allocations. The creating flag keeps creation single-flight.
// Failures are classified through the vulki sentinels: an out-of-device-memory
// failure retires the idle contexts to return their memory and retries once,
// and a second failure latches the pool as exhausted until contexts are
// actually closed or the next decode reopens the pool. Any other failure
// keeps the cached contexts and the pool state untouched - the caller fails
// its route to the CPU fallback.
func (pool *gpuRouteContextPool) createUnlocked(capWidth, capHeight int, need uint64) (*gpuRouteContext, error) {
	// The creating flag stays held across every unlocked device operation,
	// including teardown: drain and Close wait on it, so the workspace never
	// releases device resources under a mid-flight creation. The budget
	// reservation is taken here and rolled back on failure.
	pool.creating = true
	pool.planned += need
	pool.mu.Unlock()
	ctx, err := pool.newContext(capWidth, capHeight)
	pool.mu.Lock()
	if err != nil && errors.Is(err, vulki.ErrOutOfDeviceMemory) && len(pool.free) > 0 {
		idle := pool.free
		pool.free = nil
		pool.dropLiveLocked(idle)
		pool.mu.Unlock()
		for _, retired := range idle {
			_ = retired.Close()
		}
		var retryErr error
		ctx, retryErr = pool.newContext(capWidth, capHeight)
		pool.mu.Lock()
		if retryErr != nil {
			err = errors.Join(err, retryErr)
		} else {
			err = nil
		}
	}
	if err != nil {
		pool.creating = false
		pool.planned -= need
		if errors.Is(err, vulki.ErrOutOfDeviceMemory) {
			pool.exhausted = true
		}
		pool.cond.Broadcast()
		return nil, err
	}
	ctx.deviceBytes = need
	if pool.closed || pool.draining {
		pool.mu.Unlock()
		_ = ctx.Close()
		pool.mu.Lock()
		pool.creating = false
		pool.planned -= need
		pool.cond.Broadcast()
		return nil, fmt.Errorf("jabcode: GPU route context pool is closed")
	}
	pool.creating = false
	pool.exhausted = false
	pool.live = append(pool.live, ctx)
	pool.cond.Broadcast()
	return ctx, nil
}

func (pool *gpuRouteContextPool) dropLiveLocked(retired []*gpuRouteContext) {
	if len(retired) == 0 {
		return
	}
	kept := pool.live[:0]
	for _, ctx := range pool.live {
		dropped := false
		for _, gone := range retired {
			if ctx == gone {
				dropped = true
				break
			}
		}
		if !dropped {
			kept = append(kept, ctx)
		} else {
			pool.planned -= ctx.deviceBytes
		}
	}
	pool.live = kept
	// Closing contexts returned device memory; let creation probe again.
	pool.exhausted = false
}

func (pool *gpuRouteContextPool) release(ctx *gpuRouteContext) {
	if pool == nil || ctx == nil {
		return
	}
	ctx.epoch.Add(1)
	pool.mu.Lock()
	pool.outstanding--
	if pool.closed {
		pool.dropLiveLocked([]*gpuRouteContext{ctx})
		_ = ctx.Close()
	} else {
		pool.free = append(pool.free, ctx)
	}
	pool.cond.Broadcast()
	pool.mu.Unlock()
}

// drain fails new acquisitions and waits until every leased context returns
// and any in-flight creation settles. The session close path runs it so the
// cached workspace is quiescent before its lease releases and a later decode
// rebuilds the ladder over it.
func (pool *gpuRouteContextPool) drain() {
	if pool == nil {
		return
	}
	pool.mu.Lock()
	pool.draining = true
	pool.cond.Broadcast()
	for pool.outstanding > 0 || pool.creating {
		pool.cond.Wait()
	}
	pool.mu.Unlock()
}

func (pool *gpuRouteContextPool) reopen() {
	pool.mu.Lock()
	pool.draining = false
	pool.exhausted = false
	pool.mu.Unlock()
}

func (pool *gpuRouteContextPool) Close() error {
	if pool == nil {
		return nil
	}
	pool.mu.Lock()
	pool.closed = true
	pool.cond.Broadcast()
	for pool.outstanding > 0 || pool.creating {
		pool.cond.Wait()
	}
	var closeErrors []error
	for _, ctx := range pool.free {
		closeErrors = append(closeErrors, ctx.Close())
	}
	pool.free = nil
	pool.live = nil
	pool.planned = 0
	pool.mu.Unlock()
	return errors.Join(closeErrors...)
}

func (session *GPUDecodeSession) enter() (*gpuDecodeWorkspace, error) {
	if session == nil || session.closing.Load() || session.workspace == nil {
		return nil, fmt.Errorf("jabcode: GPU decode session is closed")
	}
	return session.workspace, nil
}

// DownloadLevel copies one retained pyramid level back to the host as a
// packed RGBA bitmap. The levels are read-only once the session's build
// finished, so downloads may run concurrently with route work; the CPU-side
// half-scale chain produces byte-identical pixels (the ladder parity gate),
// which is what lets a lazy CPU consumer download instead of re-halving.
func (session *GPUDecodeSession) DownloadLevel(level int) (*core.Bitmap, error) {
	workspace, err := session.enter()
	if err != nil {
		return nil, err
	}
	return workspace.ladder.DownloadLevel(level)
}

// LocateLevelFamilies runs the complete integrated finder retry ladder on one
// retained pyramid level. Every retry reuses the leased context's resident
// balanced pixels and returns only packed masks or compact reductions until
// pixels are genuinely needed downstream.
func (session *GPUDecodeSession) LocateLevelFamilies(
	level int,
	wanted FinderFamilySet,
	mode int,
	quit func() bool,
	trace *DetectorTrace,
) (*PrimaryDetector, FinderFamilySet, error) {
	workspace, err := session.enter()
	if err != nil {
		return nil, 0, err
	}
	if level < 0 || level >= len(workspace.ladder.levels) {
		return nil, 0, fmt.Errorf("jabcode: invalid GPU decode level %d", level)
	}
	retained := workspace.ladder.levels[level]
	ctx, err := workspace.contexts.acquire(retained.width, retained.height, quit)
	if err != nil {
		return nil, 0, err
	}
	defer workspace.contexts.release(ctx)
	detector, err := ctx.bufferDetector(retained.buffer, retained.width, retained.height, mode, wanted, quit, trace)
	if err != nil {
		return nil, 0, err
	}
	found, err := detector.locateFinderFamilies(wanted, ctx.preparer)
	if err != nil {
		return nil, 0, err
	}
	return finishGPUDetector(detector, found, trace)
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
	workspace, err := session.enter()
	if err != nil {
		return nil, 0, image.Point{}, err
	}
	size, err := workspace.ladder.rotatedRouteSize(level, crop, angle)
	if err != nil {
		return nil, 0, image.Point{}, err
	}
	ctx, err := workspace.contexts.acquire(size.X, size.Y, quit)
	if err != nil {
		return nil, 0, image.Point{}, err
	}
	defer workspace.contexts.release(ctx)
	size, err = ctx.rotate(level, crop, angle)
	if err != nil {
		return nil, 0, image.Point{}, err
	}
	detector, err := ctx.bufferDetector(ctx.canvas.route, size.X, size.Y, mode, wanted, quit, trace)
	if err != nil {
		return nil, 0, image.Point{}, err
	}
	found, err := detector.locateFinderFamilies(wanted, ctx.preparer)
	if err != nil {
		return nil, 0, image.Point{}, err
	}
	detector, found, err = finishGPUDetector(detector, found, trace)
	return detector, found, size, err
}

func (ctx *gpuRouteContext) bufferDetector(
	input *vulki.Buffer,
	width, height int,
	mode int,
	wanted FinderFamilySet,
	quit func() bool,
	trace *DetectorTrace,
) (*PrimaryDetector, error) {
	channels, hits, materialize, err := ctx.resident.Binarize(
		input,
		width,
		height,
		nil,
		false,
		finderScanChannelMask(
			wanted.Has(FinderFamilyCurrent),
			wanted.Has(FinderFamilyBSI) && bsiFamilyFinderEnabled,
		),
	)
	if err != nil {
		return nil, err
	}
	ctx.preparer.setInput(width, height, trace != nil)
	balanced := &core.Bitmap{
		Width: width, Height: height, Channels: 4,
	}
	detector := &PrimaryDetector{
		BM: balanced, Ch: channels, Mode: mode, Quit: quit, Trace: trace,
		rowHits: hits, materializeChannels: materialize,
	}
	leaseEpoch := ctx.epoch.Load()
	detector.materializeBitmap = func() error {
		// Materialization normally happens while the route still holds the
		// context; the epoch guard keeps a stale detector from reading pixels
		// a later route overwrote.
		if ctx.epoch.Load() != leaseEpoch {
			return fmt.Errorf("jabcode: GPU route context was released before materialization")
		}
		downloaded, err := ctx.resident.DownloadBalanced(width, height)
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
	// A located success hands its channels downstream; a failed lazy mask
	// expansion surfaces here instead of as absent pixels later.
	if (found != 0 || trace != nil) && !detector.ensureChannels() {
		if detector.materializeChanErr != nil {
			return nil, 0, detector.materializeChanErr
		}
		return nil, 0, fmt.Errorf("jabcode: materialize resident GPU mask channels")
	}
	return detector, found, nil
}

// Close waits for the in-flight routes to release their contexts, then
// releases the workspace. Automatic sessions cache it for another same-sized
// decode; borrowed-device sessions release their buffers and pipelines.
func (session *GPUDecodeSession) Close() error {
	if session == nil || session.closing.Swap(true) {
		return nil
	}
	if session.workspace != nil {
		session.workspace.contexts.drain()
	}
	if session.release != nil {
		return session.release()
	}
	return nil
}
