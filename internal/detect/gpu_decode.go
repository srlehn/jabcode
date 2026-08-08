//go:build !js

package detect

import (
	"errors"
	"fmt"
	"image"
	"sync"
	"sync/atomic"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/phaseprobe"
)

var automaticGPUDecode = newGPUDecodeRuntime(automaticGPUDevices)

type gpuDecodeRuntime struct {
	devices *gpuDeviceCache

	workspaceMu sync.Mutex
	workspace   *gpuDecodeWorkspace
	// kernels live as long as the process-wide device: workspaces of any size
	// share the compiled pipelines instead of recompiling WGSL per resize.
	kernels *gpuDecodeKernels

	// warmMu guards warming, which is non-nil exactly while a warm-up is in
	// flight. begin waits on it so a decode joins the preparation it started
	// itself; without that the TryLock below would read its own warm-up as
	// another decode's lease and drop this read to the CPU.
	warmMu  sync.Mutex
	warming chan struct{}
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

	// ops gates every public session method between entry and return, so
	// Close quiesces the whole session rather than only leased route
	// contexts: a method can pass the closing check and be descheduled
	// before it acquires a context, and tearing down or reusing the
	// workspace under such an operation would race its ladder and pool
	// reads. enterMu makes the closed check and the op registration one
	// atomic step against Close.
	enterMu sync.Mutex
	ops     sync.WaitGroup
}

// NewAutomaticGPUDecodeSession starts a resident decode workspace when the
// image crosses the measured GPU threshold and a measured discrete Vulkan
// adapter is available. A nil session means the caller should use the CPU path.
func NewAutomaticGPUDecodeSession(base *core.Bitmap, levelCount int) (*GPUDecodeSession, error) {
	return automaticGPUDecode.begin(base, levelCount)
}

// ReplaceBase refreshes the retained image pyramid without rebuilding its
// sized workspace. A stream reuses this seam for coherent frames with stable
// geometry; the upload and every halving pass overwrite all frame-owned
// pixels before another route can acquire the workspace.
func (session *GPUDecodeSession) ReplaceBase(base *core.Bitmap) error {
	workspace, err := session.enter()
	if err != nil {
		return err
	}
	defer session.leave()
	if base == nil || base.Width != workspace.width || base.Height != workspace.height {
		return fmt.Errorf("jabcode: GPU frame geometry changed from %dx%d", workspace.width, workspace.height)
	}
	workspace.contexts.beginDrain()
	workspace.contexts.drain()
	err = workspace.ladder.UploadAndBuild(base)
	workspace.contexts.reopen()
	return err
}

// warm builds everything a decode route needs that depends on the frame
// geometry alone and returns immediately. It is one at a time and idempotent
// against begin: both publish through the same cached workspace, so a warm-up
// that loses the race has nothing left to do.
func (runtime *gpuDecodeRuntime) warm(width, height, levelCount int) {
	if runtime == nil || runtime.devices == nil || levelCount <= 0 ||
		gpuRoutesDisabled.Load() || !automaticGPUWorkload(width, height) {
		return
	}
	runtime.warmMu.Lock()
	if runtime.warming != nil {
		runtime.warmMu.Unlock()
		return
	}
	done := make(chan struct{})
	runtime.warming = done
	runtime.warmMu.Unlock()
	go func() {
		phaseprobe.Markf("warm.start", "width=%d height=%d levels=%d", width, height, levelCount)
		defer func() {
			runtime.warmMu.Lock()
			runtime.warming = nil
			runtime.warmMu.Unlock()
			close(done)
			phaseprobe.Mark("warm.end")
		}()
		runtime.prepare(width, height, levelCount)
	}()
}

// awaitWarm blocks until any in-flight warm-up has published its workspace.
// Only the read that started the warm-up normally reaches this, and it has
// already spent the image decode overlapping with it.
func (runtime *gpuDecodeRuntime) awaitWarm() {
	if runtime == nil {
		return
	}
	runtime.warmMu.Lock()
	done := runtime.warming
	runtime.warmMu.Unlock()
	if done != nil {
		<-done
	}
}

// prepare is begin without the pixels: the device, the shared kernel set, the
// asynchronous finder chain compilation and the size-matched workspace. Every
// failure is silent because a warm-up owes nothing - begin does the same work
// itself and reports properly if it still fails.
func (runtime *gpuDecodeRuntime) prepare(width, height, levelCount int) {
	phaseprobe.Mark("prepare.start")
	defer phaseprobe.Mark("prepare.end")
	device, err := runtime.devices.deviceFor(width, height)
	phaseprobe.Markf("prepare.device.ready", "error=%t available=%t", err != nil, device != nil)
	if err != nil || device == nil {
		return
	}
	// A busy workspace means a decode is already holding the lease, so its
	// workspace exists and there is nothing to prepare. Waiting for it would
	// hold up the read that started this warm-up for the whole of that decode.
	if !runtime.workspaceMu.TryLock() {
		return
	}
	defer runtime.workspaceMu.Unlock()
	if runtime.kernels == nil {
		runtime.kernels = newGPUDecodeKernels(device)
	}
	phaseprobe.Mark("prepare.kernels.start")
	runtime.kernels.warmFinderChains()
	phaseprobe.Mark("prepare.kernels.end")
	if runtime.workspace != nil && runtime.workspace.matches(width, height, levelCount) {
		return
	}
	if retired := runtime.workspace; retired != nil {
		runtime.workspace = nil
		if err := retired.Close(); err != nil {
			return
		}
	}
	phaseprobe.Mark("prepare.workspace.start")
	workspace, err := newGPUDecodeWorkspace(device, runtime.kernels, width, height, levelCount)
	phaseprobe.Markf("prepare.workspace.end", "error=%t", err != nil)
	if err != nil {
		return
	}
	runtime.workspace = workspace
	phaseprobe.Mark("prepare.contexts.start")
	warmRouteContexts(workspace)
	phaseprobe.Mark("prepare.contexts.end")
	phaseprobe.Mark("prepare.staging.start")
	err = workspace.ladder.ReserveUpload()
	phaseprobe.Markf("prepare.staging.end", "error=%t", err != nil)
}

// warmRouteContexts allocates one route context per ladder level so the first
// decode does not. Every pyramid level is read concurrently and each needs its
// own context, so on a first read the allocations land together and their cost
// falls on whichever level the read is waiting for. The sizes come from the
// ladder, which the warm-up has already built, so this needs no pixels either.
//
// Every lease is held until they all exist. Releasing each one before asking
// for the next warms exactly one context and no more: the pool satisfies a
// request from any free context large enough, so the full-size one it starts
// with answers every smaller level in turn and none of the others is ever
// created.
//
// Holding leases is also why this must never park. Each level fits the budget
// alone or the pool rejects it outright, but all of them together need not, and
// then the pool waits for a lease to come back - leases this function is itself
// holding and does not return until it finishes. That wait would never end, and
// the read that started the warm-up joins it. So warming takes the
// non-blocking path and settles for the contexts it got.
func warmRouteContexts(workspace *gpuDecodeWorkspace) {
	held := make([]*gpuRouteContext, 0, len(workspace.ladder.levels))
	defer func() {
		for _, ctx := range held {
			workspace.contexts.release(ctx)
		}
	}()
	for _, level := range workspace.ladder.levels {
		ctx, err := workspace.contexts.acquireWaiting(level.width, level.height, nil, false)
		if err != nil {
			return
		}
		held = append(held, ctx)
	}
}

func (runtime *gpuDecodeRuntime) begin(
	base *core.Bitmap,
	levelCount int,
) (*GPUDecodeSession, error) {
	phaseprobe.Mark("session.begin.start")
	defer phaseprobe.Mark("session.begin.return")
	if runtime == nil || runtime.devices == nil || base == nil ||
		gpuRoutesDisabled.Load() || !automaticGPUWorkload(base.Width, base.Height) {
		return nil, nil
	}
	phaseprobe.Mark("session.await_warm.start")
	runtime.awaitWarm()
	phaseprobe.Mark("session.await_warm.end")
	device, err := runtime.devices.deviceFor(base.Width, base.Height)
	phaseprobe.Markf("session.device.ready", "error=%t available=%t", err != nil, device != nil)
	if err != nil || device == nil {
		return nil, nil
	}
	if !runtime.workspaceMu.TryLock() {
		return nil, nil
	}
	phaseprobe.Mark("session.workspace.acquired")
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
	phaseprobe.Mark("session.upload.start")
	if err := runtime.workspace.ladder.UploadAndBuild(base); err != nil {
		phaseprobe.Markf("session.upload.end", "error=true")
		return nil, err
	}
	phaseprobe.Markf("session.upload.end", "error=false")
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
	return newGPUDecodeSessionWithDevice(device, base, levelCount, false)
}

// NewGPUDecodeSessionWithDeviceScanOnly creates a borrowed-device session whose
// route contexts never replay on the device, so the bit-identical CPU per-hit
// chain and pitch fold classify every hit. It exists to exercise those twins
// deterministically, and as the degraded mode every session already falls back
// to before the chain kernels finish compiling; it is slower than the default
// on everything but a small symbol with large modules.
func NewGPUDecodeSessionWithDeviceScanOnly(
	device *vulki.Device,
	base *core.Bitmap,
	levelCount int,
) (*GPUDecodeSession, error) {
	return newGPUDecodeSessionWithDevice(device, base, levelCount, true)
}

func newGPUDecodeSessionWithDevice(
	device *vulki.Device,
	base *core.Bitmap,
	levelCount int,
	scanOnly bool,
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
	workspace.contexts.scanOnly = scanOnly
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
// one resident binarizer instance and one finder-pass preparer. Routes share
// only the device, the retained pyramid levels (read-only after the build) and
// the compiled kernels. The pool hands a context to one route at a time.
type gpuRouteContext struct {
	capWidth  int
	capHeight int
	// deviceBytes is the pool's budgeted device-memory cost of this context,
	// fixed at creation (see gpuRouteContextDeviceBytes).
	deviceBytes uint64
	// hostBytes is the pool's budgeted host-scratch cost of this context,
	// fixed at creation (see gpuRouteContextHostBytes).
	hostBytes uint64
	resident  *gpuResidentBinarizer
	preparer  *gpuFinderPassPreparer

	// epoch counts pool releases. Detector closures that materialize resident
	// pixels capture the epoch at lease time and refuse to touch buffers a
	// later route may have overwritten.
	epoch atomic.Uint64

	// retainedExtraBytes accumulates lazy directional allocations and retained
	// overflow growth since the last release. The route thread adds to it
	// lock-free while holding the resident binarizer lock, which must never wait
	// on the pool lock; release folds it into deviceBytes and the pool's planned
	// total so cached buffers stay budgeted.
	retainedExtraBytes atomic.Uint64
}

// gpuRouteContextFixedBytes sums the fixed-size device buffers every route
// context holds: the RGB histogram and bounds reductions, the binarizer,
// scan, chain, canvas, finder-average, pitch, descreen and pitch-lag
// parameter buffers, the finder-average partials, the pitch line sums and
// means, the module grid and its sampler parameters, the edge-walk counts and
// their parameters, the row chain's per-channel fold and its compacted
// candidate regions, the alignment cell table and its search parameters, the
// preserved masks of a located pass, the alignment search's per-tile scratch,
// the LDPC corrector's parity rows, codeword and output, the payload chain's
// data map, deinterleaving permutation and parameters, the initial scan
// record buffer and the initial chain outcome buffer.
const gpuRouteContextFixedBytes = gpuRGBHistogramBytes + gpuRGBBoundsBytes +
	gpuBinarizerParamsSize + gpuFinderScanBufferBytes +
	gpuFinderScanParamsSize + gpuFinderChainBufferBytes +
	gpuFinderChainParamsSize + gpuCanvasParamsSize +
	gpuFinderAverageParamsSize + gpuFinderAveragePartialSize +
	gpuPitchParamsSize + gpuDescreenParamsSize + gpuPitchLagParamsSize +
	2*gpuPitchLagLineBytes +
	gpuSampleResultWords*4 + gpuSampleParamWords*4 +
	gpuModuleCountResultWords*4 + gpuModuleCountParamWords*4 +
	gpuAlignMaxCells*gpuAlignCellWords*4 + gpuAlignParamWords*4 +
	gpuAlignMaxCells*gpuAlignTiles*gpuAlignCellWords*4 +
	gpuLDPCRetainedBytes + gpuPayloadRetainedBytes + gpuMetadataRetainedBytes +
	gpuFinderFoldRetainedBytes + gpuRowSummaryBytes + gpuRowCompactBytes

// gpuRouteContextBufferCount counts the distinct device buffers a route
// context can allocate; each may cost up to one alignment rounding of driver
// memory beyond its requested size.
const gpuRouteContextBufferCount = 51

// gpuRouteContextAllocationAllowance covers per-buffer allocation-alignment
// rounding in the driver, at the conventional 256-byte storage alignment.
const gpuRouteContextAllocationAllowance = gpuRouteContextBufferCount * 256

// gpuRouteContextDeviceBytes bounds the device memory one route context of
// the given capacity can hold before opportunistic directional allocation and
// overflow growth. It sums every mandatory buffer and the retry chains whose
// admission must be guaranteed: the balanced image
// (4 B/px), the raw and final masks (4+4), the
// packed masks (~0.5), the lazy descreen pair (16+4), the block thresholds,
// the pitch sample and centered-sample buffers, the per-axis autocorrelation
// output, and the fixed parameter, reduction, scan and chain buffers, plus a
// per-buffer alignment allowance.
// The lazy descreen and pitch-lag chains are budgeted even though they only
// materialize on their retry tiers, so an admitted context never fails those
// retries. Directional records and outcomes and overflow growth of the row
// buffers stay outside this estimate: they are opportunistic and degrade to
// the CPU walk when the device cannot afford them. Every retained allocation
// is charged to the pool when the context returns to the free list (see
// release), so cached buffers cannot silently exceed the pool budget. Update
// this when a
// per-context device buffer is added or resized;
// TestGPURouteContextDeviceBytesCoversAllocations fails when a real context
// allocates beyond it.
func gpuRouteContextDeviceBytes(capWidth, capHeight int) uint64 {
	area := uint64(capWidth) * uint64(capHeight)
	blocks := uint64((capWidth+binMinBlock-1)/binMinBlock) *
		uint64((capHeight+binMinBlock-1)/binMinBlock)
	pitchSamples := uint64(gpuPitchSampleCount(capWidth, capHeight))
	pitchLags := uint64(max(2, min(capWidth, capHeight)/8) + 1)
	// Two packed-mask buffers per context: the pass's own, and the preserved
	// copy a located pass keeps for a reader that may never come.
	return 32*area + 2*((area+7)/8*4) +
		blocks*gpuThresholdCellSize +
		pitchSamples*12 +
		pitchLags*16 +
		gpuRouteContextFixedBytes + gpuRouteContextAllocationAllowance
}

func newGPURouteContext(
	device *vulki.Device,
	kernels *gpuDecodeKernels,
	ladder *gpuCanvasLadder,
	capWidth, capHeight int,
	scanOnly bool,
) (*gpuRouteContext, error) {
	resident, err := newGPUResidentBinarizerWithKernels(device, kernels, capWidth, capHeight)
	if err != nil {
		return nil, err
	}
	resident.binarizer.scanOnly = scanOnly
	preparer, err := newGPUFinderPassPreparer(device, kernels, resident)
	if err != nil {
		_ = resident.Close()
		return nil, err
	}
	ctx := &gpuRouteContext{
		capWidth: capWidth, capHeight: capHeight,
		resident: resident, preparer: preparer,
	}
	resident.binarizer.onRetainedAllocation = func(delta uint64) {
		ctx.retainedExtraBytes.Add(delta)
	}
	return ctx, nil
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
	)
}

// gpuRouteContextPad quantizes context capacities so a context is reusable
// across the similar canvas sizes neighbouring routes request, instead of one
// exact-size context per distinct level.
const gpuRouteContextPad = 256

// gpuRouteHostScratchFrames bounds the pool's host-side scratch as a multiple
// of the base frame's own packed-mask size, so the bound scales with the work
// rather than with the machine.
//
// The resource being bounded is host RAM: each context holds a packed-mask
// staging buffer its resident binarizer downloads into, and nothing else of
// consequence on the host. Device memory is bounded separately, by bytes, in
// gpuRouteContextPoolBudget.
//
// This replaced a bound on the *number* of live contexts, which was the wrong
// unit and the reason the ladder's largest routes were being pushed onto the
// CPU. Per-context host cost tracks canvas area, and the canvases in one read
// span the coarse pyramid levels and the full-resolution ones - a range of two
// orders of magnitude. A count therefore bounded
// nothing in particular, and it bound the wrong end: the small canvases ask
// first and ask often, so they filled the slots and starved the large ones.
// Counting bytes cannot invert like that, because a context's claim on the
// budget is its actual cost.
//
// The multiple is chosen rather than calibrated, but it is a ratio against the
// frame, so it means the same thing on every adapter and at every capture
// size. It is deliberately generous: the device budget and the out-of-memory
// latch are the intended backstops, and this exists so a pathological ladder
// cannot pin unbounded host memory.
const gpuRouteHostScratchFrames = 64

// gpuRouteContextHostBytes is the host scratch one context of this capacity
// holds: the packed-mask staging buffer, four bytes per eight canvas pixels.
func gpuRouteContextHostBytes(capWidth, capHeight int) uint64 {
	return (uint64(capWidth)*uint64(capHeight) + 7) / 8 * 4
}

// errGPURouteAborted reports an acquisition abandoned because the route's
// quit hook fired while it waited for a context.
var errGPURouteAborted = errors.New("jabcode: GPU route aborted before acquiring a context")

// gpuRouteContextPool hands out route contexts sized for the requesting
// route's canvas. Contexts are created on demand and kept for reuse.
//
// Admission is deterministic when the device reports its memory: a request is
// admitted iff its worst-case context fits the pool budget alone, a pure
// function of the frame and the device. An admitted request under a full byte
// budget retires idle contexts (smallest first) or waits for a lease to
// return. At the live-context cap it behaves the same way: a request no live
// context can cover retires the smallest idle context and builds the size it
// needs, because the cap bounds host scratch rather than device memory, so an
// unleased slot is recoverable at no cost.
//
// Only when every slot is leased does the request fail to its CPU route
// instead of waiting, and only that case is what the measurement behind this
// rule covers: parking routes on a lease measurably doubled adverse-capture
// wall time on the dev machine. Retiring an idle context parks nothing, and
// conflating the two is what previously sent a read's largest routes to the
// CPU while the adapter idled - the coarse levels and region crops fill the
// slots with small contexts, and every full-resolution rotation then finds
// none it can use. Unadmitted requests fail immediately to their CPU route.
//
// Backend placement in the all-leased corner can still depend on arrival
// order; the CPU route returns identical results, so no decode outcome does.
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

	// scanOnly keeps this pool's contexts out of the device replay tiers,
	// the per-hit chains and the resident pitch fold, so the CPU twins run
	// instead (see gpuBinarizer.scanOnly for why that is the exception and
	// not the rule). Set before the first acquisition, never mutated
	// afterwards.
	scanOnly bool

	// budget is the device memory the pool may spend on route contexts when
	// budgetKnown; admission against it is what keeps the CPU-or-GPU backend
	// choice deterministic (see acquire). Without the device's memory size
	// the pool admits everything and relies on the out-of-memory latch.
	budget      uint64
	budgetKnown bool

	// hostBudget bounds the packed-mask scratch live contexts may hold, in
	// bytes (see gpuRouteHostScratchFrames). Zero disables the bound and is
	// only reachable without a canvas ladder to size it against.
	hostBudget uint64

	mu          sync.Mutex
	cond        *sync.Cond
	free        []*gpuRouteContext
	live        []*gpuRouteContext
	planned     uint64 // deviceBytes of live contexts plus in-flight creations
	hostPlanned uint64 // hostBytes of live contexts plus in-flight creations
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
	pool.hostBudget = gpuRouteHostScratchBudget(ladder)
	pool.cond = sync.NewCond(&pool.mu)
	return pool
}

// gpuRouteHostScratchBudget sizes the pool's host-scratch bound against the
// ladder's largest level, so a small capture cannot pin the host memory a
// large one is entitled to. Without a ladder to measure there is nothing to
// scale against and the bound is left off.
func gpuRouteHostScratchBudget(ladder *gpuCanvasLadder) uint64 {
	if ladder == nil {
		return 0
	}
	var largest uint64
	for _, level := range ladder.levels {
		if bytes := gpuRouteContextHostBytes(level.width, level.height); bytes > largest {
			largest = bytes
		}
	}
	return largest * gpuRouteHostScratchFrames
}

// gpuRouteContextPoolBudget derives the pool's context budget: the device's
// reported local memory minus the ladder's retained levels. A device that does
// not report its memory returns known=false and the pool falls back to
// probe-and-latch admission.
//
// The budget is an admission bound, not a reservation, and it deliberately
// claims the whole adapter rather than a share of it. Holding back half was a
// guess at what the driver and the display need, and it cost real headroom:
// contexts the pool could have kept were retired to stay under it. Genuine
// contention is what the out-of-memory path is for - a failed creation retires
// every idle context and latches the pool as backpressure - so the driver
// gets to arbitrate instead of a fixed fraction chosen up front.
func gpuRouteContextPoolBudget(device *vulki.Device, ladder *gpuCanvasLadder) (uint64, bool) {
	if device == nil || ladder == nil {
		return 0, false
	}
	total := device.Info().DeviceLocalMemoryBytes
	if total == 0 {
		return 0, false
	}
	usable := total
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
	return newGPURouteContext(pool.device, pool.kernels, pool.ladder, capWidth, capHeight, pool.scanOnly)
}

func (pool *gpuRouteContextPool) acquire(
	width, height int,
	quit func() bool,
) (*gpuRouteContext, error) {
	return pool.acquireWaiting(width, height, quit, true)
}

// errGPURoutePoolBusy reports that a request could only be served by waiting
// for a lease to come back.
var errGPURoutePoolBusy = errors.New("jabcode: every GPU route context is leased")

// acquireWaiting is acquire with the parking behaviour made explicit. A route
// waits, because a context will come back and its alternative is the CPU path
// for a whole level. A caller that already holds leases must not: the only
// leases that could satisfy it may be its own, and then the wait is forever.
func (pool *gpuRouteContextPool) acquireWaiting(
	width, height int,
	quit func() bool,
	wait bool,
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
		hostNeed := gpuRouteContextHostBytes(capWidth, capHeight)
		creatable := pool.hostFitsLocked(hostNeed) && !pool.exhausted
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
				if !wait {
					return nil, errGPURoutePoolBusy
				}
				pool.cond.Wait()
				continue
			}
			ctx, err := pool.createUnlocked(capWidth, capHeight, need, hostNeed)
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
			// Nothing live can cover this request and the host-scratch budget
			// or the exhaustion latch forbids creating more. Scratch held by
			// an idle context that cannot serve this canvas is recoverable:
			// retire idle contexts, smallest first, and create a correctly
			// sized one. This is the same trade the device-budget branch above
			// makes, and it costs no wait - the contexts retired here are
			// unleased by definition.
			//
			// Without it the route ladder loses exactly its most expensive
			// routes to the CPU. The coarse levels and the region crops claim
			// the budget first with small canvases, and then every
			// full-resolution rotation finds nothing it can use and silently
			// takes the CPU path while the adapter idles. That failure is
			// invisible to every behaviour gate, because the CPU route returns
			// the same bytes: only the CPU-time column shows it.
			if !pool.exhausted {
				if retired := pool.takeIdlesForHostLocked(hostNeed); len(retired) > 0 {
					pool.mu.Unlock()
					for _, ctx := range retired {
						_ = ctx.Close()
					}
					pool.mu.Lock()
					continue
				}
			}
			// Every context is leased, so no retirement can free scratch.
			// Waiting for a lease here stalls the route ladder: the dev-machine
			// adverse-capture A/B measured about a 2x wall regression from
			// parking these routes instead of letting them run their CPU
			// fallback immediately. Backend placement in this corner can
			// depend on arrival order, but the CPU route returns identical
			// results.
			return nil, fmt.Errorf(
				"jabcode: no GPU route context can hold a %dx%d canvas", width, height,
			)
		}
		if !wait {
			return nil, errGPURoutePoolBusy
		}
		pool.cond.Wait()
	}
}

// takeSmallestFreeLocked removes and returns the smallest-capacity idle
// context, or nil when every context is leased. Retirement is deterministic,
// smallest first, so larger cached contexts survive the longest.
func (pool *gpuRouteContextPool) takeSmallestFreeLocked() *gpuRouteContext {
	if len(pool.free) == 0 {
		return nil
	}
	smallest := 0
	for index, ctx := range pool.free {
		if uint64(ctx.capWidth)*uint64(ctx.capHeight) <
			uint64(pool.free[smallest].capWidth)*uint64(pool.free[smallest].capHeight) {
			smallest = index
		}
	}
	ctx := pool.free[smallest]
	pool.free = append(pool.free[:smallest], pool.free[smallest+1:]...)
	return ctx
}

// takeIdlesForBytesLocked removes idle contexts, smallest capacity first,
// until the freed budget can hold need more bytes, and returns them for the
// caller to close outside the pool lock. An empty result means the free list
// had nothing left to give; whatever was removed still frees real memory
// either way.
func (pool *gpuRouteContextPool) takeIdlesForBytesLocked(need uint64) []*gpuRouteContext {
	var retired []*gpuRouteContext
	for pool.planned+need > pool.budget && len(pool.free) > 0 {
		ctx := pool.takeSmallestFreeLocked()
		pool.dropLiveLocked([]*gpuRouteContext{ctx})
		retired = append(retired, ctx)
	}
	return retired
}

// hostFitsLocked reports whether hostNeed more bytes of packed-mask scratch
// fit the pool's host budget. A zero budget means no ladder was available to
// scale one against, and the bound is off.
func (pool *gpuRouteContextPool) hostFitsLocked(hostNeed uint64) bool {
	return pool.hostBudget == 0 || pool.hostPlanned+hostNeed <= pool.hostBudget
}

// takeIdlesForHostLocked removes idle contexts, smallest capacity first, until
// the host budget can hold hostNeed more bytes, and returns them for the caller
// to close outside the pool lock. An empty result means every context is
// leased, so no retirement could free scratch.
func (pool *gpuRouteContextPool) takeIdlesForHostLocked(hostNeed uint64) []*gpuRouteContext {
	var retired []*gpuRouteContext
	for !pool.hostFitsLocked(hostNeed) && len(pool.free) > 0 {
		ctx := pool.takeSmallestFreeLocked()
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
func (pool *gpuRouteContextPool) createUnlocked(capWidth, capHeight int, need, hostNeed uint64) (*gpuRouteContext, error) {
	// The creating flag stays held across every unlocked device operation,
	// including teardown: drain and Close wait on it, so the workspace never
	// releases device resources under a mid-flight creation. The budget
	// reservation is taken here and rolled back on failure.
	pool.creating = true
	pool.planned += need
	pool.hostPlanned += hostNeed
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
		pool.hostPlanned -= hostNeed
		if errors.Is(err, vulki.ErrOutOfDeviceMemory) {
			pool.exhausted = true
		}
		pool.cond.Broadcast()
		return nil, err
	}
	ctx.deviceBytes = need
	ctx.hostBytes = hostNeed
	if pool.closed || pool.draining {
		pool.mu.Unlock()
		_ = ctx.Close()
		pool.mu.Lock()
		pool.creating = false
		pool.planned -= need
		pool.hostPlanned -= hostNeed
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
			pool.hostPlanned -= ctx.hostBytes
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
	// Retained overflow growth becomes budgeted memory once the context is
	// cached for reuse; folding it here keeps planned equal to the sum of
	// the live contexts' deviceBytes.
	if grown := ctx.retainedExtraBytes.Swap(0); grown > 0 {
		ctx.deviceBytes += grown
		pool.planned += grown
	}
	if pool.closed {
		pool.dropLiveLocked([]*gpuRouteContext{ctx})
		_ = ctx.Close()
	} else {
		pool.free = append(pool.free, ctx)
	}
	pool.cond.Broadcast()
	pool.mu.Unlock()
}

// beginDrain fails new and waiting acquisitions without waiting for leases.
// The session close path signals it before waiting for its registered
// operations, so an operation blocked inside acquire wakes into the closed
// error instead of deadlocking the close.
func (pool *gpuRouteContextPool) beginDrain() {
	if pool == nil {
		return
	}
	pool.mu.Lock()
	pool.draining = true
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

// enter registers one public session operation and returns the workspace.
// Every caller must pair a successful enter with a deferred leave, so Close
// can wait for operations that have passed this gate but not yet acquired a
// route context.
func (session *GPUDecodeSession) enter() (*gpuDecodeWorkspace, error) {
	if session == nil {
		return nil, fmt.Errorf("jabcode: GPU decode session is closed")
	}
	session.enterMu.Lock()
	defer session.enterMu.Unlock()
	if session.closing.Load() || session.workspace == nil {
		return nil, fmt.Errorf("jabcode: GPU decode session is closed")
	}
	session.ops.Add(1)
	return session.workspace, nil
}

func (session *GPUDecodeSession) leave() {
	session.ops.Done()
}

// WaitReplayKernels blocks until every kernel the replay policy switches on is
// compiled and usable, or returns the first compilation error.
//
// Sessions warm these in a goroutine nothing waits on, because a cold driver
// pipeline cache can take minutes on the largest modules this package submits.
// That makes replay a race: a pass that starts before the warm finishes
// silently takes the CPU twin instead. Anything comparing the two routes has to
// settle that race first, and by waiting on the compile rather than by
// sleeping, since a sleep long enough to be safe on a cold cache is one that
// also hides what it is waiting for.
//
// It waits for the pitch-lag kernels as well as the finder chains because
// scanOnly gates both, and waiting on only one leaves the other free to switch
// mechanism mid-measurement. Compilation is per-kernel idempotent, so joining
// the warm costs nothing once it has already run.
func (session *GPUDecodeSession) WaitReplayKernels() error {
	workspace, err := session.enter()
	if err != nil {
		return err
	}
	defer session.leave()
	if err := workspace.kernels.compileFinderChains(); err != nil {
		return err
	}
	if err := workspace.kernels.compileDirectionalFinderChain(); err != nil {
		return err
	}
	return workspace.kernels.compilePitchLag()
}

// DownloadLevel copies one retained pyramid level back to the host as a
// packed RGBA bitmap. The levels are read-only once the session's build
// finished, so downloads may run concurrently with route work.
//
// No decode route calls it, and none may: a level is the device route's own
// working set, and moving one to the host to spare a CPU route its halving
// chain was the largest single line in the transfer census. It survives as the
// ladder's accessor for the parity and close-race gates.
func (session *GPUDecodeSession) DownloadLevel(level int) (*core.Bitmap, error) {
	workspace, err := session.enter()
	if err != nil {
		return nil, err
	}
	defer session.leave()
	return workspace.ladder.DownloadLevel(level)
}

// LocateLevelFamilies runs the complete integrated finder retry ladder on one
// retained pyramid level. Every retry reuses the leased context's resident
// balanced pixels and returns only packed masks or compact reductions until
// pixels are genuinely needed downstream.
//
// **The returned release must be called when the caller is done with the
// detector**, and it is never nil once the detector is. The lease used to end
// here, which forced every level that located anything to download its whole
// balanced image before returning - tens of megabytes per level, on every
// level, when only the level that goes on to sample ever reads them. Holding
// the lease across the caller's decode is what makes that download lazy. It
// costs no extra contexts: the pool is warmed with one per pyramid level, and a
// level only ever holds its own.
func (session *GPUDecodeSession) LocateLevelFamilies(
	level int,
	wanted FinderFamilySet,
	mode int,
	quit func() bool,
	trace *DetectorTrace,
) (detector *PrimaryDetector, found FinderFamilySet, release func(), err error) {
	phaseprobe.Markf("level.enter", "level=%d", level)
	workspace, err := session.enter()
	if err != nil {
		phaseprobe.Markf("level.return", "level=%d", level)
		return nil, 0, nil, err
	}
	held := false
	defer func() {
		if !held {
			session.leave()
			phaseprobe.Markf("level.return", "level=%d", level)
		}
	}()
	if level < 0 || level >= len(workspace.ladder.levels) {
		return nil, 0, nil, fmt.Errorf("jabcode: invalid GPU decode level %d", level)
	}
	retained := workspace.ladder.levels[level]
	phaseprobe.Markf("level.context.start", "level=%d", level)
	ctx, err := workspace.contexts.acquire(retained.width, retained.height, quit)
	phaseprobe.Markf("level.context.end", "level=%d error=%t", level, err != nil)
	if err != nil {
		return nil, 0, nil, err
	}
	defer func() {
		if !held {
			workspace.contexts.release(ctx)
		}
	}()
	phaseprobe.Markf("level.binarize.start", "level=%d", level)
	detector, err = ctx.bufferDetector(retained.buffer, retained.width, retained.height, mode, wanted, quit, trace)
	phaseprobe.Markf("level.binarize.end", "level=%d error=%t", level, err != nil)
	if err != nil {
		return nil, 0, nil, err
	}
	phaseprobe.Markf("level.locate.start", "level=%d", level)
	found, err = detector.locateFinderFamilies(wanted, ctx.preparer)
	phaseprobe.Markf("level.locate.end",
		"level=%d error=%t directional_chain_hits=%d directional_device_sweeps=%d "+
			"dirchain_ready=%t sweep_ms=%.1f host_ms=%.1f",
		level, err != nil, detector.directionalDeviceChainHits,
		detector.directionalDeviceSweeps, workspace.kernels.directionalFinderChainReady(),
		float64(detector.directionalSweepNanos)/1e6, float64(detector.directionalHostNanos)/1e6)
	if err != nil {
		return nil, 0, nil, err
	}
	detector, found, err = finishGPUDetector(detector, found, trace)
	if err != nil || detector == nil {
		return nil, 0, nil, err
	}
	held = true
	var once sync.Once
	return detector, found, func() {
		once.Do(func() {
			workspace.contexts.release(ctx)
			session.leave()
			phaseprobe.Markf("level.return", "level=%d", level)
		})
	}, nil
}

func (ctx *gpuRouteContext) bufferDetector(
	input *vulki.Buffer,
	width, height int,
	mode int,
	wanted FinderFamilySet,
	quit func() bool,
	trace *DetectorTrace,
) (*PrimaryDetector, error) {
	ctx.resident.SetRowStride(finderRowStride(height, mode))
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
	detector.sampleGrid = func(
		pt core.Perspective, side image.Point, delta [3]core.PointF,
	) (*core.Bitmap, error) {
		if ctx.epoch.Load() != leaseEpoch {
			return nil, fmt.Errorf("jabcode: GPU route context was released before sampling")
		}
		return ctx.resident.SampleSymbol(width, height, pt, side, delta)
	}
	detector.sampleBlocks = func(
		side image.Point, blocks []AlignmentBlock,
	) (*core.Bitmap, error) {
		if ctx.epoch.Load() != leaseEpoch {
			return nil, fmt.Errorf("jabcode: GPU route context was released before the alignment resample")
		}
		return ctx.resident.SampleBlocks(width, height, side, blocks)
	}
	detector.correctPayload = gpuPayloadCorrector{resident: ctx.resident, epoch: &ctx.epoch, lease: leaseEpoch}
	detector.walkMetadata = gpuMetadataWalker{resident: ctx.resident, epoch: &ctx.epoch, lease: leaseEpoch}
	detector.materializeGrid = gpuGridMaterializer{resident: ctx.resident, epoch: &ctx.epoch, lease: leaseEpoch}
	detector.walkModuleCounts = func(fps []FinderPattern) ([4]int, error) {
		if ctx.epoch.Load() != leaseEpoch {
			return [4]int{}, fmt.Errorf("jabcode: GPU route context was released before the edge walk")
		}
		return ctx.resident.LocalModuleCounts(width, height, fps)
	}
	detector.searchAlignment = func(grid alignmentGrid) ([]FinderPattern, error) {
		if ctx.epoch.Load() != leaseEpoch {
			return nil, fmt.Errorf("jabcode: GPU route context was released before the alignment search")
		}
		return ctx.resident.SearchAlignment(width, height, grid)
	}
	detector.detachChannels = func() error {
		if ctx.epoch.Load() != leaseEpoch {
			return fmt.Errorf("jabcode: GPU route context was released before mask snapshot")
		}
		materialize, err := ctx.resident.snapshotChannels(detector.Ch)
		if err != nil {
			return err
		}
		detector.materializeChannels = materialize
		return nil
	}
	return detector, nil
}

func finishGPUDetector(
	detector *PrimaryDetector,
	found FinderFamilySet,
	trace *DetectorTrace,
) (*PrimaryDetector, FinderFamilySet, error) {
	// Diagnostics render the balanced image itself, so a traced pass downloads
	// it here where a failure can still be reported. An ordinary located pass
	// does not: the caller holds the lease for its whole decode and asks for the
	// pixels at the one host stage that reads them, so a level that locates
	// finders but never reaches sampling costs nothing.
	if trace != nil && !detector.ensureBitmap() {
		if detector.materializeErr != nil {
			return nil, 0, detector.materializeErr
		}
		return nil, 0, fmt.Errorf("jabcode: materialize resident GPU balanced image")
	}
	// Diagnostics visualize every pass's channels, so a traced detector
	// expands its masks eagerly; a failed lazy expansion surfaces here
	// instead of as absent pixels later.
	if trace != nil && !detector.ensureChannels() {
		if detector.materializeChanErr != nil {
			return nil, 0, detector.materializeChanErr
		}
		return nil, 0, fmt.Errorf("jabcode: materialize resident GPU mask channels")
	}
	// A located success preserves its masks so the consumers that read mask
	// pixels - the docked traversal and the historical wire routes - still see
	// this pass's masks after a later pass overwrote the shared buffers. The
	// preservation is a device-side copy and the fetch is deferred to first
	// need, so a route that never reads a mask pixel pays no host traffic for
	// the guarantee.
	if found != 0 {
		if err := detector.detachLocatedChannels(); err != nil {
			return nil, 0, err
		}
	}
	return detector, found, nil
}

// Close waits for every registered session operation and in-flight route to
// finish, then releases the workspace. Automatic sessions cache it for
// another same-sized decode; borrowed-device sessions release their buffers
// and pipelines. The operation gate covers the window between a method's
// entry and its context acquisition, so no operation can straddle the
// release and touch a torn-down or reused workspace.
func (session *GPUDecodeSession) Close() error {
	if session == nil {
		return nil
	}
	// Flipping the closing flag under enterMu serializes it against the
	// enter gate: after this point no new operation can register, and every
	// registered operation is visible to the wait below.
	session.enterMu.Lock()
	alreadyClosing := session.closing.Swap(true)
	session.enterMu.Unlock()
	if alreadyClosing {
		return nil
	}
	if session.workspace != nil {
		session.workspace.contexts.beginDrain()
	}
	session.ops.Wait()
	if session.workspace != nil {
		session.workspace.contexts.drain()
	}
	if session.release != nil {
		return session.release()
	}
	return nil
}
