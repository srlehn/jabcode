//go:build !js

package detect

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/srlehn/vulki"
)

// gpuDecodeKernels shares one compiled compute kernel per shader across every
// consumer on one device. Kernel pipelines are immutable after creation and
// every vulki binding set owns its own descriptor pool, so concurrent binding
// creation and dispatch against a shared kernel are safe. Sharing avoids
// recompiling WGSL for every route context; compilation is lazy so callers
// only pay for the stages they use. The set never closes a kernel while its
// binding sets are alive - owners close contexts before the set.
type gpuDecodeKernels struct {
	device *vulki.Device

	mu     sync.Mutex
	cells  map[string]*gpuKernelCell
	closed bool

	chainWarm     sync.Once
	chainReady    atomic.Bool
	pitchLagReady atomic.Bool

	// ballotFallback holds the first failure that pushed finderWindows onto the
	// portable kernel, so a fallback is never silent. See ballotFallbackError.
	ballotFallback atomic.Pointer[error]
}

// gpuKernelCell compiles one kernel exactly once on first request. Requests
// for the same kernel wait on its cell; requests for different kernels never
// serialize each other, so the background chain warmup cannot stall a
// route's cheap kernel lookups.
type gpuKernelCell struct {
	once   sync.Once
	kernel *vulki.Kernel
	err    error
}

func newGPUDecodeKernels(device *vulki.Device) *gpuDecodeKernels {
	return &gpuDecodeKernels{device: device, cells: make(map[string]*gpuKernelCell)}
}

func (set *gpuDecodeKernels) kernel(
	name, wgsl string,
	bindings []vulki.BindingLayout,
) (*vulki.Kernel, error) {
	return set.kernelWith(vulki.KernelOptions{WGSL: wgsl, Bindings: bindings}, name)
}

// kernelWith is kernel with the full option set exposed, for the kernels that
// need a pipeline guarantee rather than only source and bindings.
func (set *gpuDecodeKernels) kernelWith(
	options vulki.KernelOptions,
	name string,
) (*vulki.Kernel, error) {
	if set == nil {
		return nil, fmt.Errorf("jabcode: GPU kernel set is closed")
	}
	set.mu.Lock()
	if set.closed || set.device == nil || set.device.Closed() {
		set.mu.Unlock()
		return nil, fmt.Errorf("jabcode: GPU kernel set is closed")
	}
	cell, ok := set.cells[name]
	if !ok {
		cell = &gpuKernelCell{}
		set.cells[name] = cell
	}
	device := set.device
	set.mu.Unlock()
	cell.once.Do(func() {
		kernel, err := device.NewKernel(options)
		if err != nil {
			cell.err = fmt.Errorf("jabcode: create GPU %s kernel: %w", name, err)
			return
		}
		cell.kernel = kernel
	})
	return cell.kernel, cell.err
}

// gpuKernelLayoutInOutParams is the common one-input, one-output,
// one-parameter storage-buffer layout most decode kernels use.
var gpuKernelLayoutInOutParams = []vulki.BindingLayout{
	{Binding: 0, Access: vulki.BufferReadOnly},
	{Binding: 1, Access: vulki.BufferReadWrite},
	{Binding: 2, Access: vulki.BufferReadOnly},
}

func (set *gpuDecodeKernels) halve() (*vulki.Kernel, error) {
	return set.kernel("half-scale", halveNRGBAWGSL, gpuKernelLayoutInOutParams)
}

func (set *gpuDecodeKernels) histogramRGB() (*vulki.Kernel, error) {
	return set.kernel("RGB histogram", histogramRGBWGSL, gpuKernelLayoutInOutParams)
}

func (set *gpuDecodeKernels) histogramBounds() (*vulki.Kernel, error) {
	return set.kernel("histogram bounds", histogramBoundsWGSL, []vulki.BindingLayout{
		{Binding: 0, Access: vulki.BufferReadWrite},
		{Binding: 1, Access: vulki.BufferReadWrite},
	})
}

func (set *gpuDecodeKernels) balanceRGB() (*vulki.Kernel, error) {
	return set.kernel("RGB balance", balanceRGBWGSL, []vulki.BindingLayout{
		{Binding: 0, Access: vulki.BufferReadOnly},
		{Binding: 1, Access: vulki.BufferReadWrite},
		{Binding: 2, Access: vulki.BufferReadOnly},
		{Binding: 3, Access: vulki.BufferReadOnly},
	})
}

func (set *gpuDecodeKernels) blockThresholds() (*vulki.Kernel, error) {
	return set.kernel("block thresholds", blockThresholdsWGSL, gpuKernelLayoutInOutParams)
}

func (set *gpuDecodeKernels) classifyRGB() (*vulki.Kernel, error) {
	return set.kernel("RGB classifier", binarizeRGBWGSL, []vulki.BindingLayout{
		{Binding: 0, Access: vulki.BufferReadOnly},
		{Binding: 1, Access: vulki.BufferReadOnly},
		{Binding: 2, Access: vulki.BufferReadWrite},
		{Binding: 3, Access: vulki.BufferReadOnly},
	})
}

func (set *gpuDecodeKernels) filterBinary() (*vulki.Kernel, error) {
	return set.kernel("binary filter", filterBinaryWGSL, gpuKernelLayoutInOutParams)
}

func (set *gpuDecodeKernels) packMasks() (*vulki.Kernel, error) {
	return set.kernel("mask packer", packBinaryMasksWGSL, gpuKernelLayoutInOutParams)
}

func (set *gpuDecodeKernels) finderRowScan() (*vulki.Kernel, error) {
	return set.kernel("finder row scan", finderRowScanWGSL, gpuKernelLayoutInOutParams)
}

// finderLineScan is the arbitrary-direction form of the row scan. It is a
// measured baseline for the parallel replacement, not a route: see the shader's
// own header for why a serial per-line walk is the wrong shape here.
func (set *gpuDecodeKernels) finderLineScan() (*vulki.Kernel, error) {
	return set.kernel("finder line scan", finderLineScanWGSL, gpuKernelLayoutInOutParams)
}

// finderScanLayout selects how the packed binary masks are stored for the
// directional prototypes. Storage is a measured axis, not a fixed choice: the
// interleaved layout is what the resident binarizer already writes, while the
// bitplane layout covers four times as many pixels per word for a
// single-channel walk, and which one wins depends on the scan primitive above
// it.
type finderScanLayout int

const (
	// finderScanInterleaved is three channel bits per pixel, eight pixels per
	// word: the resident binarizer's own layout.
	finderScanInterleaved finderScanLayout = iota
	// finderScanBitplane is one contiguous plane per channel, 32 pixels per
	// word.
	finderScanBitplane
)

func (l finderScanLayout) name() string {
	if l == finderScanBitplane {
		return "bitplane"
	}
	return "interleaved"
}

func (l finderScanLayout) prelude() string {
	source := finderScanParamsWGSL
	if l == finderScanBitplane {
		source += finderScanMaskPlaneWGSL
	} else {
		source += finderScanMaskPackedWGSL
	}
	return source + finderScanGeometryWGSL
}

// gpuKernelLayoutScan is the directional prototypes' shared binding layout:
// masks, output, parameters, counts.
var gpuKernelLayoutScan = []vulki.BindingLayout{
	{Binding: 0, Access: vulki.BufferReadOnly},
	{Binding: 1, Access: vulki.BufferReadWrite},
	{Binding: 2, Access: vulki.BufferReadOnly},
	{Binding: 3, Access: vulki.BufferReadWrite},
}

// finderRunsHillis extracts directional run boundaries in parallel, compacting
// them with a workgroup Hillis-Steele scan.
func (set *gpuDecodeKernels) finderRunsHillis(layout finderScanLayout) (*vulki.Kernel, error) {
	return set.kernel(
		"finder runs hillis "+layout.name(),
		layout.prelude()+finderRunsHillisWGSL,
		gpuKernelLayoutScan,
	)
}

// wgslEnableSubgroups must precede every declaration in a module that uses
// subgroup operations. It is prepended here rather than written at the top of a
// shader file because the shaders are concatenated after a shared prelude, so
// no single file is the start of the module. The vendored naga accepts the
// directive mid-module; the specification does not, and a stricter compiler
// would reject it.
const wgslEnableSubgroups = "enable subgroups;\n"

// finderRunsSubgroup is finderRunsHillis with the scan replaced by a subgroup
// ballot and a bit-count prefix. It derives a lane's subgroup index the same way
// the fused ballot kernel does, so it needs the same full-subgroup guarantee;
// building it as an ordinary pipeline would leave its boundary ordering resting
// on an assumption nothing checks.
func (set *gpuDecodeKernels) finderRunsSubgroup(layout finderScanLayout) (*vulki.Kernel, error) {
	return set.kernelWith(vulki.KernelOptions{
		WGSL:                 wgslEnableSubgroups + layout.prelude() + finderRunsSubgroupWGSL,
		Bindings:             gpuKernelLayoutScan,
		RequireFullSubgroups: true,
	}, "finder runs subgroup "+layout.name())
}

// finderBallotOperations is the exact set of subgroup operation classes the
// ballot kernels use: SubgroupBasic for the subgroup_size and
// subgroup_invocation_id builtins, SubgroupBallot for subgroupBallot itself.
// The prefix is countOneBits, which is core WGSL and not a subgroup operation.
//
// **Update this if a ballot kernel gains an operation from another class** -
// subgroupExclusiveAdd and friends need SubgroupArithmetic, the shuffles need
// their own classes. Forgetting to would let a device be selected that cannot
// build the kernel, which is why finderWindows also falls back on any build
// failure rather than trusting this mask alone.
const finderBallotOperations = vulki.SubgroupBasic | vulki.SubgroupBallot

// subgroupBallotUsable reports whether this device can run the ballot kernels.
// A zero subgroup size means the implementation reported nothing, which is
// unknown rather than supported, so it is treated as unavailable.
func (set *gpuDecodeKernels) subgroupBallotUsable() bool {
	if set == nil || set.device == nil {
		return false
	}
	limits := set.device.Info().Limits
	return limits.SubgroupSize > 0 &&
		limits.SubgroupOperations&finderBallotOperations == finderBallotOperations
}

// finderWindowsBallot fuses the run extraction and the five-run test, emitting
// only surviving windows and never materializing a boundary buffer. It is the
// fastest form measured and the one the pipeline is designed around.
//
// RequireFullSubgroups is what makes its compaction sound rather than merely
// observed to work: the ballot prefix derives a lane's subgroup index from the
// local invocation index, which is meaningful only when the workgroup is split
// into fully populated subgroups. Without the flag that is an assumption about
// the driver; with it the implementation either guarantees it or refuses to
// build the pipeline, and finderWindows takes the refusal as a signal to use
// the portable twin.
func (set *gpuDecodeKernels) finderWindowsBallot(layout finderScanLayout) (*vulki.Kernel, error) {
	return set.kernelWith(vulki.KernelOptions{
		WGSL:                 wgslEnableSubgroups + layout.prelude() + finderWindowsCommonWGSL + finderWindowsBallotWGSL,
		Bindings:             gpuKernelLayoutScan,
		RequireFullSubgroups: true,
	}, "finder windows ballot "+layout.name())
}

// finderWindows returns the fastest fused window kernel this device can run.
// Every consumer should take this rather than choosing a variant itself, so
// that the capability decision lives in one place and a device that cannot run
// the ballot form still gets a fused kernel rather than a boundary buffer.
//
// Any failure to build the ballot kernel falls back, not only a refused
// full-subgroup guarantee. The reported operation mask is a claim by the
// implementation and finderBallotOperations is a claim by this package about
// what the shader uses; either can be wrong, and a driver may reject a module
// for a reason neither anticipates. Since a correct, slightly slower kernel is
// always available, no such disagreement is worth failing a decode over.
//
// A failure that is not a missing capability is kept rather than dropped. A
// fallback is a permanent 30% loss on every read, and an editing mistake in the
// ballot shader would otherwise produce exactly that with nothing anywhere to
// say it had happened. ballotFallbackError makes that case observable.
//
// ErrFullSubgroupsUnsupported is excluded, because full subgroups are a
// capability separate from ballot support and Vulkan lets a device have one
// without the other. Such a device is running its intended route, not a
// degraded one, and must not be reported as defective.
func (set *gpuDecodeKernels) finderWindows(layout finderScanLayout) (*vulki.Kernel, error) {
	if set.subgroupBallotUsable() {
		kernel, err := set.finderWindowsBallot(layout)
		if err == nil {
			return kernel, nil
		}
		if !errors.Is(err, vulki.ErrFullSubgroupsUnsupported) {
			set.ballotFallback.CompareAndSwap(nil, &err)
		}
	}
	return set.finderWindowsScan(layout)
}

// ballotFallbackError reports the first failure that forced finderWindows onto
// the portable kernel for a reason that is not a capability limit. It is nil
// when no fallback happened, when the device never advertised ballot support,
// and when it advertised ballot support but cannot guarantee full subgroups.
// Non-nil means the device should have been able to build the kernel and could
// not, which is a defect worth failing a gate over.
func (set *gpuDecodeKernels) ballotFallbackError() error {
	if set == nil {
		return nil
	}
	if held := set.ballotFallback.Load(); held != nil {
		return *held
	}
	return nil
}

// subgroupKernelsUsable reports whether the ballot kernels can actually be
// built here, which needs both the ballot operation class and a full-subgroup
// guarantee. The latter is not a queryable limit - vulki keeps it internal and
// only reports it by refusing the pipeline - so this establishes it by building
// one, which is cached and therefore paid once.
//
// The returned error is a defect, never a capability limit: it is nil both when
// the kernels are usable and when the device simply cannot run them.
func (set *gpuDecodeKernels) subgroupKernelsUsable() (bool, error) {
	if !set.subgroupBallotUsable() {
		return false, nil
	}
	switch _, err := set.finderWindowsBallot(finderScanInterleaved); {
	case err == nil:
		return true, nil
	case errors.Is(err, vulki.ErrFullSubgroupsUnsupported):
		return false, nil
	default:
		return false, err
	}
}

// finderWindowsScan is finderWindowsBallot's portable twin: same output, same
// record layout, compaction by a workgroup scan so it needs nothing beyond core
// WGSL. Falling back to a boundary kernel instead would reinstate the per-line
// slot layout the fused design exists to remove.
func (set *gpuDecodeKernels) finderWindowsScan(layout finderScanLayout) (*vulki.Kernel, error) {
	return set.kernel(
		"finder windows scan "+layout.name(),
		layout.prelude()+finderWindowsCommonWGSL+finderWindowsScanWGSL,
		gpuKernelLayoutScan,
	)
}

// gpuKernelLayoutChain is the two-input, one-output, parameters layout the
// finder chain kernels (packed masks, records, outcomes) and the pitch
// center kernel (samples, means, centered values) use.
var gpuKernelLayoutChain = []vulki.BindingLayout{
	{Binding: 0, Access: vulki.BufferReadOnly},
	{Binding: 1, Access: vulki.BufferReadOnly},
	{Binding: 2, Access: vulki.BufferReadWrite},
	{Binding: 3, Access: vulki.BufferReadOnly},
}

func (set *gpuDecodeKernels) finderChain() (*vulki.Kernel, error) {
	return set.kernel(
		"finder chain",
		softfloat64WGSL+finderChainPreludeWGSL+finderChainCurrentWGSL,
		gpuKernelLayoutChain,
	)
}

func (set *gpuDecodeKernels) finderChainBSI() (*vulki.Kernel, error) {
	return set.kernel(
		"BSI finder chain",
		softfloat64WGSL+finderChainPreludeWGSL+finderChainBSIWGSL,
		gpuKernelLayoutChain,
	)
}

// compileFinderChains compiles the finder chain kernels of every compiled
// family synchronously and marks them usable.
func (set *gpuDecodeKernels) compileFinderChains() error {
	if _, err := set.finderChain(); err != nil {
		return err
	}
	if bsiFamilyFinderEnabled {
		if _, err := set.finderChainBSI(); err != nil {
			return err
		}
	}
	set.chainReady.Store(true)
	return nil
}

// warmFinderChains compiles the finder chain kernels in the background. The
// chain modules are the largest this package submits and a cold driver
// pipeline cache can take minutes to compile them, so nothing ever blocks on
// their compilation. Route contexts consume them as soon as they exist and
// classify hits with the bit-identical CPU per-hit chain until then (see
// gpuBinarizer.scanOnly), so a read is never blocked on this and never wrong
// for having started early - it is only slower on the passes that beat the
// compiler. The small pitch-lag kernels follow in the same goroutine and gate
// the descreen retry tier's resident fold the same way.
func (set *gpuDecodeKernels) warmFinderChains() {
	set.chainWarm.Do(func() {
		go func() {
			_ = set.compileFinderChains()
			_ = set.compilePitchLag()
		}()
	})
}

// finderChainsReady reports whether the compiled chain kernels are usable;
// after it returns true the accessors return cached kernels without blocking.
func (set *gpuDecodeKernels) finderChainsReady() bool {
	return set.chainReady.Load()
}

func (set *gpuDecodeKernels) finderAverage() (*vulki.Kernel, error) {
	return set.kernel("finder average", finderAverageWGSL, gpuKernelLayoutInOutParams)
}

func (set *gpuDecodeKernels) pitchSamples() (*vulki.Kernel, error) {
	return set.kernel("pitch samples", pitchSamplesWGSL, gpuKernelLayoutInOutParams)
}

func (set *gpuDecodeKernels) pitchLineSums() (*vulki.Kernel, error) {
	return set.kernel("pitch line sums", softfloat64WGSL+pitchLineSumsWGSL, gpuKernelLayoutInOutParams)
}

func (set *gpuDecodeKernels) pitchCenter() (*vulki.Kernel, error) {
	return set.kernel("pitch center", softfloat64WGSL+pitchCenterWGSL, gpuKernelLayoutChain)
}

func (set *gpuDecodeKernels) pitchACF() (*vulki.Kernel, error) {
	return set.kernel("pitch autocorrelation", softfloat64WGSL+pitchACFWGSL, gpuKernelLayoutInOutParams)
}

// compilePitchLag compiles the resident pitch-lag kernels synchronously and
// marks them usable.
func (set *gpuDecodeKernels) compilePitchLag() error {
	if _, err := set.pitchLineSums(); err != nil {
		return err
	}
	if _, err := set.pitchCenter(); err != nil {
		return err
	}
	if _, err := set.pitchACF(); err != nil {
		return err
	}
	set.pitchLagReady.Store(true)
	return nil
}

// pitchLagKernelsReady reports whether the resident pitch-lag kernels are
// usable; until then estimatePitch downloads the samples and folds the
// autocorrelation on the host, with bit-identical results.
func (set *gpuDecodeKernels) pitchLagKernelsReady() bool {
	return set.pitchLagReady.Load()
}

func (set *gpuDecodeKernels) descreenHorizontal() (*vulki.Kernel, error) {
	return set.kernel("horizontal descreen", descreenHorizontalWGSL, gpuKernelLayoutInOutParams)
}

func (set *gpuDecodeKernels) descreenVertical() (*vulki.Kernel, error) {
	return set.kernel("vertical descreen", descreenVerticalWGSL, []vulki.BindingLayout{
		{Binding: 0, Access: vulki.BufferReadOnly},
		{Binding: 1, Access: vulki.BufferReadWrite},
		{Binding: 2, Access: vulki.BufferReadOnly},
		{Binding: 3, Access: vulki.BufferReadOnly},
	})
}

func (set *gpuDecodeKernels) Close() error {
	if set == nil {
		return nil
	}
	set.mu.Lock()
	if set.closed {
		set.mu.Unlock()
		return nil
	}
	set.closed = true
	cells := set.cells
	set.cells = make(map[string]*gpuKernelCell)
	set.mu.Unlock()
	var closeErrors []error
	for _, cell := range cells {
		// Do waits for an in-flight compile of this cell and marks a cell
		// that never compiled as closed, so no kernel is created after Close.
		cell.once.Do(func() {
			cell.err = fmt.Errorf("jabcode: GPU kernel set is closed")
		})
		if cell.kernel != nil {
			closeErrors = append(closeErrors, cell.kernel.Close())
		}
	}
	return errors.Join(closeErrors...)
}
