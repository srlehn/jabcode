//go:build !js

package detect

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/phaseprobe"
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

	chainWarm             sync.Once
	chainReady            atomic.Bool
	directionalChainReady atomic.Bool
	directionalChainErr   atomic.Pointer[error]
	pitchLagReady         atomic.Bool

	// ballotFallback holds the first failure that pushed finderWindows onto the
	// portable kernel, so a fallback is never silent. See ballotFallbackError.
	ballotFallback atomic.Pointer[error]

	// The subgroup partitioning probe runs at most once per set; see
	// subgroupLayoutUsable.
	subgroupProbeOnce sync.Once
	subgroupProbeOK   bool
	subgroupProbeErr  error
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
// masks, output, parameters, counts. The parameter block is the one binding
// every invocation reads identically, so it is bound as a uniform buffer; see
// shaders/finder_scan_params.wgsl for why the mask buffer cannot be.
var gpuKernelLayoutScan = []vulki.BindingLayout{
	{Binding: 0, Access: vulki.BufferReadOnly},
	{Binding: 1, Access: vulki.BufferReadWrite},
	{Binding: 2, Access: vulki.BufferUniform},
	{Binding: 3, Access: vulki.BufferReadWrite},
}

// The fused window kernels' record layout, shared by the route and the
// harnesses so neither can drift from shaders/finder_windows_common.wgsl.
const (
	// finderWindowRecordWords is RECORD: key, six boundaries, module size.
	finderWindowRecordWords = 8
	// finderWindowCounterCount is the kernel's four counts: records required,
	// cross-checked candidates with inner runs of at least three samples, those
	// a diagonal rescued after the perpendicular failed, and the windows that
	// passed along the line before any of that.
	finderWindowCounterCount = 4
	// finderEvidenceBits is EVIDENCE_SHIFT: the key word's top two bits say
	// which of the three walks confirmed the candidate, or zero where none did.
	finderEvidenceBits = 30
)

// finderScanParamsWords is the Params struct in shaders/finder_scan_params.wgsl,
// fourteen scalars with no padding between them.
const finderScanParamsWords = 14

// finderScanEmitUnconfirmed is FLAG_EMIT_UNCONFIRMED in the same file: the fused
// window kernels record every window that passed along the scan line, confirmed
// or not, with each record's verdict still attached. It is how the cross-check's
// recall is measured against another candidate generator, since a counter gives
// a total and the question is about the set.
const finderScanEmitUnconfirmed = 1

// finderScanSkipCrossCheck is FLAG_SKIP_CROSS_CHECK: no off-line walk runs and
// every window that passes along the scan line is recorded. This is the mode the
// directional route dispatches with, because the walks confirm on the seek
// channel while the host chain confirms on the other two, and they reject
// candidates that chain keeps.
const finderScanSkipCrossCheck = 2

// finderScanParamsBytes is the size a scan parameter buffer must have, which is
// just the struct's own size. The uniform address space raises the struct's
// required *alignment* to 16 and leaves its size alone, so there is nothing to
// round up here.
const finderScanParamsBytes = finderScanParamsWords * 4

// finderRunsHillis extracts directional run boundaries in parallel, compacting
// them with a workgroup Hillis-Steele scan.
func (set *gpuDecodeKernels) finderRunsHillis(layout finderScanLayout) (*vulki.Kernel, error) {
	return set.kernel(
		"finder runs hillis "+layout.name(),
		layout.prelude()+finderRunsHillisWGSL,
		gpuKernelLayoutScan,
	)
}

// finderRunsSubgroup is finderRunsHillis with the scan replaced by a subgroup
// ballot and a bit-count prefix. It derives a lane's subgroup index the same way
// the fused ballot kernel does, so it needs the same full-subgroup guarantee;
// building it as an ordinary pipeline would leave its boundary ordering resting
// on an assumption nothing checks.
func (set *gpuDecodeKernels) finderRunsSubgroup(layout finderScanLayout) (*vulki.Kernel, error) {
	return set.kernelWith(vulki.KernelOptions{
		WGSL:                 enableSubgroupsWGSL + layout.prelude() + finderRunsSubgroupWGSL,
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
//
// The size is never pinned. Without ALLOW_VARYING_SUBGROUP_SIZE, which vulki
// does not set, the pipeline runs at the device's own SubgroupSize, so there is
// nothing to correct; pinning would also bring VUID-VkPipelineShaderStageCreate
// Info-pNext-02756 into play, whose limit vulki does not report.
func (set *gpuDecodeKernels) finderWindowsBallot(layout finderScanLayout) (*vulki.Kernel, error) {
	return set.kernelWith(vulki.KernelOptions{
		WGSL:                 enableSubgroupsWGSL + layout.prelude() + finderCrossCheckWGSL + finderWindowsCommonWGSL + finderWindowsBallotWGSL,
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
// Every failure on the way is recorded, including the two that used to be
// dropped here: a probe that would not dispatch, and a ballot kernel that builds
// for one mask layout and not the other. Both leave the route permanently slower
// and neither is a capability limit.
//
// The one thing it cannot substitute for is the workgroup size. Both variants
// declare 256 lanes, so a device below that limit gets a plain error here rather
// than a kernel that will not build; there is no smaller variant to fall back to.
func (set *gpuDecodeKernels) finderWindows(layout finderScanLayout) (*vulki.Kernel, error) {
	if set == nil || set.device == nil {
		return nil, fmt.Errorf("jabcode: GPU kernel set is closed")
	}
	if !finderScanWorkgroupSupported(set.device.Info().Limits) {
		return nil, fmt.Errorf(
			"jabcode: GPU device cannot launch the %d-lane workgroup the directional scan kernels need",
			finderBallotWorkgroup)
	}
	usable, err := set.subgroupKernelsUsable()
	if err != nil {
		set.ballotFallback.CompareAndSwap(nil, &err)
	}
	if usable {
		kernel, err := set.finderWindowsBallot(layout)
		if err == nil {
			return kernel, nil
		}
		set.ballotFallback.CompareAndSwap(nil, &err)
	}
	return set.finderWindowsScan(layout)
}

// ballotFallbackError reports the first failure that forced finderWindows onto
// the portable kernel for a reason that is not a capability limit. It is nil
// when no fallback happened and when the device does not advertise what the
// ballot kernels need. Non-nil means the device said it could build the kernel
// and then could not, which is a defect worth failing a gate over.
func (set *gpuDecodeKernels) ballotFallbackError() error {
	if set == nil {
		return nil
	}
	if held := set.ballotFallback.Load(); held != nil {
		return *held
	}
	return nil
}

// subgroupKernelsUsable reports whether the ballot kernels may be used here.
// Three conditions, none of which implies another:
//
//   - the device advertises the ballot operation class, full subgroups, and a
//     size the shaders can be built for;
//   - it partitions a workgroup into full subgroups indexed the way the kernels
//     assume, which is measured rather than inferred from the pipeline flag,
//     because Vulkan defines no relationship between SubgroupLocalInvocationId
//     and LocalInvocationIndex;
//   - the kernel itself builds.
//
// Everything here is cached, so the cost is paid once per device.
//
// Every capability limit is settled by the first condition, which reads limits
// rather than building anything. That is what lets a build failure after it be
// treated as a defect without exception: the device has already said it can do
// this. The returned error is therefore never a capability limit, and is nil
// both when the kernels are usable and when the device simply cannot run them.
func (set *gpuDecodeKernels) subgroupKernelsUsable() (bool, error) {
	if !set.finderBallotUsable() {
		return false, nil
	}
	switch layout, err := set.subgroupLayoutUsable(); {
	case err != nil:
		return false, err
	case !layout:
		return false, nil
	}
	if _, err := set.finderWindowsBallot(finderScanInterleaved); err != nil {
		set.ballotFallback.CompareAndSwap(nil, &err)
		return false, err
	}
	return true, nil
}

// finderWindowsScan is finderWindowsBallot's portable twin: same output, same
// record layout, compaction by a workgroup scan so it needs nothing beyond core
// WGSL. Falling back to a boundary kernel instead would reinstate the per-line
// slot layout the fused design exists to remove.
func (set *gpuDecodeKernels) finderWindowsScan(layout finderScanLayout) (*vulki.Kernel, error) {
	return set.kernel(
		"finder windows scan "+layout.name(),
		layout.prelude()+finderCrossCheckWGSL+finderWindowsCommonWGSL+finderWindowsScanWGSL,
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

// gpuKernelLayoutRowChain is gpuKernelLayoutChain with the balanced source
// image, which the current-family row chain reads for the source-colour signal.
var gpuKernelLayoutRowChain = append(
	append([]vulki.BindingLayout(nil), gpuKernelLayoutChain...),
	vulki.BindingLayout{Binding: 4, Access: vulki.BufferReadOnly},
	vulki.BindingLayout{Binding: 5, Access: vulki.BufferReadWrite},
	vulki.BindingLayout{Binding: 6, Access: vulki.BufferReadWrite},
)

func (set *gpuDecodeKernels) finderChain() (*vulki.Kernel, error) {
	return set.kernel(
		"finder chain",
		finderChainBindingsWGSL+finderChainPreludeWGSL+
			finderChainRowWGSL+finderChainCurrentWGSL,
		gpuKernelLayoutRowChain,
	)
}

func (set *gpuDecodeKernels) finderChainBSI() (*vulki.Kernel, error) {
	return set.kernel(
		"BSI finder chain",
		finderChainBindingsWGSL+finderChainPreludeWGSL+
			finderChainRowWGSL+finderChainBSIWGSL,
		gpuKernelLayoutRowChain,
	)
}

// gpuKernelLayoutChainColor is gpuKernelLayoutChain with the balanced source
// image, the sweep summary and the indirect dispatch arguments bound: the chain
// that decides the colour signal also folds every hit into counters the host
// reads instead of the hits, and takes its own invocation bound from the device.
var gpuKernelLayoutChainColor = append(
	append([]vulki.BindingLayout(nil), gpuKernelLayoutChain...),
	vulki.BindingLayout{Binding: 4, Access: vulki.BufferReadOnly},
	vulki.BindingLayout{Binding: 5, Access: vulki.BufferReadWrite},
	vulki.BindingLayout{Binding: 6, Access: vulki.BufferReadOnly},
)

func (set *gpuDecodeKernels) finderChainDirectional() (*vulki.Kernel, error) {
	return set.kernel(
		"directional finder chain",
		finderChainDirectionalBindingsWGSL+finderChainPreludeWGSL+
			finderChainDirectionalWGSL+finderChainDirectionalCurrentWGSL,
		gpuKernelLayoutChainColor,
	)
}

func (set *gpuDecodeKernels) finderChainDirectionalBSI() (*vulki.Kernel, error) {
	return set.kernel(
		"BSI directional finder chain",
		finderChainDirectionalBindingsWGSL+finderChainPreludeWGSL+
			finderChainDirectionalWGSL+finderChainDirectionalBSIWGSL,
		gpuKernelLayoutChainColor,
	)
}

// gpuKernelLayoutDispatchArgs binds the scan counter, the indirect arguments,
// the summary and the chain parameter block.
var gpuKernelLayoutDispatchArgs = []vulki.BindingLayout{
	{Binding: 0, Access: vulki.BufferReadOnly},
	{Binding: 1, Access: vulki.BufferReadWrite},
	{Binding: 2, Access: vulki.BufferReadWrite},
	{Binding: 3, Access: vulki.BufferReadOnly},
}

func (set *gpuDecodeKernels) finderDispatchArgs() (*vulki.Kernel, error) {
	return set.kernel("finder dispatch arguments", finderDispatchArgsWGSL, gpuKernelLayoutDispatchArgs)
}

// compileFinderChains compiles the row-chain kernels of every compiled family
// synchronously and marks them usable. The much larger directional chain has
// its own join point so row-only users never pay for it.
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

// compileDirectionalFinderChain compiles every compiled family's directional
// chain and the tiny kernel that dispatches them. Readiness covers all of
// them, because a chain is only ever dispatched indirectly from arguments that
// kernel writes.
func (set *gpuDecodeKernels) compileDirectionalFinderChain() error {
	kernels := []func() (*vulki.Kernel, error){
		set.finderChainDirectional,
		set.finderDispatchArgs,
	}
	if bsiFamilyFinderEnabled {
		kernels = append(kernels, set.finderChainDirectionalBSI)
	}
	for _, kernel := range kernels {
		if _, err := kernel(); err != nil {
			set.directionalChainErr.CompareAndSwap(nil, &err)
			return err
		}
	}
	set.directionalChainReady.Store(true)
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
			phaseprobe.Mark("kernels.warm.start")
			err := set.compileFinderChains()
			phaseprobe.Markf("kernels.rowchain.ready", "error=%t", err != nil)
			err = set.compileDirectionalFinderChain()
			phaseprobe.Markf("kernels.dirchain.ready", "error=%t", err != nil)
			err = set.compilePitchLag()
			phaseprobe.Markf("kernels.pitchlag.ready", "error=%t", err != nil)
		}()
	})
}

// finderChainsReady reports whether the compiled chain kernels are usable;
// after it returns true the accessors return cached kernels without blocking.
func (set *gpuDecodeKernels) finderChainsReady() bool {
	return set.chainReady.Load()
}

func (set *gpuDecodeKernels) directionalFinderChainReady() bool {
	return set.directionalChainReady.Load()
}

func (set *gpuDecodeKernels) directionalFinderChainError() error {
	if err := set.directionalChainErr.Load(); err != nil {
		return *err
	}
	return nil
}

func (set *gpuDecodeKernels) finderAverage() (*vulki.Kernel, error) {
	return set.kernel("finder average", finderAverageWGSL, gpuKernelLayoutInOutParams)
}

func (set *gpuDecodeKernels) sampleSymbol() (*vulki.Kernel, error) {
	return set.kernel("symbol sampler", sampleSymbolWGSL, gpuKernelLayoutInOutParams)
}

// gpuKernelLayoutAlignment adds the per-tile scratch the alignment search folds
// over to the common in/out/params layout.
var gpuKernelLayoutAlignment = []vulki.BindingLayout{
	{Binding: 0, Access: vulki.BufferReadOnly},
	{Binding: 1, Access: vulki.BufferReadWrite},
	{Binding: 2, Access: vulki.BufferReadOnly},
	{Binding: 3, Access: vulki.BufferReadWrite},
}

func (set *gpuDecodeKernels) ldpcHard() (*vulki.Kernel, error) {
	return set.kernel("hard LDPC correction", ldpcHardWGSL, gpuKernelLayoutAlignment)
}

// gpuKernelLayoutParamsOut is the layout of a stage whose only input is its
// parameter block and whose whole result stays on the device.
var gpuKernelLayoutParamsOut = []vulki.BindingLayout{
	{Binding: 0, Access: vulki.BufferReadOnly},
	{Binding: 1, Access: vulki.BufferReadWrite},
}

// gpuKernelLayoutPayloadBits is the payload classifier's layout: parameters,
// the sampled grid, the data map and the deinterleaving permutation in, the
// codeword out.
var gpuKernelLayoutPayloadBits = []vulki.BindingLayout{
	{Binding: 0, Access: vulki.BufferReadOnly},
	{Binding: 1, Access: vulki.BufferReadOnly},
	{Binding: 2, Access: vulki.BufferReadOnly},
	{Binding: 3, Access: vulki.BufferReadOnly},
	{Binding: 4, Access: vulki.BufferReadWrite},
}

// gpuKernelLayoutMetadata is the metadata walk's layout: parameters and the
// sampled grid in, the codeword for the corrector and the interpretation record
// out.
var gpuKernelLayoutMetadata = []vulki.BindingLayout{
	{Binding: 0, Access: vulki.BufferReadOnly},
	{Binding: 1, Access: vulki.BufferReadOnly},
	{Binding: 2, Access: vulki.BufferReadWrite},
	{Binding: 3, Access: vulki.BufferReadWrite},
}

// gpuKernelLayoutMetadataPalette is the palette stage's layout: parameters, the
// sampled grid and the corrector's output in, the record out.
var gpuKernelLayoutMetadataPalette = []vulki.BindingLayout{
	{Binding: 0, Access: vulki.BufferReadOnly},
	{Binding: 1, Access: vulki.BufferReadOnly},
	{Binding: 2, Access: vulki.BufferReadOnly},
	{Binding: 3, Access: vulki.BufferReadWrite},
}

// gpuKernelLayoutFinderSelect is the selection stage's layout: parameters, the
// folded patterns and the fold's record in, the selection out.
var gpuKernelLayoutFinderSelect = []vulki.BindingLayout{
	{Binding: 0, Access: vulki.BufferReadOnly},
	{Binding: 1, Access: vulki.BufferReadOnly},
	{Binding: 2, Access: vulki.BufferReadOnly},
	{Binding: 3, Access: vulki.BufferReadWrite},
}

// gpuKernelLayoutFinderFold is the fold's layout: parameters and the ordered
// candidates in, the accumulated patterns, the fold record and the weak seed
// list out.
var gpuKernelLayoutFinderFold = []vulki.BindingLayout{
	{Binding: 0, Access: vulki.BufferReadOnly},
	{Binding: 1, Access: vulki.BufferReadOnly},
	{Binding: 2, Access: vulki.BufferReadWrite},
	{Binding: 3, Access: vulki.BufferReadWrite},
	{Binding: 4, Access: vulki.BufferReadWrite},
}

// gpuKernelLayoutFinderCandidates is the assembly's layout: parameters and one
// direction's compacted chain outcomes in, the candidate list, the shared fold
// parameters and the assembly record out.
var gpuKernelLayoutFinderCandidates = []vulki.BindingLayout{
	{Binding: 0, Access: vulki.BufferReadOnly},
	{Binding: 1, Access: vulki.BufferReadOnly},
	{Binding: 2, Access: vulki.BufferReadWrite},
	{Binding: 3, Access: vulki.BufferReadWrite},
	{Binding: 4, Access: vulki.BufferReadWrite},
}

// gpuKernelLayoutFinderPool is the pool accumulation's layout: parameters, the
// source patterns and the record carrying their count in, the pool and its own
// record in and out - the pool is resumed rather than rebuilt.
var gpuKernelLayoutFinderPool = []vulki.BindingLayout{
	{Binding: 0, Access: vulki.BufferReadOnly},
	{Binding: 1, Access: vulki.BufferReadOnly},
	{Binding: 2, Access: vulki.BufferReadOnly},
	{Binding: 3, Access: vulki.BufferReadWrite},
	{Binding: 4, Access: vulki.BufferReadWrite},
}

// gpuKernelLayoutMetadataFinish is the field stage's layout: parameters and the
// corrector's output in, the record out. It needs no grid; by then every module
// it depends on has already been read.
var gpuKernelLayoutMetadataFinish = []vulki.BindingLayout{
	{Binding: 0, Access: vulki.BufferReadOnly},
	{Binding: 1, Access: vulki.BufferReadOnly},
	{Binding: 2, Access: vulki.BufferReadWrite},
}

func (set *gpuDecodeKernels) payloadMap() (*vulki.Kernel, error) {
	return set.kernel("payload data map", payloadMapWGSL, gpuKernelLayoutParamsOut)
}

func (set *gpuDecodeKernels) payloadPermute() (*vulki.Kernel, error) {
	return set.kernel("deinterleaving permutation", payloadPermuteWGSL, gpuKernelLayoutParamsOut)
}

func (set *gpuDecodeKernels) payloadBits() (*vulki.Kernel, error) {
	return set.kernel("payload classification", payloadBitsWGSL, gpuKernelLayoutPayloadBits)
}

func (set *gpuDecodeKernels) metadataPart1() (*vulki.Kernel, error) {
	return set.kernel("metadata part I", metadataPart1WGSL, gpuKernelLayoutMetadata)
}

func (set *gpuDecodeKernels) metadataPalette() (*vulki.Kernel, error) {
	return set.kernel("metadata palette", metadataPaletteWGSL, gpuKernelLayoutMetadataPalette)
}

func (set *gpuDecodeKernels) metadataPart2() (*vulki.Kernel, error) {
	return set.kernel("metadata part II", metadataPart2WGSL, gpuKernelLayoutMetadata)
}

func (set *gpuDecodeKernels) metadataFinish() (*vulki.Kernel, error) {
	return set.kernel("metadata fields", metadataFinishWGSL, gpuKernelLayoutMetadataFinish)
}

func (set *gpuDecodeKernels) finderFold() (*vulki.Kernel, error) {
	return set.kernel("finder candidate fold", finderFoldWGSL, gpuKernelLayoutFinderFold)
}

func (set *gpuDecodeKernels) finderCandidates() (*vulki.Kernel, error) {
	return set.kernel("finder candidate assembly", finderCandidatesWGSL, gpuKernelLayoutFinderCandidates)
}

func (set *gpuDecodeKernels) finderPool() (*vulki.Kernel, error) {
	return set.kernel("finder candidate pool", finderPoolWGSL, gpuKernelLayoutFinderPool)
}

func (set *gpuDecodeKernels) finderSelect() (*vulki.Kernel, error) {
	return set.kernel("finder selection", finderSelectWGSL, gpuKernelLayoutFinderSelect)
}

func (set *gpuDecodeKernels) finderSort() (*vulki.Kernel, error) {
	return set.kernel("finder candidate order", finderSortWGSL, gpuKernelLayoutParamsOut)
}

func (set *gpuDecodeKernels) alignmentSearch() (*vulki.Kernel, error) {
	return set.kernel("alignment search", alignmentSearchWGSL, gpuKernelLayoutAlignment)
}

func (set *gpuDecodeKernels) localModuleCount() (*vulki.Kernel, error) {
	return set.kernel("local module count", localModuleCountWGSL, gpuKernelLayoutInOutParams)
}

func (set *gpuDecodeKernels) pitchSamples() (*vulki.Kernel, error) {
	return set.kernel("pitch samples", pitchSamplesWGSL, gpuKernelLayoutInOutParams)
}

func (set *gpuDecodeKernels) pitchLineSums() (*vulki.Kernel, error) {
	return set.kernel("pitch line sums", pitchLineSumsWGSL, gpuKernelLayoutInOutParams)
}

func (set *gpuDecodeKernels) pitchCenter() (*vulki.Kernel, error) {
	return set.kernel("pitch center", pitchCenterWGSL, gpuKernelLayoutChain)
}

func (set *gpuDecodeKernels) pitchACF() (*vulki.Kernel, error) {
	return set.kernel("pitch autocorrelation", pitchACFWGSL, gpuKernelLayoutInOutParams)
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
