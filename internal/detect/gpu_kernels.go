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
		kernel, err := device.NewKernel(vulki.KernelOptions{WGSL: wgsl, Bindings: bindings})
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

func (set *gpuDecodeKernels) rotate() (*vulki.Kernel, error) {
	return set.kernel("rotation", rotateNRGBAWGSL, gpuKernelLayoutInOutParams)
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
// their compilation. Pooled route contexts no longer consume them at all -
// they run scan-only with the bit-identical CPU per-hit chain (see
// gpuBinarizer.deviceReplay) - so this warm now serves the persistent
// pipeline cache and the borrowed-device seam. The small pitch-lag kernels
// follow in the same goroutine and gate the descreen retry tier's resident
// fold for deviceReplay constructions; pooled contexts fold on the CPU.
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
