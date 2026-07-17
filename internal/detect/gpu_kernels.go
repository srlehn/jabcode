package detect

import (
	"errors"
	"fmt"
	"sync"

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

	mu      sync.Mutex
	kernels map[string]*vulki.Kernel
	closed  bool
}

func newGPUDecodeKernels(device *vulki.Device) *gpuDecodeKernels {
	return &gpuDecodeKernels{device: device, kernels: make(map[string]*vulki.Kernel)}
}

func (set *gpuDecodeKernels) kernel(
	name, wgsl string,
	bindings []vulki.BindingLayout,
) (*vulki.Kernel, error) {
	if set == nil {
		return nil, fmt.Errorf("jabcode: GPU kernel set is closed")
	}
	set.mu.Lock()
	defer set.mu.Unlock()
	if set.closed || set.device == nil || set.device.Closed() {
		return nil, fmt.Errorf("jabcode: GPU kernel set is closed")
	}
	if kernel, ok := set.kernels[name]; ok {
		return kernel, nil
	}
	kernel, err := set.device.NewKernel(vulki.KernelOptions{WGSL: wgsl, Bindings: bindings})
	if err != nil {
		return nil, fmt.Errorf("jabcode: create GPU %s kernel: %w", name, err)
	}
	set.kernels[name] = kernel
	return kernel, nil
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

func (set *gpuDecodeKernels) finderAverage() (*vulki.Kernel, error) {
	return set.kernel("finder average", finderAverageWGSL, gpuKernelLayoutInOutParams)
}

func (set *gpuDecodeKernels) pitchSamples() (*vulki.Kernel, error) {
	return set.kernel("pitch samples", pitchSamplesWGSL, gpuKernelLayoutInOutParams)
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
	defer set.mu.Unlock()
	if set.closed {
		return nil
	}
	set.closed = true
	var closeErrors []error
	for name, kernel := range set.kernels {
		closeErrors = append(closeErrors, kernel.Close())
		delete(set.kernels, name)
	}
	return errors.Join(closeErrors...)
}
