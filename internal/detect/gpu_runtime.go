//go:build !js

package detect

import (
	"fmt"
	"sync"

	"github.com/srlehn/vulki"
)

// automaticGPUMinPixels is the smallest measured workload with a demonstrated
// end-to-end preprocessing win on the reference discrete GPU. Smaller images
// stay on the CPU without initializing Vulkan. This is a scheduling crossover,
// not an image-processing scale.
const automaticGPUMinPixels = 1024 * 1024

var automaticGPUDevices = newGPUDeviceCache(vulki.Open)

type gpuDeviceCache struct {
	once sync.Once
	open func() (*vulki.Device, error)

	device *vulki.Device
	err    error
}

func newGPUDeviceCache(open func() (*vulki.Device, error)) *gpuDeviceCache {
	return &gpuDeviceCache{open: open}
}

// deviceFor returns the process-wide borrowed device for a workload large
// enough to benefit. Discovery is lazy and attempted at most once; callers
// transparently use their CPU route when no device is returned.
func (cache *gpuDeviceCache) deviceFor(width, height int) (*vulki.Device, error) {
	if !automaticGPUWorkload(width, height) {
		return nil, nil
	}
	if cache == nil {
		return nil, fmt.Errorf("jabcode: automatic GPU device cache is unavailable")
	}
	cache.once.Do(func() {
		if cache.open == nil {
			cache.err = fmt.Errorf("jabcode: automatic GPU device opener is unavailable")
			return
		}
		device, err := cache.open()
		if err != nil {
			cache.err = fmt.Errorf("jabcode: open automatic GPU device: %w", err)
			return
		}
		if !automaticGPUAdapter(device.Info()) {
			info := device.Info()
			cache.err = fmt.Errorf(
				"jabcode: Vulkan adapter %q has unsupported automatic device type %d",
				info.AdapterName,
				info.DeviceType,
			)
			_ = device.Close()
			return
		}
		cache.device = device
	})
	if cache.device == nil || cache.device.Closed() {
		return nil, cache.err
	}
	// A sticky device fault (a lost device, an unfenced submission) makes
	// every later lease fail anyway; gate here so new decodes go straight to
	// their CPU route instead of probing a sick device per route.
	if err := cache.device.Err(); err != nil {
		return nil, fmt.Errorf("jabcode: automatic GPU device is unavailable: %w", err)
	}
	return cache.device, nil
}

// WarmAutomaticGPUDevice starts device discovery for a frame of this size and
// returns immediately. Opening a Vulkan device is loader, instance and
// logical-device creation and costs well over a hundred milliseconds on the
// reference adapter - a large share of a one-second read, and all of it spent
// before the first dispatch can even be recorded. None of it depends on the
// image, so a caller that knows the frame size can overlap it with decoding the
// image and building the pyramid; the route's own acquisition then joins this
// one-shot discovery rather than starting it. A caller that turns out never to
// reach a device route has spent one goroutine, and a frame below the automatic
// threshold still never initializes Vulkan.
func WarmAutomaticGPUDevice(width, height int) { automaticGPUDevices.warm(width, height) }

func (cache *gpuDeviceCache) warm(width, height int) {
	// The switch has to be read here: deviceFor does not know about it, and a
	// warm is the one caller that reaches discovery without passing through
	// begin. The size test only avoids a goroutine per small-frame decode,
	// which a stream does many of - deviceFor declines the same frames anyway.
	if cache == nil || gpuRoutesDisabled.Load() || !automaticGPUWorkload(width, height) {
		return
	}
	go func() { _, _ = cache.deviceFor(width, height) }()
}

func automaticGPUWorkload(width, height int) bool {
	if width <= 0 || height <= 0 {
		return false
	}
	return uint64(width)*uint64(height) >= automaticGPUMinPixels
}

func automaticGPUAdapter(info vulki.DeviceInfo) bool {
	return info.DeviceType == vulki.DeviceTypeDiscreteGPU
}
