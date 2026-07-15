package detect

import (
	"errors"
	"testing"

	"github.com/srlehn/vulki"
)

func TestGPUDeviceCacheSkipsSmallWorkload(t *testing.T) {
	openCalls := 0
	cache := newGPUDeviceCache(func() (*vulki.Device, error) {
		openCalls++
		return nil, errors.New("must not open")
	})
	device, err := cache.deviceFor(1023, 1024)
	if err != nil || device != nil {
		t.Fatalf("small workload device = %v, error = %v; want CPU fallback", device, err)
	}
	if openCalls != 0 {
		t.Fatalf("small workload opened Vulkan %d times", openCalls)
	}
}

func TestGPUDeviceCacheCachesUnavailableDevice(t *testing.T) {
	openCalls := 0
	wantErr := errors.New("Vulkan unavailable")
	cache := newGPUDeviceCache(func() (*vulki.Device, error) {
		openCalls++
		return nil, wantErr
	})
	for range 2 {
		device, err := cache.deviceFor(1024, 1024)
		if device != nil {
			t.Fatalf("unavailable device = %v, want CPU fallback", device)
		}
		if !errors.Is(err, wantErr) {
			t.Fatalf("unavailable device error = %v, want %v", err, wantErr)
		}
	}
	if openCalls != 1 {
		t.Fatalf("unavailable device opened Vulkan %d times, want once", openCalls)
	}
}

func TestAutomaticGPUAdapterClassification(t *testing.T) {
	for _, test := range []struct {
		name       string
		deviceType vulki.DeviceType
		want       bool
	}{
		{name: "other", deviceType: vulki.DeviceTypeOther},
		{name: "integrated", deviceType: vulki.DeviceTypeIntegratedGPU},
		{name: "discrete", deviceType: vulki.DeviceTypeDiscreteGPU, want: true},
		{name: "virtual", deviceType: vulki.DeviceTypeVirtualGPU},
		{name: "CPU", deviceType: vulki.DeviceTypeCPU},
	} {
		t.Run(test.name, func(t *testing.T) {
			got := automaticGPUAdapter(vulki.DeviceInfo{DeviceType: test.deviceType})
			if got != test.want {
				t.Fatalf("automaticGPUAdapter(%d) = %v, want %v", test.deviceType, got, test.want)
			}
		})
	}
}
