//go:build !js

package detect

import (
	"testing"

	"github.com/srlehn/vulki"
)

// The line scan is the device form of the CPU directional sweep, which is the
// single largest cost in a rotated read and had no device route. Compiling it
// is its own gate: the kernel is only reachable through the retry ladder, so a
// WGSL error in it would otherwise surface as a silent fallback to the very CPU
// walk it exists to replace, and the read would still be correct - just as slow
// as before, with nothing to show which.
func TestGPUFinderLineScanCompiles(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	kernels := newGPUDecodeKernels(device)
	t.Cleanup(func() {
		if err := kernels.Close(); err != nil {
			t.Errorf("close kernels: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close device: %v", err)
		}
	})
	if _, err := kernels.finderLineScan(); err != nil {
		t.Fatalf("compile finder line scan: %v", err)
	}
	// Subgroup variants are optional: an adapter without ballot support or
	// without a full-subgroup guarantee runs the portable kernels by design, so
	// requiring them to build would fail the suite on a device that is working
	// exactly as intended.
	subgroups, err := kernels.subgroupKernelsUsable()
	if err != nil {
		t.Fatalf("device advertises ballot support but the ballot kernel did not build: %v", err)
	}
	for _, layout := range []finderScanLayout{finderScanInterleaved, finderScanBitplane} {
		if _, err := kernels.finderRunsHillis(layout); err != nil {
			t.Fatalf("compile finder runs hillis %s: %v", layout.name(), err)
		}
		if _, err := kernels.finderWindowsScan(layout); err != nil {
			t.Fatalf("compile finder windows scan %s: %v", layout.name(), err)
		}
		if !subgroups {
			continue
		}
		if _, err := kernels.finderRunsSubgroup(layout); err != nil {
			t.Fatalf("compile finder runs subgroup %s: %v", layout.name(), err)
		}
		if _, err := kernels.finderWindowsBallot(layout); err != nil {
			t.Fatalf("compile finder windows ballot %s: %v", layout.name(), err)
		}
	}
}
