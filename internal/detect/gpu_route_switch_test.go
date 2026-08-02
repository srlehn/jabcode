//go:build !js

package detect

import (
	"errors"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
)

// A disabled switch has to stop the route before device discovery, not merely
// somewhere. Counting opens is what proves it: a check placed after the device
// was borrowed would still return no session and still look correct from the
// outside, while paying for enumeration on every read.
func TestGPURoutesDisabledSkipsDeviceDiscovery(t *testing.T) {
	defer SetGPURoutesDisabled(false)

	opens := 0
	newRuntime := func() *gpuDecodeRuntime {
		opens = 0
		return newGPUDecodeRuntime(newGPUDeviceCache(func() (*vulki.Device, error) {
			opens++
			return nil, errors.New("no device in this test")
		}))
	}
	// Large enough to clear the automatic workload threshold, so a session is
	// declined for the switch and nothing else.
	base := &core.Bitmap{Width: 4000, Height: 3000, Channels: 4}

	SetGPURoutesDisabled(true)
	session, err := newRuntime().begin(base, 1)
	if err != nil || session != nil {
		t.Fatalf("disabled begin returned session %v, error %v", session, err)
	}
	if opens != 0 {
		t.Fatalf("disabled begin opened a device %d times", opens)
	}

	SetGPURoutesDisabled(false)
	if _, err := newRuntime().begin(base, 1); err != nil {
		t.Fatalf("enabled begin: %v", err)
	}
	if opens != 1 {
		t.Fatalf("enabled begin opened a device %d times, want 1", opens)
	}
}
