//go:build !js

package detect

import (
	"errors"
	"testing"
	"time"

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

// Warming exists to move device discovery off the critical path, so it has to
// obey exactly the conditions that decide whether discovery happens at all - a
// warm that opened a device the route would have declined would turn the two
// cases the automatic gate protects, a small frame and a disabled switch, into
// the very cost they avoid. It also has to stay one-shot: the route's own
// acquisition must join this discovery rather than start a second one.
func TestWarmAutomaticGPUDeviceObeysTheAutomaticGate(t *testing.T) {
	defer SetGPURoutesDisabled(false)

	opens := make(chan struct{}, 8)
	newCache := func() *gpuDeviceCache {
		return newGPUDeviceCache(func() (*vulki.Device, error) {
			opens <- struct{}{}
			return nil, errors.New("no device in this test")
		})
	}
	const bigW, bigH = 4000, 3000
	// Warming is asynchronous, so an open that should not happen has to be
	// waited out rather than sampled. A warm that did start opens well inside
	// this window, which is what makes the wait an assertion rather than a
	// guess.
	noOpen := func(what string) {
		t.Helper()
		select {
		case <-opens:
			t.Fatalf("%s opened a device", what)
		case <-time.After(200 * time.Millisecond):
		}
	}

	SetGPURoutesDisabled(true)
	newCache().warm(bigW, bigH)
	noOpen("a disabled warm")

	SetGPURoutesDisabled(false)
	newCache().warm(320, 240)
	noOpen("a warm below the automatic threshold")

	cache := newCache()
	cache.warm(bigW, bigH)
	select {
	case <-opens:
	case <-time.After(5 * time.Second):
		t.Fatal("a warm above the threshold never opened a device")
	}
	if _, err := cache.deviceFor(bigW, bigH); err == nil {
		t.Fatal("deviceFor reported a device this test never provides")
	}
	noOpen("the route's own acquisition after a warm")
}
