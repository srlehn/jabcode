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

// Warming exists to move device and workspace preparation off the critical
// path, so it has to obey exactly the conditions that decide whether that
// preparation happens at all - a warm that opened a device the route would have
// declined would turn the two cases the automatic gate protects, a small frame
// and a disabled switch, into the very cost they avoid. It also has to stay
// one-shot: the route's own acquisition must join this discovery rather than
// start a second one.
func TestWarmAutomaticGPUDecodeObeysTheAutomaticGate(t *testing.T) {
	defer SetGPURoutesDisabled(false)

	opens := make(chan struct{}, 8)
	newRuntime := func() *gpuDecodeRuntime {
		return newGPUDecodeRuntime(newGPUDeviceCache(func() (*vulki.Device, error) {
			opens <- struct{}{}
			return nil, errors.New("no device in this test")
		}))
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
	newRuntime().warm(bigW, bigH, 4)
	noOpen("a disabled warm")

	SetGPURoutesDisabled(false)
	newRuntime().warm(320, 240, 4)
	noOpen("a warm below the automatic threshold")
	newRuntime().warm(bigW, bigH, 0)
	noOpen("a warm for a frame with no pyramid")

	runtime := newRuntime()
	runtime.warm(bigW, bigH, 4)
	select {
	case <-opens:
	case <-time.After(5 * time.Second):
		t.Fatal("a warm above the threshold never opened a device")
	}
	if session, err := runtime.begin(&core.Bitmap{Width: bigW, Height: bigH, Channels: 4}, 4); session != nil || err != nil {
		t.Fatalf("begin returned session %v, error %v on a device this test never provides", session, err)
	}
	noOpen("the route's own acquisition after a warm")
}

// The whole point of preparing early is that the read which started it then
// uses it. begin takes the workspace under TryLock so a second concurrent
// decode drops to the CPU instead of queueing, and an in-flight warm-up holds
// that same lock - so without an explicit join a read would routinely abandon
// the device because its own preparation was still running.
func TestBeginJoinsAnInFlightWarmUp(t *testing.T) {
	runtime := newGPUDecodeRuntime(newGPUDeviceCache(func() (*vulki.Device, error) {
		return nil, errors.New("no device in this test")
	}))
	warming := make(chan struct{})
	runtime.warming = warming

	returned := make(chan struct{})
	go func() {
		defer close(returned)
		_, _ = runtime.begin(&core.Bitmap{Width: 4000, Height: 3000, Channels: 4}, 4)
	}()
	select {
	case <-returned:
		t.Fatal("begin returned while its warm-up was still in flight")
	case <-time.After(200 * time.Millisecond):
	}

	close(warming)
	select {
	case <-returned:
	case <-time.After(5 * time.Second):
		t.Fatal("begin never returned after its warm-up finished")
	}
}
