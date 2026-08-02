package detect

import "sync/atomic"

// gpuRoutesDisabled forces every automatic decode onto the CPU path.
//
// It exists for manual A/B testing of the two routes, which needs the arms to
// differ in one thing only. The alternative in use until now was hiding the
// Vulkan loader with VK_ICD_FILENAMES, which also changes device enumeration,
// driver logging and process startup, so a difference measured that way is not
// attributable to the route.
var gpuRoutesDisabled atomic.Bool

// SetGPURoutesDisabled turns the automatic GPU routes off or on. GPU routing
// enabled is the supported default; disabling it is a debugging control for
// comparing the two routes, not a tuning knob.
//
// It is read only where a route decides to acquire or keep a device session:
// the native and browser automatic constructors, and a stream's session
// refresh. The refresh checks it separately because a live stream reuses its
// session rather than reacquiring one, so an acquisition-time check alone would
// never be consulted again; a stream therefore changes route on its next frame.
// Nothing further down reads it - no detection, sampling or correction step
// branches on it - so it selects a route and never alters what a route does.
// It does not reach a session a caller opened explicitly against a device it
// supplied, and it does not interrupt a decode already running.
func SetGPURoutesDisabled(disabled bool) {
	gpuRoutesDisabled.Store(disabled)
}

// GPURoutesDisabled reports the switch's state.
func GPURoutesDisabled() bool {
	return gpuRoutesDisabled.Load()
}
