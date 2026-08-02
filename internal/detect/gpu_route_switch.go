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

// SetGPURoutesDisabled turns the automatic GPU routes off or on. It is a
// debugging control, not a tuning knob: leaving it on is the supported
// configuration, and no decision inside the decoder reads it.
//
// The scope is every automatic session acquisition, native and browser, plus a
// stream's session refresh, which is checked separately because a live stream
// reuses its session rather than reacquiring one and would otherwise never
// consult this again. A stream therefore changes route on its next frame. It
// does not reach a session a caller opened explicitly against a device it
// supplied, and it does not interrupt a decode already running.
func SetGPURoutesDisabled(disabled bool) {
	gpuRoutesDisabled.Store(disabled)
}

// GPURoutesDisabled reports the switch's state.
func GPURoutesDisabled() bool {
	return gpuRoutesDisabled.Load()
}
