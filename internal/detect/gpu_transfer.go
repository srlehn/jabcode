//go:build !js

package detect

import (
	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/phaseprobe"
)

// recordGPUUpdate keeps command-buffer updates in the transfer census. Vulkan
// embeds their bytes in the host-built command stream and copies them into the
// device buffer during execution, so omitting them would make a one-upload
// route claim describe the API spelling rather than the traffic.
func recordGPUUpdate(
	recorder *vulki.Recorder,
	label string,
	buffer *vulki.Buffer,
	offset uint64,
	data []byte,
) error {
	phaseprobe.Count(label, len(data))
	return recorder.Update(buffer, offset, data)
}
