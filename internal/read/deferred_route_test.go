//go:build !js && (jabcode_bsi || jabcode_legacy)

package read

import (
	"testing"

	"github.com/srlehn/jabcode/internal/detect"
)

// assertNoFurtherExpansion pins the deferred-read property the historical wire
// routes owe: whatever the finder walk already materialized, interpreting the
// symbol adds no expansion of its own. When locate left the masks packed the
// check tightens to full residency, since the counter alone saturates at the
// first expansion and would stop discriminating after it.
func assertNoFurtherExpansion(t *testing.T, route string, detector *detect.PrimaryDetector, located int) {
	t.Helper()
	if got := detector.ChannelExpansionCount(); got != located {
		t.Fatalf("GPU %s expanded channels %d times, want the %d from locate", route, got, located)
	}
	if located != 0 {
		return
	}
	for channel, ch := range detector.Ch {
		if ch.Pix != nil {
			t.Fatalf("GPU %s materialized channel %d after a deferred locate", route, channel)
		}
	}
}
