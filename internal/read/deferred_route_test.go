//go:build !js && (jabcode_bsi || jabcode_legacy)

package read

import (
	"bytes"
	"image"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/wire"
)

// TestGPUHistoricalRoutesKeepDeferredChannels gates the deferred-mask property
// for the historical wire families end to end. The gate is deliberately
// relative to locate rather than an absolute zero, because what locate itself
// costs is not fixed:
//
// A scan-only session is the permanent default for automatic workloads. The
// device seeds row hits but the bit-identical CPU per-hit chain classifies
// them, and that walk reads concrete mask rows, so locate materializes. A
// device-replay session can chain on the device instead and leave the masks
// packed - but only once the chain kernels finish compiling, and that
// compilation runs in a background goroutine nothing ever blocks on. Whether
// any single device-replay locate defers is therefore a race, and asserting an
// exact per-family count here would only buy a flaky test.
//
// What the wire routes owe regardless is that interpreting a symbol - docked
// traversal included - adds no expansion of its own on top of the finder walk.
// Whenever locate did win the race and leave the masks packed, the stronger
// property is checked too: the channels stay unmaterialized through the whole
// decode, which is the deferred-window path actually being exercised.
func TestGPUHistoricalRoutesKeepDeferredChannels(t *testing.T) {
	tests := []struct {
		name     string
		fixture  string
		variant  wire.Variant
		family   detect.FinderFamily
		wantData []byte
	}{
		{name: "bsi", fixture: "bsi_tr_03137_8c_docked_custom_3x2_5x2.png", variant: wire.BSI, family: detect.FinderFamilyBSI, wantData: []byte("BSI fixed two symbol custom-side oracle")},
		{name: "pre-v2", fixture: "legacy_c_reference_pre_v2_multi.png", variant: wire.PreV2C, family: detect.FinderFamilyBSI, wantData: []byte("Legacy C-reference JAB Code multi-symbol fixture 0123456789")},
	}
	for _, tc := range tests {
		for _, mode := range []struct {
			name   string
			replay bool
		}{{"scan-only", false}, {"device-replay", true}} {
			t.Run(tc.name+"/"+mode.name, func(t *testing.T) {
				if (tc.variant == wire.BSI && !bsiReadEnabled) || (tc.variant == wire.PreV2C && !preV2CReadEnabled) {
					t.Skip("wire family is not compiled in")
				}
				img := loadLegacyCReferenceFixture(t, tc.fixture)
				base := core.BitmapFromImage(img)
				device, err := vulki.Open()
				if err != nil {
					t.Skipf("Vulkan unavailable: %v", err)
				}
				newSession := detect.NewGPUDecodeSessionWithDeviceScanOnly
				if mode.replay {
					newSession = detect.NewGPUDecodeSessionWithDevice
				}
				session, err := newSession(device, base, 1)
				if err != nil || session == nil {
					_ = device.Close()
					t.Skipf("automatic GPU session unavailable: %v", err)
				}
				t.Cleanup(func() { _ = session.Close(); _ = device.Close() })

				locate := func() (*detect.PrimaryDetector, detect.FinderFamilySet, int) {
					detector, found, _, err := session.LocateRouteFamilies(
						0, image.Rect(0, 0, base.Width, base.Height), 0,
						finderFamiliesForCapabilities(tc.variant.Mask()), detect.IntensiveDetect, nil, nil,
					)
					if err != nil || detector == nil || !found.Has(tc.family) {
						t.Fatalf("GPU locate: detector=%v found=%v err=%v", detector != nil, found, err)
					}
					located := detector.ChannelExpansionCount()
					if located == 0 {
						for channel, ch := range detector.Ch {
							if ch.Pix != nil {
								t.Fatalf("GPU %s locate materialized channel %d without counting it", tc.name, channel)
							}
						}
					}
					t.Logf("%s %s: locate expansions=%d", tc.name, mode.name, located)
					return detector, found, located
				}

				detector, found, located := locate()
				data, _, _ := decodeGPUDetectorCapabilities(detector, found, nil, nil, tc.variant.Mask())
				if data == nil || !bytes.Equal(messageTransmission(data), tc.wantData) {
					t.Fatalf("GPU %s decode = %q, want %q", tc.name, messageTransmission(data), tc.wantData)
				}
				assertNoFurtherExpansion(t, tc.name+" decode", detector, located)

				detector, found, located = locate()
				var stream Stream
				observed := stream.observeLocatedDetector(detector.BM, detector, found, nil, tc.variant.Mask())
				if observed == nil {
					t.Fatal("Stream observation rejected GPU historical route")
				}
				assertNoFurtherExpansion(t, tc.name+" Stream observation", detector, located)
				streamData, ok := stream.finishStreamObservation(
					detector.BM, func() [3]*core.Bitmap { return detector.Ch }, observed,
					finding{}, image.Point{}, tc.variant.Mask(),
				)
				if !ok || streamData == nil || !bytes.Equal(messageTransmission(streamData), tc.wantData) {
					t.Fatalf("GPU Stream %s decode = %q, want %q", tc.name, messageTransmission(streamData), tc.wantData)
				}
				assertNoFurtherExpansion(t, tc.name+" Stream", detector, located)
			})
		}
	}
}

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
