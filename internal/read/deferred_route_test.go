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
		t.Run(tc.name, func(t *testing.T) {
			if (tc.variant == wire.BSI && !bsiReadEnabled) || (tc.variant == wire.PreV2C && !preV2CReadEnabled) {
				t.Skip("wire family is not compiled in")
			}
			img := loadLegacyCReferenceFixture(t, tc.fixture)
			base := core.BitmapFromImage(img)
			device, err := vulki.Open()
			if err != nil {
				t.Skipf("Vulkan unavailable: %v", err)
			}
			session, err := detect.NewGPUDecodeSessionWithDeviceScanOnly(device, base, 1)
			if err != nil || session == nil {
				_ = device.Close()
				t.Skipf("automatic GPU session unavailable: %v", err)
			}
			t.Cleanup(func() { _ = session.Close(); _ = device.Close() })

			locate := func() (*detect.PrimaryDetector, detect.FinderFamilySet) {
				detector, found, _, err := session.LocateRouteFamilies(
					0, image.Rect(0, 0, base.Width, base.Height), 0,
					finderFamiliesForCapabilities(tc.variant.Mask()), detect.IntensiveDetect, nil, nil,
				)
				if err != nil || detector == nil || !found.Has(tc.family) {
					t.Fatalf("GPU locate: detector=%v found=%v err=%v", detector != nil, found, err)
				}
				assertDeferredLocatedChannels(t, tc.name, detector)
				return detector, found
			}

			detector, found := locate()
			data, _, _ := decodeGPUDetectorCapabilities(detector, found, nil, nil, tc.variant.Mask())
			if data == nil || !bytes.Equal(messageTransmission(data), tc.wantData) {
				t.Fatalf("GPU %s decode = %q, want %q", tc.name, messageTransmission(data), tc.wantData)
			}
			assertDeferredLocatedChannels(t, tc.name+" decode", detector)

			detector, found = locate()
			var stream Stream
			observed := stream.observeLocatedDetector(detector.BM, detector, found, nil, tc.variant.Mask())
			if observed == nil {
				t.Fatal("Stream observation rejected GPU historical route")
			}
			streamData, ok := stream.finishStreamObservation(
				detector.BM, func() [3]*core.Bitmap { return detector.Ch }, observed,
				finding{}, image.Point{}, tc.variant.Mask(),
			)
			if !ok || streamData == nil || !bytes.Equal(messageTransmission(streamData), tc.wantData) {
				t.Fatalf("GPU Stream %s decode = %q, want %q", tc.name, messageTransmission(streamData), tc.wantData)
			}
			assertDeferredLocatedChannels(t, tc.name+" Stream", detector)
		})
	}
}

func assertDeferredLocatedChannels(t *testing.T, route string, detector *detect.PrimaryDetector) {
	t.Helper()
	if got := detector.ChannelExpansionCount(); got != 0 {
		t.Fatalf("GPU %s locate expanded channels %d times", route, got)
	}
	for channel, ch := range detector.Ch {
		if ch.Pix != nil {
			t.Fatalf("GPU %s locate materialized channel %d", route, channel)
		}
	}
}
