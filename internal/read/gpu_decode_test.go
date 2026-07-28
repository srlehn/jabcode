//go:build !js

package read

import (
	"bytes"
	"errors"
	"image"
	"reflect"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/encode"
)

func TestGPUDecodePyramidLevelParity(t *testing.T) {
	payload := []byte("resident GPU decode parity")
	img, err := encode.Run(encode.Config{
		Colors:       8,
		ModuleSize:   12,
		SymbolNumber: 1,
	}, payload)
	if err != nil {
		t.Fatalf("encode GPU decode parity symbol: %v", err)
	}
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	base := core.BitmapFromImage(img)
	session, err := detect.NewGPUDecodeSessionWithDevice(device, base, 1)
	if err != nil {
		_ = device.Close()
		t.Fatalf("new GPU decode session: %v", err)
	}
	t.Cleanup(func() {
		if err := session.Close(); err != nil {
			t.Errorf("close GPU decode session: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close GPU decode device: %v", err)
		}
	})

	var wantFinding finding
	wantData, wantStage, wantEvidence := decodeBitmapFindingTracedCapabilities(
		core.BitmapFromImage(img),
		nil,
		&wantFinding,
		nil,
		compiledCapabilities(),
	)
	var gotFinding finding
	gotData, gotStage, gotEvidence := decodePyramidLevelFindingCapabilities(
		func() image.Image { return img },
		nil,
		&gotFinding,
		nil,
		compiledCapabilities(),
		session,
		0,
	)
	if !equalMessages(gotData, wantData) {
		t.Fatalf("GPU decode payload = %q, CPU payload = %q", messageTransmission(gotData), messageTransmission(wantData))
	}
	if gotStage != wantStage || gotEvidence != wantEvidence {
		t.Fatalf(
			"GPU decode stage/evidence = %v/%v, CPU = %v/%v",
			gotStage,
			gotEvidence,
			wantStage,
			wantEvidence,
		)
	}
	if !reflect.DeepEqual(gotFinding, wantFinding) {
		t.Fatalf("GPU finding = %+v, CPU finding = %+v", gotFinding, wantFinding)
	}
}

func TestDecodePyramidGPUUnavailableFallsBack(t *testing.T) {
	payload := []byte("automatic GPU fallback parity")
	img, err := encode.Run(encode.Config{
		Colors:       8,
		ModuleSize:   32,
		SymbolNumber: 1,
	}, payload)
	if err != nil {
		t.Fatalf("encode automatic GPU fallback symbol: %v", err)
	}
	p := newPyramid(img)
	if p == nil || p.count() < 2 {
		t.Fatal("automatic GPU fallback image does not hold at least 2 pyramid levels")
	}
	openCalls := 0
	data, _, ok := decodePyramidCapabilitiesWithGPU(
		p,
		nil,
		compiledCapabilities(),
		func(*core.Bitmap, int) (*detect.GPUDecodeSession, error) {
			openCalls++
			return nil, errors.New("forced Vulkan failure")
		},
	)
	if !ok {
		t.Fatal("CPU fallback did not decode after forced Vulkan failure")
	}
	if !bytes.Equal(messageTransmission(data), isoPayload(payload)) {
		t.Fatalf("CPU fallback payload = %q, want %q", messageTransmission(data), isoPayload(payload))
	}
	if openCalls != 1 {
		t.Fatalf("automatic GPU session factory called %d times, want once", openCalls)
	}
}
