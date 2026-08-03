//go:build !js

package read

import (
	"bytes"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/encode"
	"github.com/srlehn/jabcode/internal/testutil"
)

func coldStartRouteImage(t *testing.T) (*pyramid, []byte) {
	t.Helper()
	payload := []byte("cold-start automatic route")
	img, err := encode.Run(encode.Config{
		Colors: 8, ModuleSize: 32, SymbolNumber: 1,
	}, payload)
	if err != nil {
		t.Fatalf("encode cold-start fixture: %v", err)
	}
	p := newPyramid(img)
	if p == nil {
		t.Fatal("cold-start fixture did not produce a resolution pyramid")
	}
	return p, isoPayload(payload)
}

// A clean row-settled frame races device preparation but returns without
// acquiring a session. Starting the warm before the probe is load-bearing for
// hard reads: starting it afterward serializes the probe and device discovery.
func TestAutomaticGPUColdStartProbeRacesDevicePreparation(t *testing.T) {
	p, want := coldStartRouteImage(t)
	warmed, opened := 0, 0
	data, _, ok := decodePyramidCapabilitiesWithAutomaticPolicy(
		p,
		nil,
		compiledCapabilities(),
		true,
		func() { warmed++ },
		func(*core.Bitmap, int) (*detect.GPUDecodeSession, error) {
			opened++
			return nil, nil
		},
	)
	if !ok || !bytes.Equal(messageTransmission(data), want) {
		t.Fatalf("cold-start probe returned ok=%v payload=%q, want %q", ok, messageTransmission(data), want)
	}
	if warmed != 1 || opened != 0 {
		t.Fatalf("row-settled cold start warmed %d times and opened %d sessions, want one warm and no session", warmed, opened)
	}
}

// The probe is deliberately axis-aligned. Letting it run the directional
// sweep would make a rotated frame look cheap only by completing the expensive
// work on the CPU, suppressing the route that wins that workload on the GPU.
func TestAutomaticGPUColdStartProbeDefersDirectionalReads(t *testing.T) {
	p, _ := coldStartRouteImage(t)
	if !coldStartRowProbeUseful(p.base) {
		t.Fatal("axis-aligned clean frame was not selected for the cold-start probe")
	}
	rotated := testutil.RotateImage(p.base, 45)
	turned := newPyramid(rotated)
	if turned == nil {
		t.Fatal("rotated cold-start fixture did not produce a resolution pyramid")
	}
	if coldStartRowProbeUseful(turned.base) {
		t.Fatal("45-degree frame was selected for the row-only cold-start probe")
	}
	if data, _, ok := decodePyramidColdStartCapabilities(turned, nil, compiledCapabilities()); ok {
		t.Fatalf("axis-aligned cold-start probe decoded a 45-degree frame: %q", messageTransmission(data))
	}

	warmed, opened := 0, 0
	_, _, _ = decodePyramidCapabilitiesWithAutomaticPolicy(
		turned,
		nil,
		compiledCapabilities(),
		true,
		func() { warmed++ },
		func(*core.Bitmap, int) (*detect.GPUDecodeSession, error) {
			opened++
			return nil, nil
		},
	)
	if warmed != 1 || opened != 1 {
		t.Fatalf("directional cold start warmed %d times and opened %d sessions, want one each", warmed, opened)
	}
}
