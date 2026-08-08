//go:build !js

package read

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"math"
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
	if diff := findingDifference(gotFinding, wantFinding); diff != "" {
		t.Fatalf("GPU finding = %+v, CPU finding = %+v: %s", gotFinding, wantFinding, diff)
	}
}

// findingGeometryTolerance bounds how far the two routes' finder centres and
// module sizes may sit apart, in pixels. The device folds candidate centres as
// a running mean in f32 where the host arm sums in f64, so the two differ by
// about sqrt(merges) ulps of the coordinate: a few thousandths of a pixel even
// on a 4096-pixel frame with hundreds of merges. A sixteenth of a pixel is far
// above that and far below the smallest module any route samples, so it cannot
// hide a difference that would move a grid.
//
// The exact equality this replaces was never contracted. It held because the
// row pass ran the same host code on both arms; the device now selects the quad
// where the candidates lie, and the acceptance gate for that is the capture
// census rather than a float comparison.
const findingGeometryTolerance = 1.0 / 16

// findingDifference reports how two routes' findings disagree, or "" when they
// agree. Everything discrete is compared exactly: a differing side size, family
// or scan direction is a different read, not a rounding difference.
func findingDifference(got, want finding) string {
	switch {
	case got.located != want.located:
		return fmt.Sprintf("located %v against %v", got.located, want.located)
	case got.side != want.side:
		return fmt.Sprintf("side %v against %v", got.side, want.side)
	case got.family != want.family:
		return fmt.Sprintf("family %v against %v", got.family, want.family)
	case got.deg != want.deg:
		return fmt.Sprintf("scan direction %v against %v", got.deg, want.deg)
	case !reflect.DeepEqual(got.payload, want.payload):
		return "payload differs"
	}
	for corner := range got.quad {
		if math.Abs(got.quad[corner].X-want.quad[corner].X) > findingGeometryTolerance ||
			math.Abs(got.quad[corner].Y-want.quad[corner].Y) > findingGeometryTolerance {
			return fmt.Sprintf(
				"corner %d at %v against %v", corner, got.quad[corner], want.quad[corner],
			)
		}
		if math.Abs(got.sizes[corner]-want.sizes[corner]) > findingGeometryTolerance {
			return fmt.Sprintf(
				"corner %d module size %v against %v",
				corner, got.sizes[corner], want.sizes[corner],
			)
		}
	}
	return ""
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
