//go:build js

package detect

import (
	"image"
	"math/rand"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
)

func TestWebGPUPyramidMatchesCPU(t *testing.T) {
	device := webgpuTestDevice(t)
	base := image.NewNRGBA(image.Rect(0, 0, 129, 77))
	rng := rand.New(rand.NewSource(17))
	for i := range base.Pix {
		base.Pix[i] = byte(rng.Intn(256))
	}
	pyramid, err := newWebGPUPyramid(device, base, 4)
	if err != nil {
		t.Fatal(err)
	}
	defer pyramid.close()
	want := base
	for level := range 4 {
		got, err := pyramid.download(level)
		if err != nil {
			t.Fatalf("download level %d: %v", level, err)
		}
		if got.Rect != want.Rect {
			t.Fatalf("level %d rect got %v want %v", level, got.Rect, want.Rect)
		}
		for i := range want.Pix {
			if got.Pix[i] != want.Pix[i] {
				t.Fatalf("level %d byte %d got %d want %d", level, i, got.Pix[i], want.Pix[i])
			}
		}
		want = HalveNRGBA(want)
	}
}

func TestWebGPUBinarizeMatchesCPU(t *testing.T) {
	device := webgpuTestDevice(t)
	bm := core.NewBitmap(129, 77, 4)
	rng := rand.New(rand.NewSource(23))
	for i := range bm.Pix {
		bm.Pix[i] = byte(rng.Intn(256))
	}
	for _, printLevels := range []bool{false, true} {
		want := BinarizerRGB(bm, nil)
		if printLevels {
			want = BinarizerRGBPrint(bm)
		}
		got, err := device.webgpuBinarizeRGB(bm, printLevels)
		if err != nil {
			t.Fatalf("binarize print=%v: %v", printLevels, err)
		}
		for channel := range got {
			if len(got[channel].Pix) != len(want[channel].Pix) {
				t.Fatalf("binarize print=%v channel=%d size mismatch", printLevels, channel)
			}
			for i := range want[channel].Pix {
				if got[channel].Pix[i] != want[channel].Pix[i] {
					t.Fatalf("binarize print=%v channel=%d byte=%d got=%d want=%d",
						printLevels, channel, i, got[channel].Pix[i], want[channel].Pix[i])
				}
			}
		}
	}
}

func TestWebGPURoutePreparation(t *testing.T) {
	device := webgpuTestDevice(t)
	base := image.NewNRGBA(image.Rect(0, 0, 129, 77))
	for i := range base.Pix {
		base.Pix[i] = byte((i * 29) & 255)
	}
	pyramid, err := newWebGPUPyramid(device, base, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer pyramid.close()
	session := &GPUDecodeSession{device: device, pyramid: pyramid}
	detector, _, size, err := session.LocateRouteFamilies(
		0, base.Bounds(), 17, FinderFamilyCurrent.Mask(), IntensiveDetect, nil, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if detector == nil || size.X <= 0 || size.Y <= 0 {
		t.Fatalf("route returned detector=%v size=%v", detector != nil, size)
	}
}

func TestWebGPUCoarseProbePreparation(t *testing.T) {
	device := webgpuTestDevice(t)
	base := image.NewNRGBA(image.Rect(0, 0, 129, 77))
	for i := range base.Pix {
		base.Pix[i] = byte((i * 11) & 255)
	}
	pyramid, err := newWebGPUPyramid(device, base, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer pyramid.close()
	session := &GPUDecodeSession{device: device, pyramid: pyramid}
	families, handled := session.ProbeLevelFamilies(0, nil)
	if !handled {
		t.Fatal("WebGPU coarse probe was not handled")
	}
	if families == nil {
		t.Fatal("WebGPU coarse probe returned nil families")
	}
	if _, handled := session.ProbeLevelFamilies(0, &CoarseProbeTrace{}); handled {
		t.Fatal("traced coarse probe should use the CPU fallback")
	}
}
