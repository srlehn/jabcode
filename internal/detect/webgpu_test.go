//go:build js

package detect

import (
	"image"
	"math/rand"
	"sync"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
)

// Parity tests share one long-lived device, matching the backend's real
// ownership. It is opened once and never closed inside the process: wgpu-native
// under Deno invalidates a fresh device's buffers if a prior device was
// destroyed in the same process, and the backend opens exactly one device, so
// exercising open/destroy/open is neither realistic nor stable here.
var (
	testDeviceOnce sync.Once
	testDevice     *webgpuDevice
	testDeviceErr  error
)

func webgpuTestDevice(t *testing.T) *webgpuDevice {
	t.Helper()
	if !webgpuPresent() {
		t.Skip("no navigator.gpu in this runtime")
	}
	testDeviceOnce.Do(func() { testDevice, testDeviceErr = openWebGPUDevice() })
	if testDeviceErr != nil {
		t.Fatalf("open WebGPU device: %v", testDeviceErr)
	}
	return testDevice
}

// TestWebGPUHalveMatchesCPU gates the resident halve kernel against the actual
// CPU pipeline function (HalveNRGBA), not a reimplementation of its arithmetic.
// It runs only where WebGPU is present (Deno, headless Chromium); the Node wasm
// runner has no navigator.gpu and the test skips there.
func TestWebGPUHalveMatchesCPU(t *testing.T) {
	dev := webgpuTestDevice(t)

	// Sizes straddle the workgroup tile (8x8) and odd/even dimensions so the
	// ceil-halving and the box's remainder handling are both exercised.
	for _, dim := range [][2]int{{4, 4}, {17, 9}, {33, 33}, {256, 181}, {640, 480}} {
		w, h := dim[0], dim[1]
		in := image.NewNRGBA(image.Rect(0, 0, w, h))
		rng := rand.New(rand.NewSource(int64(w*100003 + h)))
		for i := range in.Pix {
			in.Pix[i] = byte(rng.Intn(256))
		}
		want := HalveNRGBA(in)
		got, err := dev.halveNRGBA(in)
		if err != nil {
			t.Fatalf("halve %dx%d: %v", w, h, err)
		}
		if got.Rect != want.Rect {
			t.Fatalf("halve %dx%d: rect got %v want %v", w, h, got.Rect, want.Rect)
		}
		for i := range want.Pix {
			if got.Pix[i] != want.Pix[i] {
				t.Fatalf("halve %dx%d: byte %d got %d want %d", w, h, i, got.Pix[i], want.Pix[i])
			}
		}
	}
}

// TestWebGPUBalanceMatchesCPU gates the multi-stage histogram -> bounds ->
// balance chain against the actual CPU pipeline function BalanceRGB. Content
// regimes exercise the count-above-20 bound rule and the degenerate empty-range
// case (uniform frame) that BalanceRGB reproduces rather than sanitises.
func TestWebGPUBalanceMatchesCPU(t *testing.T) {
	dev := webgpuTestDevice(t)

	fill := func(bm *core.Bitmap, kind string, seed int64) {
		rng := rand.New(rand.NewSource(seed))
		for i := range bm.Pix {
			if bm.Channels == 4 && i%4 == 3 {
				bm.Pix[i] = 255
				continue
			}
			switch kind {
			case "uniform":
				bm.Pix[i] = 128
			case "narrow":
				bm.Pix[i] = byte(120 + rng.Intn(16))
			case "twolevel":
				if rng.Intn(2) == 0 {
					bm.Pix[i] = 40
				} else {
					bm.Pix[i] = 210
				}
			default:
				bm.Pix[i] = byte(rng.Intn(256))
			}
		}
	}

	for _, ch := range []int{3, 4} {
		for _, kind := range []string{"noise", "uniform", "narrow", "twolevel"} {
			for _, dim := range [][2]int{{16, 16}, {129, 77}} {
				w, h := dim[0], dim[1]
				want := core.NewBitmap(w, h, ch)
				fill(want, kind, int64(w*7+h*13+ch))
				got := core.NewBitmap(w, h, ch)
				copy(got.Pix, want.Pix)

				BalanceRGB(want)
				if err := dev.balanceRGB(got); err != nil {
					t.Fatalf("balance ch=%d %s %dx%d: %v", ch, kind, w, h, err)
				}
				for i := range want.Pix {
					if got.Pix[i] != want.Pix[i] {
						t.Fatalf("balance ch=%d %s %dx%d: byte %d got %d want %d",
							ch, kind, w, h, i, got.Pix[i], want.Pix[i])
					}
				}
			}
		}
	}
}
