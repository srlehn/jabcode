//go:build js

package detect

import (
	"image"
	"math/rand"
	"testing"
)

// TestWebGPUHalveMatchesCPU gates the resident halve kernel against the actual
// CPU pipeline function (HalveNRGBA), not a reimplementation of its arithmetic.
// It runs only where WebGPU is present (Deno, headless Chromium); the Node wasm
// runner has no navigator.gpu and the test skips there.
func TestWebGPUHalveMatchesCPU(t *testing.T) {
	if !webgpuPresent() {
		t.Skip("no navigator.gpu in this runtime")
	}
	dev, err := openWebGPUDevice()
	if err != nil {
		t.Fatalf("open WebGPU device: %v", err)
	}
	defer dev.close()

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
