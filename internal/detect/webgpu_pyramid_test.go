//go:build js

package detect

import (
	"image"
	"math/rand"
	"testing"
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
