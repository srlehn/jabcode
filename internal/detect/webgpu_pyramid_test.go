//go:build js

package detect

import (
	"github.com/srlehn/jabcode/internal/testutil"
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

func TestWebGPUReplaceBaseMatchesCPU(t *testing.T) {
	device := webgpuTestDevice(t)
	first := image.NewNRGBA(image.Rect(0, 0, 129, 77))
	second := image.NewNRGBA(first.Rect)
	for i := range first.Pix {
		first.Pix[i] = byte((i * 17) & 255)
		second.Pix[i] = byte((i*31 + 9) & 255)
	}
	for i := 3; i < len(first.Pix); i += 4 {
		first.Pix[i] = 255
		second.Pix[i] = 255
	}
	pyramid, err := newWebGPUPyramid(device, first, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer pyramid.close()
	session := &GPUDecodeSession{device: device, pyramid: pyramid}
	if err := session.ReplaceBase(core.BitmapFromImage(second)); err != nil {
		t.Fatal(err)
	}
	want := second
	for level := range 3 {
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
	thresholds := []float32{61, 73, 89}
	want := BinarizerRGB(bm, thresholds)
	got, err := device.webgpuBinarizeRGBWithThresholds(bm, thresholds, false)
	if err != nil {
		t.Fatalf("adaptive thresholds: %v", err)
	}
	for channel := range got {
		for i := range want[channel].Pix {
			if got[channel].Pix[i] != want[channel].Pix[i] {
				t.Fatalf("adaptive thresholds channel=%d byte=%d got=%d want=%d", channel, i, got[channel].Pix[i], want[channel].Pix[i])
			}
		}
	}
}

func TestWebGPUDescreenRetryMatchesCPU(t *testing.T) {
	device := webgpuTestDevice(t)
	bm := core.NewBitmap(129, 77, 4)
	for i := range bm.Pix {
		bm.Pix[i] = byte((i*37 + 5) & 255)
	}
	for i := 3; i < len(bm.Pix); i += 4 {
		bm.Pix[i] = 255
	}
	usage := device.usageStorage | device.usageCopyDst | device.usageCopySrc
	input := device.newBuffer(len(bm.Pix), usage)
	defer input.Call("destroy")
	device.writeBytes(input, bm.Pix)
	filtered, err := device.webgpuDescreenResident(input, bm.Width, bm.Height, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer filtered.Call("destroy")
	got, err := device.webgpuBinarizeResident(filtered, bm, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	want := BinarizerRGB(descreen(bm, 2, 3, nil), nil)
	for channel := range got {
		for i := range want[channel].Pix {
			if got[channel].Pix[i] != want[channel].Pix[i] {
				t.Fatalf("descreen channel=%d byte=%d got=%d want=%d", channel, i, got[channel].Pix[i], want[channel].Pix[i])
			}
		}
	}
}
