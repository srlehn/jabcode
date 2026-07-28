//go:build !js

package detect

import (
	"bytes"
	"fmt"
	"image"
	"runtime"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
)

func TestGPUCanvasLadderParity(t *testing.T) {
	const width = 257
	const height = 193
	const levelCount = 4
	base := gpuTestBitmap(width, height)
	ladder, err := newGPUCanvasLadder(width, height, levelCount)
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", ladder.device.Info().AdapterName)
	t.Cleanup(func() {
		if err := ladder.Close(); err != nil {
			t.Errorf("close GPU canvas ladder: %v", err)
		}
	})
	if err := ladder.UploadAndBuild(base); err != nil {
		t.Fatalf("upload and build GPU canvas ladder: %v", err)
	}

	cpuLevels := []*image.NRGBA{base.NRGBA()}
	for len(cpuLevels) < levelCount {
		cpuLevels = append(cpuLevels, HalveNRGBA(cpuLevels[len(cpuLevels)-1]))
	}
	for index, want := range cpuLevels {
		got, err := ladder.DownloadLevel(index)
		if err != nil {
			t.Fatalf("download GPU canvas level %d: %v", index, err)
		}
		if got.Width != want.Rect.Dx() || got.Height != want.Rect.Dy() {
			t.Fatalf(
				"level %d dimensions = %dx%d, want %dx%d",
				index, got.Width, got.Height, want.Rect.Dx(), want.Rect.Dy(),
			)
		}
		if !bytes.Equal(got.Pix, want.Pix) {
			t.Fatalf("GPU half-scale level %d differs from CPU output", index)
		}
	}
}

func TestGPUCanvasBorrowedDevice(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	t.Cleanup(func() {
		if err := device.Close(); err != nil {
			t.Errorf("close borrowed device: %v", err)
		}
	})
	kernels := newGPUDecodeKernels(device)
	ladder, err := newGPUCanvasLadderWithDevice(device, kernels, 64, 64, 3)
	if err != nil {
		t.Fatalf("new GPU canvas ladder: %v", err)
	}
	if err := ladder.Close(); err != nil {
		t.Fatalf("close GPU canvas ladder: %v", err)
	}
	if err := kernels.Close(); err != nil {
		t.Fatalf("close GPU kernel set: %v", err)
	}
	if device.Closed() {
		t.Fatal("closing GPU canvas ladder closed its borrowed device")
	}
}

func gpuCanvasFinders(source *core.Bitmap) (bool, image.Point) {
	source = cloneGPUCanvasBitmap(source)
	BalanceRGB(source)
	detector := &PrimaryDetector{
		BM:   source,
		Ch:   BinarizerRGB(source, nil),
		Mode: IntensiveDetect,
	}
	if !detector.LocateFinders() {
		return false, image.Point{}
	}
	return true, CalculateSideSize(source, detector.FPs)
}

func reportGPUCanvasBinarizationDifference(t *testing.T, got, want *core.Bitmap) {
	t.Helper()
	got = cloneGPUCanvasBitmap(got)
	want = cloneGPUCanvasBitmap(want)
	BalanceRGB(got)
	BalanceRGB(want)
	gotChannels := BinarizerRGB(got, nil)
	wantChannels := BinarizerRGB(want, nil)
	for channel := range gotChannels {
		if !bytes.Equal(gotChannels[channel].Pix, wantChannels[channel].Pix) {
			differing, _ := gpuCanvasDifference(gotChannels[channel], wantChannels[channel])
			t.Logf(
				"GPU transform changes %d binarized detector pixels in channel %d",
				differing, channel,
			)
		}
	}
}

func cloneGPUCanvasBitmap(source *core.Bitmap) *core.Bitmap {
	clone := core.NewBitmap(source.Width, source.Height, source.Channels)
	copy(clone.Pix, source.Pix)
	return clone
}

func gpuCanvasDifference(left, right *core.Bitmap) (differing, maxDelta int) {
	for index := range left.Pix {
		delta := int(left.Pix[index]) - int(right.Pix[index])
		if delta < 0 {
			delta = -delta
		}
		if delta != 0 {
			differing++
			maxDelta = max(maxDelta, delta)
		}
	}
	return differing, maxDelta
}

func BenchmarkGPUCanvasLadder(b *testing.B) {
	for _, size := range []int{512, 1024, 2048} {
		b.Run(fmt.Sprintf("%dx%d", size, size), func(b *testing.B) {
			base := gpuTestBitmap(size, size)
			ladder, err := newGPUCanvasLadder(size, size, 3)
			if err != nil {
				b.Skipf("Vulkan unavailable: %v", err)
			}
			b.Cleanup(func() {
				if err := ladder.Close(); err != nil {
					b.Errorf("close GPU canvas ladder: %v", err)
				}
			})

			b.Run("CPU-pyramid", func(b *testing.B) {
				var levels [2]*image.NRGBA
				b.ReportAllocs()
				for b.Loop() {
					levels[0] = HalveNRGBA(base.NRGBA())
					levels[1] = HalveNRGBA(levels[0])
				}
				runtime.KeepAlive(levels)
			})
			b.Run("GPU-pyramid", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if err := ladder.UploadAndBuild(base); err != nil {
						b.Fatal(err)
					}
				}
			})
		})
	}
}
