package detect

import (
	"bytes"
	"fmt"
	"image"
	"runtime"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/encode"
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

	route, err := ladder.newRouteCanvas()
	if err != nil {
		t.Fatalf("new GPU route canvas: %v", err)
	}
	t.Cleanup(func() {
		if err := route.Close(); err != nil {
			t.Errorf("close GPU route canvas: %v", err)
		}
	})
	tests := []struct {
		name  string
		level int
		crop  image.Rectangle
		angle float64
		exact bool
	}{
		{name: "identity", level: 0, crop: cpuLevels[0].Bounds(), exact: true},
		{name: "whole frame", level: 0, crop: cpuLevels[0].Bounds(), angle: 30},
		{
			name:  "region",
			level: 1,
			crop:  image.Rect(7, 5, cpuLevels[1].Rect.Dx()-9, cpuLevels[1].Rect.Dy()-11),
			angle: 135,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			size, err := route.rotate(test.level, test.crop, test.angle)
			if err != nil {
				t.Fatalf("rotate GPU canvas: %v", err)
			}
			got, err := route.download()
			if err != nil {
				t.Fatalf("download GPU route canvas: %v", err)
			}
			want := RotateToBitmap(CropImage(cpuLevels[test.level], test.crop), test.angle)
			if size != image.Pt(want.Width, want.Height) || got.Width != want.Width || got.Height != want.Height {
				t.Fatalf(
					"rotated dimensions = %v and %dx%d, want %dx%d",
					size, got.Width, got.Height, want.Width, want.Height,
				)
			}
			if test.exact {
				if !bytes.Equal(got.Pix, want.Pix) {
					t.Fatal("identity GPU rotation differs from CPU output")
				}
			} else {
				differing, maxDelta := gpuCanvasDifference(got, want)
				t.Logf("%d differing components, maximum delta %d", differing, maxDelta)
				if maxDelta > 1 {
					t.Fatalf("GPU rotation maximum component delta = %d, want at most 1", maxDelta)
				}
			}
			reportGPUCanvasBinarizationDifference(t, got, want)
		})
	}
}

func TestGPUCanvasFinderParity(t *testing.T) {
	rendered, err := encode.Render(encode.Config{
		Colors:       8,
		ModuleSize:   12,
		SymbolNumber: 1,
	}, []byte("resident GPU finder parity"))
	if err != nil {
		t.Fatalf("encode finder parity symbol: %v", err)
	}
	base := RotateToBitmap(rendered.Image, -30)
	ladder, err := newGPUCanvasLadder(base.Width, base.Height, 1)
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
		t.Fatalf("upload GPU finder parity canvas: %v", err)
	}
	route, err := ladder.newRouteCanvas()
	if err != nil {
		t.Fatalf("new GPU finder parity route canvas: %v", err)
	}
	t.Cleanup(func() {
		if err := route.Close(); err != nil {
			t.Errorf("close GPU finder parity route canvas: %v", err)
		}
	})
	if _, err := route.rotate(0, image.Rect(0, 0, base.Width, base.Height), 30); err != nil {
		t.Fatalf("rotate GPU finder parity canvas: %v", err)
	}
	got, err := route.download()
	if err != nil {
		t.Fatalf("download GPU finder parity canvas: %v", err)
	}
	want := RotateToBitmap(base.NRGBA(), 30)
	gotFound, gotSide := gpuCanvasFinders(got)
	wantFound, wantSide := gpuCanvasFinders(want)
	if gotFound != wantFound || gotSide != wantSide {
		t.Fatalf(
			"GPU finder result = %v, %v; CPU result = %v, %v",
			gotFound, gotSide, wantFound, wantSide,
		)
	}
	if !gotFound {
		t.Fatal("finder parity symbol was not detected")
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

			if err := ladder.UploadAndBuild(base); err != nil {
				b.Fatal(err)
			}
			route, err := ladder.newRouteCanvas()
			if err != nil {
				b.Fatal(err)
			}
			b.Cleanup(func() {
				if err := route.Close(); err != nil {
					b.Errorf("close GPU route canvas: %v", err)
				}
			})
			level := ladder.levels[1]
			crop := image.Rect(0, 0, level.width, level.height)
			cpuLevel := HalveNRGBA(base.NRGBA())
			b.Run("CPU-rotate", func(b *testing.B) {
				var got *core.Bitmap
				b.ReportAllocs()
				for b.Loop() {
					got = RotateToBitmap(cpuLevel, 45)
				}
				runtime.KeepAlive(got)
			})
			b.Run("GPU-resident-rotate", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if _, err := route.rotate(1, crop, 45); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("GPU-rotate-download", func(b *testing.B) {
				var got *core.Bitmap
				b.ReportAllocs()
				for b.Loop() {
					if _, err := route.rotate(1, crop, 45); err != nil {
						b.Fatal(err)
					}
					got, err = route.download()
					if err != nil {
						b.Fatal(err)
					}
				}
				runtime.KeepAlive(got)
			})
		})
	}
}
