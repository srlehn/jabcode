//go:build !js

package detect

import (
	"bytes"
	"fmt"
	"runtime"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
)

func TestGPUBinarizerParity(t *testing.T) {
	const maxWidth = 257
	const maxHeight = 193
	binarizer, err := newGPUBinarizer(maxWidth, maxHeight)
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", binarizer.device.Info().AdapterName)
	t.Cleanup(func() {
		if err := binarizer.Close(); err != nil {
			t.Errorf("close GPU binarizer: %v", err)
		}
	})

	tests := []struct {
		name        string
		width       int
		height      int
		thresholds  []float32
		printLevels bool
	}{
		{name: "single block", width: 61, height: 47},
		{name: "multiple blocks", width: maxWidth, height: maxHeight},
		{name: "fixed thresholds", width: 131, height: 97, thresholds: []float32{83.5, 119.25, 147.75}},
		{name: "print levels", width: maxWidth, height: maxHeight, printLevels: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			bm := gpuTestBitmap(test.width, test.height)
			want := binarizeRGB(bm, test.thresholds, test.printLevels)
			got, err := binarizer.Binarize(bm, test.thresholds, test.printLevels)
			if err != nil {
				t.Fatalf("GPU Binarize: %v", err)
			}
			for channel := range 3 {
				if bytes.Equal(got[channel].Pix, want[channel].Pix) {
					continue
				}
				for index := range want[channel].Pix {
					if got[channel].Pix[index] != want[channel].Pix[index] {
						t.Fatalf(
							"channel %d differs at (%d,%d): got %d, want %d",
							channel, index%test.width, index/test.width,
							got[channel].Pix[index], want[channel].Pix[index],
						)
					}
				}
			}
		})
	}
}

func TestGPUBinarizerBorrowedDevice(t *testing.T) {
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
	binarizer, err := newGPUBinarizerPipelineWithDevice(device, kernels, 64, 64, true)
	if err != nil {
		t.Fatalf("new GPU binarizer: %v", err)
	}
	if err := binarizer.Close(); err != nil {
		t.Fatalf("close GPU binarizer: %v", err)
	}
	if err := kernels.Close(); err != nil {
		t.Fatalf("close GPU kernel set: %v", err)
	}
	if device.Closed() {
		t.Fatal("closing GPU binarizer closed its borrowed device")
	}
}

func BenchmarkGPUBinarizer(b *testing.B) {
	for _, size := range []int{256, 512, 1024, 2048} {
		b.Run(fmt.Sprintf("%dx%d", size, size), func(b *testing.B) {
			bm := gpuTestBitmap(size, size)
			BalanceRGB(bm)
			binarizer, err := newGPUBinarizer(size, size)
			if err != nil {
				b.Skipf("Vulkan unavailable: %v", err)
			}
			b.Cleanup(func() {
				if err := binarizer.Close(); err != nil {
					b.Errorf("close GPU binarizer: %v", err)
				}
			})
			params, thresholds := gpuBinarizerInputs(bm, nil, false)
			if err := binarizer.input.Upload(bm.Pix); err != nil {
				b.Fatal(err)
			}
			if err := binarizer.thresholds.Upload(thresholds); err != nil {
				b.Fatal(err)
			}
			if err := binarizer.params.Upload(params); err != nil {
				b.Fatal(err)
			}
			if err := runBenchmarkGPUCompute(b, binarizer, size); err != nil {
				b.Fatal(err)
			}
			pixelGroups := vulki.Workgroups{
				X: uint32((size + gpuBinarizerWorkgroupWidth - 1) / gpuBinarizerWorkgroupWidth),
				Y: uint32((size + gpuBinarizerWorkgroupHeight - 1) / gpuBinarizerWorkgroupHeight),
				Z: 1,
			}
			packGroups := vulki.Workgroups{
				X: uint32(((size*size+7)/8 + gpuPackWorkgroupSize - 1) / gpuPackWorkgroupSize),
				Y: 1,
				Z: 1,
			}

			b.Run("CPU", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					BinarizerRGB(bm, nil)
				}
			})
			b.Run("GPU", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if _, err := binarizer.Binarize(bm, nil, false); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("inputs-floor", func(b *testing.B) {
				var gotParams, gotThresholds []byte
				b.ReportAllocs()
				for b.Loop() {
					gotParams, gotThresholds = gpuBinarizerInputs(bm, nil, false)
				}
				runtime.KeepAlive(gotParams)
				runtime.KeepAlive(gotThresholds)
			})
			b.Run("unpack-floor", func(b *testing.B) {
				packedMasks := binarizer.hostMasks[:((size*size+7)/8)*4]
				var got [3]*core.Bitmap
				b.ReportAllocs()
				for b.Loop() {
					got = unpackGPUBinarizerMasks(bm, packedMasks)
				}
				runtime.KeepAlive(got)
			})
			b.Run("compute-floor", func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for b.Loop() {
					if err := runBenchmarkGPUCompute(b, binarizer, size); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("submit-floor", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					recorder, err := binarizer.device.NewRecorder()
					if err != nil {
						b.Fatal(err)
					}
					if err := recorder.SubmitAndWait(); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("classify-floor", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if err := runBenchmarkGPUStage(b, binarizer, binarizer.classify, pixelGroups); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("filter-floor", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if err := runBenchmarkGPUStage(b, binarizer, binarizer.filter, pixelGroups); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("pack-floor", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if err := runBenchmarkGPUStage(b, binarizer, binarizer.pack, packGroups); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("upload-floor", func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					recorder, err := binarizer.device.NewRecorder()
					if err != nil {
						b.Fatal(err)
					}
					if err := recorder.Upload(binarizer.input, 0, bm.Pix); err != nil {
						_ = recorder.Abort()
						b.Fatal(err)
					}
					if err := recorder.SubmitAndWait(); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("download-floor", func(b *testing.B) {
				download := make([]byte, ((size*size+7)/8)*4)
				b.ReportAllocs()
				for b.Loop() {
					recorder, err := binarizer.device.NewRecorder()
					if err != nil {
						b.Fatal(err)
					}
					if err := recorder.Download(binarizer.packedMasks, 0, download); err != nil {
						_ = recorder.Abort()
						b.Fatal(err)
					}
					if err := recorder.SubmitAndWait(); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("transfer-floor", func(b *testing.B) {
				download := make([]byte, ((size*size+7)/8)*4)
				b.ReportAllocs()
				for b.Loop() {
					recorder, err := binarizer.device.NewRecorder()
					if err != nil {
						b.Fatal(err)
					}
					if err := recorder.Upload(binarizer.input, 0, bm.Pix); err != nil {
						_ = recorder.Abort()
						b.Fatal(err)
					}
					if err := recorder.Download(binarizer.packedMasks, 0, download); err != nil {
						_ = recorder.Abort()
						b.Fatal(err)
					}
					if err := recorder.SubmitAndWait(); err != nil {
						b.Fatal(err)
					}
				}
			})
		})
	}
}

func runBenchmarkGPUCompute(b *testing.B, binarizer *gpuBinarizer, size int) error {
	b.Helper()
	recorder, err := binarizer.device.NewRecorder()
	if err != nil {
		return err
	}
	if err := binarizer.recordCompute(recorder, size, size); err != nil {
		_ = recorder.Abort()
		return err
	}
	return recorder.SubmitAndWait()
}

func runBenchmarkGPUStage(
	b *testing.B,
	binarizer *gpuBinarizer,
	stage gpuBinarizerStage,
	groups vulki.Workgroups,
) error {
	b.Helper()
	recorder, err := binarizer.device.NewRecorder()
	if err != nil {
		return err
	}
	if err := recorder.Dispatch(stage.kernel, stage.bindings, groups); err != nil {
		_ = recorder.Abort()
		return err
	}
	return recorder.SubmitAndWait()
}

func gpuTestBitmap(width, height int) *core.Bitmap {
	bm := core.NewBitmap(width, height, 4)
	state := uint32(0x9e3779b9)
	for index := 0; index < width*height; index++ {
		state = state*1664525 + 1013904223
		bm.Pix[index*4+0] = byte(state >> 24)
		state = state*1664525 + 1013904223
		bm.Pix[index*4+1] = byte(state >> 24)
		state = state*1664525 + 1013904223
		bm.Pix[index*4+2] = byte(state >> 24)
		bm.Pix[index*4+3] = 255
	}
	return bm
}
