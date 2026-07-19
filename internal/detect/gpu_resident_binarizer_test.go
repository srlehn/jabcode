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

func TestGPUResidentBinarizerParity(t *testing.T) {
	const maxWidth = 257
	const maxHeight = 193
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	input, err := device.NewBuffer(maxWidth * maxHeight * 4)
	if err != nil {
		_ = device.Close()
		t.Fatalf("allocate resident GPU test input: %v", err)
	}
	resident, err := newGPUResidentBinarizerWithDevice(device, maxWidth, maxHeight)
	if err != nil {
		_ = input.Close()
		_ = device.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		if err := resident.Close(); err != nil {
			t.Errorf("close resident GPU binarizer: %v", err)
		}
		if err := input.Close(); err != nil {
			t.Errorf("close resident GPU test input: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close resident GPU test device: %v", err)
		}
	})

	tests := []struct {
		name        string
		width       int
		height      int
		prepare     func(*core.Bitmap)
		thresholds  []float32
		printLevels bool
	}{
		{name: "single block", width: 61, height: 47},
		{name: "multiple blocks", width: maxWidth, height: maxHeight},
		{
			name:    "limited range",
			width:   maxWidth,
			height:  maxHeight,
			prepare: limitGPUResidentRange,
		},
		{
			name:       "fixed thresholds",
			width:      131,
			height:     97,
			thresholds: []float32{83.5, 119.25, 147.75},
		},
		{
			name:        "print levels",
			width:       maxWidth,
			height:      maxHeight,
			printLevels: true,
		},
		{
			name:    "flat channel",
			width:   73,
			height:  69,
			prepare: flattenGPUResidentChannels,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			bm := gpuTestBitmap(test.width, test.height)
			if test.prepare != nil {
				test.prepare(bm)
			}
			wantBalanced := cloneGPUResidentBitmap(bm)
			BalanceRGB(wantBalanced)
			want := binarizeRGB(wantBalanced, test.thresholds, test.printLevels)
			if err := input.Upload(bm.Pix); err != nil {
				t.Fatalf("upload resident GPU test input: %v", err)
			}
			got, _, materialize, err := resident.Binarize(
				input,
				test.width,
				test.height,
				test.thresholds,
				test.printLevels,
				0,
			)
			if err != nil {
				t.Fatalf("resident GPU Binarize: %v", err)
			}
			if err := materialize(); err != nil {
				t.Fatalf("materialize resident GPU masks: %v", err)
			}
			gotBalanced, err := resident.DownloadBalanced(test.width, test.height)
			if err != nil {
				t.Fatalf("download resident GPU balanced image: %v", err)
			}
			if !bytes.Equal(gotBalanced.Pix, wantBalanced.Pix) {
				t.Fatal("resident GPU RGB balance differs from CPU output")
			}
			assertGPUResidentMasksEqual(t, got, want)
			gotRebinarized, _, rebinarizedMaterialize, err := resident.BinarizeBalanced(
				test.width,
				test.height,
				test.thresholds,
				test.printLevels,
				0,
			)
			if err != nil {
				t.Fatalf("resident GPU BinarizeBalanced: %v", err)
			}
			if err := rebinarizedMaterialize(); err != nil {
				t.Fatalf("materialize resident GPU rebinarized masks: %v", err)
			}
			assertGPUResidentMasksEqual(t, gotRebinarized, want)
		})
	}

	if err := resident.Close(); err != nil {
		t.Fatalf("close resident GPU binarizer: %v", err)
	}
	if device.Closed() {
		t.Fatal("closing resident GPU binarizer closed its borrowed device")
	}
}

func TestGPUResidentCanvasBinarizerParity(t *testing.T) {
	const width = 257
	const height = 193
	base := gpuTestBitmap(width, height)
	ladder, err := newGPUCanvasLadder(width, height, 3)
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", ladder.device.Info().AdapterName)
	resident, err := newGPUResidentBinarizerWithDevice(ladder.device, 400, 400)
	if err != nil {
		_ = ladder.Close()
		t.Fatalf("new resident GPU binarizer: %v", err)
	}
	t.Cleanup(func() {
		if err := resident.Close(); err != nil {
			t.Errorf("close resident GPU binarizer: %v", err)
		}
		if err := ladder.Close(); err != nil {
			t.Errorf("close GPU canvas ladder: %v", err)
		}
	})
	if err := ladder.UploadAndBuild(base); err != nil {
		t.Fatalf("upload and build GPU canvas ladder: %v", err)
	}

	level := ladder.levels[1]
	wantLevel, err := ladder.DownloadLevel(1)
	if err != nil {
		t.Fatalf("download GPU canvas level: %v", err)
	}
	BalanceRGB(wantLevel)
	wantLevelMasks := BinarizerRGB(wantLevel, nil)
	gotLevelMasks, _, levelMaterialize, err := resident.Binarize(level.buffer, level.width, level.height, nil, false, 0)
	if err != nil {
		t.Fatalf("binarize resident GPU level: %v", err)
	}
	if err := levelMaterialize(); err != nil {
		t.Fatalf("materialize resident GPU level masks: %v", err)
	}
	assertGPUResidentMasksEqual(t, gotLevelMasks, wantLevelMasks)

	route, err := ladder.newRouteCanvas()
	if err != nil {
		t.Fatalf("new resident GPU route canvas: %v", err)
	}
	t.Cleanup(func() {
		if err := resident.releaseInputBindings(route.route); err != nil {
			t.Errorf("release resident GPU route bindings: %v", err)
		}
		if err := route.Close(); err != nil {
			t.Errorf("close resident GPU route canvas: %v", err)
		}
	})
	if _, err := route.rotate(0, image.Rect(0, 0, width, height), 30); err != nil {
		t.Fatalf("rotate resident GPU canvas: %v", err)
	}
	wantRoute, err := route.download()
	if err != nil {
		t.Fatalf("download resident GPU route: %v", err)
	}
	BalanceRGB(wantRoute)
	wantRouteMasks := BinarizerRGB(wantRoute, nil)
	gotRouteMasks, _, routeMaterialize, err := resident.Binarize(
		route.route,
		route.width,
		route.height,
		nil,
		false,
		0,
	)
	if err != nil {
		t.Fatalf("binarize resident GPU route: %v", err)
	}
	if err := routeMaterialize(); err != nil {
		t.Fatalf("materialize resident GPU route masks: %v", err)
	}
	assertGPUResidentMasksEqual(t, gotRouteMasks, wantRouteMasks)
}

func limitGPUResidentRange(bm *core.Bitmap) {
	for pixel := 0; pixel < bm.Width*bm.Height; pixel++ {
		for channel := range 3 {
			offset := pixel*4 + channel
			bm.Pix[offset] = 40 + bm.Pix[offset]%160
		}
	}
}

func flattenGPUResidentChannels(bm *core.Bitmap) {
	for pixel := 0; pixel < bm.Width*bm.Height; pixel++ {
		bm.Pix[pixel*4+0] = 83
		bm.Pix[pixel*4+1] = 127
		bm.Pix[pixel*4+2] = 191
	}
}

func cloneGPUResidentBitmap(source *core.Bitmap) *core.Bitmap {
	clone := core.NewBitmap(source.Width, source.Height, source.Channels)
	copy(clone.Pix, source.Pix)
	return clone
}

func assertGPUResidentMasksEqual(
	t *testing.T,
	got, want [3]*core.Bitmap,
) {
	t.Helper()
	for channel := range got {
		if bytes.Equal(got[channel].Pix, want[channel].Pix) {
			continue
		}
		for index := range want[channel].Pix {
			if got[channel].Pix[index] != want[channel].Pix[index] {
				t.Fatalf(
					"channel %d differs at (%d,%d): got %d, want %d",
					channel,
					index%want[channel].Width,
					index/want[channel].Width,
					got[channel].Pix[index],
					want[channel].Pix[index],
				)
			}
		}
	}
}

func BenchmarkGPUResidentBinarizer(b *testing.B) {
	for _, size := range []int{256, 512, 1024, 2048} {
		b.Run(fmt.Sprintf("%dx%d", size, size), func(b *testing.B) {
			device, err := vulki.Open()
			if err != nil {
				b.Skipf("Vulkan unavailable: %v", err)
			}
			input, err := device.NewBuffer(uint64(size * size * 4))
			if err != nil {
				_ = device.Close()
				b.Fatal(err)
			}
			resident, err := newGPUResidentBinarizerWithDevice(device, size, size)
			if err != nil {
				_ = input.Close()
				_ = device.Close()
				b.Fatal(err)
			}
			b.Cleanup(func() {
				if err := resident.Close(); err != nil {
					b.Errorf("close resident GPU binarizer: %v", err)
				}
				if err := input.Close(); err != nil {
					b.Errorf("close resident GPU benchmark input: %v", err)
				}
				if err := device.Close(); err != nil {
					b.Errorf("close resident GPU benchmark device: %v", err)
				}
			})
			base := gpuTestBitmap(size, size)
			limitGPUResidentRange(base)
			if err := input.Upload(base.Pix); err != nil {
				b.Fatal(err)
			}

			b.Run("CPU", func(b *testing.B) {
				var got [3]*core.Bitmap
				working := cloneGPUResidentBitmap(base)
				b.ReportAllocs()
				for b.Loop() {
					BalanceRGB(working)
					got = BinarizerRGB(working, nil)
				}
				runtime.KeepAlive(got)
			})
			b.Run("GPU-resident", func(b *testing.B) {
				var got [3]*core.Bitmap
				b.ReportAllocs()
				for b.Loop() {
					var materialize func() error
					got, _, materialize, err = resident.Binarize(input, size, size, nil, false, 0)
					if err != nil {
						b.Fatal(err)
					}
					if err = materialize(); err != nil {
						b.Fatal(err)
					}
				}
				runtime.KeepAlive(got)
			})
		})
	}
}
