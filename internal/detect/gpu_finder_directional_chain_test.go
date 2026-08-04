//go:build !js

package detect

import (
	"cmp"
	"math"
	"slices"
	"sync/atomic"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
)

// TestGPUFinderDirectionalChainParity drives the production window and chain
// kernels, then compares their ordered replay with the serial host chain over
// the same raw records. It includes the host-only source-color signal, so the
// gate covers the actual split rather than only the mask kernel.
func TestGPUFinderDirectionalChainParity(t *testing.T) {
	const width, height = 360, 300
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	input, err := device.NewBuffer(width * height * 4)
	if err != nil {
		_ = device.Close()
		t.Fatalf("allocate directional chain input: %v", err)
	}
	resident, err := newGPUResidentBinarizerWithDevice(device, width, height)
	if err != nil {
		_ = input.Close()
		_ = device.Close()
		t.Fatalf("new directional chain resident binarizer: %v", err)
	}
	if err := resident.kernels.compileDirectionalFinderChain(); err != nil {
		_ = resident.Close()
		_ = input.Close()
		_ = device.Close()
		t.Fatalf("compile directional finder chain: %v", err)
	}
	t.Cleanup(func() {
		if err := resident.Close(); err != nil {
			t.Errorf("close directional chain resident binarizer: %v", err)
		}
		if err := input.Close(); err != nil {
			t.Errorf("close directional chain input: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close directional chain device: %v", err)
		}
	})

	bm := chainParityBitmap(width, height, 31, true)
	if err := input.Upload(bm.Pix); err != nil {
		t.Fatalf("upload directional chain input: %v", err)
	}
	var retained atomic.Uint64
	resident.binarizer.onRetainedAllocation = func(delta uint64) { retained.Add(delta) }
	for _, printLevels := range []bool{false, true} {
		channels, _, materialize, err := resident.Binarize(
			input, width, height, nil, printLevels, 1<<currentFamilySeekChannel,
		)
		if err != nil {
			t.Fatalf("binarize directional chain fixture: %v", err)
		}
		if err := materialize(); err != nil {
			t.Fatalf("materialize directional chain masks: %v", err)
		}
		for _, degrees := range []float64{15, 45, 75} {
			direction := newScanDirection(degrees)
			hits, err := resident.ScanDirection(
				width, height, direction, 3, currentFamilySeekChannel,
			)
			if err != nil {
				t.Fatalf("scan directional chain at %.0f degrees: %v", degrees, err)
			}
			if len(hits) == 0 {
				t.Fatalf("directional chain at %.0f degrees produced no hits", degrees)
			}
			if !hits[0].chained {
				t.Fatal("directional scan returned without device chain outcomes")
			}
			if !printLevels && degrees == 15 {
				if got := retained.Load(); got != gpuFinderDirectionalRetainedBytes {
					t.Fatalf("directional buffers charged %d bytes, want %d", got, gpuFinderDirectionalRetainedBytes)
				}
			}
			slices.SortFunc(hits, func(a, b finderDirHit) int {
				if order := cmp.Compare(a.centre.Y, b.centre.Y); order != 0 {
					return order
				}
				if order := cmp.Compare(a.centre.X, b.centre.X); order != 0 {
					return order
				}
				return cmp.Compare(a.module, b.module)
			})

			run := func(chained bool) (*PrimaryDetector, primaryFamilyScan) {
				chainChannels := channels
				if chained {
					for index, channel := range channels {
						chainChannels[index] = &core.Bitmap{
							Width: channel.Width, Height: channel.Height, Channels: channel.Channels,
						}
					}
				}
				d := &PrimaryDetector{BM: bm, Ch: chainChannels, Mode: normalDetect, printPass: printLevels}
				d.Stats.Passes = append(d.Stats.Passes, FinderPassStats{})
				state := newPrimaryFamilyScan()
				if chained {
					d.processDirectionalFamilyHits(direction, hits, &state)
				} else {
					for _, hit := range hits {
						d.processDirectionalFamilyHit(direction, hit.centre, hit.module, &state)
						if state.done {
							break
						}
					}
				}
				return d, state
			}
			gotDetector, got := run(true)
			wantDetector, want := run(false)
			if gotDetector.directionalDeviceChainHits == 0 {
				t.Fatal("production replay did not count any directional device-chain hits")
			}
			if gotDetector.channelExpansions != 0 {
				t.Fatalf("device chain expanded host mask channels %d times", gotDetector.channelExpansions)
			}
			seedDrift := math.Abs(float64(gotDetector.seedModules.len() - wantDetector.seedModules.len()))
			if !chainCountersAgree(gotDetector.Stats.Passes[0], wantDetector.Stats.Passes[0]) ||
				seedDrift > chainDecisionDriftRate*float64(wantDetector.seedModules.len()) ||
				!finderPatternsAgree(got.fps[:got.total], want.fps[:want.total]) ||
				!finderPatternsNearlyAgree(got.weak, want.weak) {
				t.Fatalf("directional chain differs at %.0f degrees, print=%t\nchain stats: %#v\nhost stats:  %#v\nchain fps: %#v\nhost fps:  %#v\nchain weak: %#v\nhost weak:  %#v",
					degrees, printLevels,
					gotDetector.Stats.Passes[0], wantDetector.Stats.Passes[0],
					got.fps[:got.total], want.fps[:want.total], got.weak, want.weak)
			}
		}
	}
}
