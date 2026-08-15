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
			// The reference arm needs the hits themselves, which only the
			// scan-only sweep returns: with the chain engaged the device keeps
			// them and hands back the candidates it kept plus their counters.
			resident.binarizer.scanOnly = true
			rawSweep, err := resident.binarizer.scanDirectionHits(
				width, height, direction, 3, currentFamilySeekChannel,
			)
			resident.binarizer.scanOnly = false
			if err != nil {
				t.Fatalf("scan directional raw at %.0f degrees: %v", degrees, err)
			}
			if len(rawSweep.hits) == 0 {
				t.Fatalf("directional raw scan at %.0f degrees produced no hits", degrees)
			}
			if rawSweep.summarized {
				t.Fatal("scan-only sweep reported a device summary")
			}
			// The seed histogram is shared and accumulates across every scan of
			// a locate, so it is taken and discarded here - the fetch clears it
			// - leaving the chained sweep below as the only contributor the
			// comparison sees. Without this the row chain's own share would
			// arrive on the device arm and on neither host one.
			resident.MaterializeSeedHistogram()
			// The single-direction chained sweep is what this compares. The
			// exported entry point routes the current family through the
			// resident batch instead, which deliberately keeps its candidates
			// and summary on the device for the assembly stage, so asking it
			// here would compare a host arm against nothing.
			sweep, err := resident.binarizer.scanDirectionHits(
				width, height, direction, 3, currentFamilySeekChannel,
			)
			hits := sweep.hits
			if err != nil {
				t.Fatalf("scan directional chain at %.0f degrees: %v", degrees, err)
			}
			if len(hits) == 0 {
				t.Fatalf("directional chain at %.0f degrees produced no candidates", degrees)
			}
			if !hits[0].chained || !sweep.summarized {
				t.Fatal("directional scan returned without a device-summarized chain")
			}
			if sweep.summary.rawHits != len(rawSweep.hits) {
				t.Fatalf("summary counted %d raw hits, the sweep produced %d",
					sweep.summary.rawHits, len(rawSweep.hits))
			}
			if len(hits) >= sweep.summary.rawHits {
				t.Fatalf("compaction kept %d of %d hits, which is no reduction",
					len(hits), sweep.summary.rawHits)
			}
			if !printLevels && degrees == 15 {
				if got := retained.Load(); got != gpuFinderDirectionalRetainedBytes {
					t.Fatalf("directional buffers charged %d bytes, want %d", got, gpuFinderDirectionalRetainedBytes)
				}
			}
			sortDirHits := func(list []finderDirHit) {
				slices.SortFunc(list, func(a, b finderDirHit) int {
					if order := cmp.Compare(a.centre.Y, b.centre.Y); order != 0 {
						return order
					}
					if order := cmp.Compare(a.centre.X, b.centre.X); order != 0 {
						return order
					}
					return cmp.Compare(a.module, b.module)
				})
			}
			sortDirHits(hits)
			sortDirHits(rawSweep.hits)

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
					d.applyDirectionalSummary(sweep.summary)
					// The module-size distribution no longer rides the summary;
					// it reaches the host the way production reads it, from the
					// shared histogram at the one decision that wants it.
					d.seedHistogram = resident.MaterializeSeedHistogram
					d.mergeDeviceSeedModules()
					d.processDirectionalFamilyHits(direction, hits, &state)
				} else {
					for _, hit := range rawSweep.hits {
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
