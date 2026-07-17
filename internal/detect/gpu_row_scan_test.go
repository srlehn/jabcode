package detect

import (
	"fmt"
	"math"
	"math/bits"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
)

// layerOKReference mirrors the WGSL layer_ok integer evaluation of the
// float64 comparison |inside/3.0 - s| < inside/6.0, for the exhaustive
// equivalence proof below. Keep it in sync with finder_row_scan.wgsl.
func layerOKReference(inside, s int) bool {
	d := inside - 3*s
	if d < 0 {
		d = -d
	}
	d2 := 2 * d
	if d2 < inside {
		return true
	}
	if d2 > inside || 3*s < inside || inside%3 == 0 {
		return false
	}
	high := bits.Len32(uint32(inside)) - 1
	exponent := high - 2
	if inside >= 3<<(high-1) {
		exponent = high - 1
	}
	shift := 52 - exponent
	remainder := inside % 3
	if shift%2 == 1 {
		remainder = (remainder * 2) % 3
	}
	return remainder == 2
}

// TestGPUFinderRowScanLayerEquivalence exhaustively proves the kernel's
// integer layer comparison equals checkPatternCross's float64 form for every
// reachable pair - inside is bounded by three times the widest supported
// row. The exact boundary 2*(3s - inside) == inside follows float64's
// rounding of inside/3, which the integer form reproduces; everything else
// has an integer margin the float64 error cannot cross. Runs without a
// device: it validates the arithmetic contract, not the dispatch.
func TestGPUFinderRowScanLayerEquivalence(t *testing.T) {
	for inside := 3; inside <= 3*8192; inside++ {
		layer := float64(inside) / 3.0
		tol := layer / 2.0
		for s := 0; s <= inside/2+2; s++ {
			want := math.Abs(layer-float64(s)) < tol
			if got := layerOKReference(inside, s); got != want {
				t.Fatalf("layerOKReference(%d, %d) = %v, float64 form = %v", inside, s, got, want)
			}
		}
	}
}

// cpuRowScanHit is a raw seek hit produced by replaying the CPU row walk's
// driver loop, for exact comparison against the device scan's records.
type cpuRowScanHit struct {
	y      int
	seq    int
	center float64
	module float64
}

// cpuRowScanChannel replays the per-row seek driver every family scan runs
// (seekPatternHorizontal under the scanCurrentFamilyRow/scanBSIFamilyRow loop)
// over one binarized channel and returns every raw hit in walk order.
func cpuRowScanChannel(bin *core.Bitmap) []cpuRowScanHit {
	w, h := bin.Width, bin.Height
	var hits []cpuRowScanHit
	for y := 0; y < h; y++ {
		row := bin.Pix[y*w : (y+1)*w]
		startX, endX, skip := 0, w, 0
		seq := 0
		for first := true; first || (startX < w && endX < w); {
			first = false
			startX += skip
			endX = w
			ps := seekPatternHorizontal(row, startX, endX)
			startX, endX = ps.start, ps.end
			if !ps.ok {
				continue
			}
			hits = append(hits, cpuRowScanHit{y: y, seq: seq, center: ps.Center, module: ps.ModuleSize})
			seq++
			skip = ps.skip
		}
	}
	return hits
}

// stripeGPURowScanBitmap paints n-1-1-1-m finder-like run patterns into the
// green and red channels so the scan has guaranteed structured hits,
// including one pattern ending exactly at the row's last pixel.
func stripeGPURowScanBitmap(width, height int) *core.Bitmap {
	bm := core.NewBitmap(width, height, 4)
	for index := 0; index < width*height; index++ {
		bm.Pix[index*4+3] = 255
	}
	// fill writes one value per run, flipping between runs.
	fill := func(y, x, channel int, runs []int) {
		value := byte(0)
		for _, run := range runs {
			for i := 0; i < run && x < width; i++ {
				bm.Pix[(y*width+x)*4+channel] = value
				x++
			}
			if value == 0 {
				value = 0xff
			} else {
				value = 0
			}
		}
	}
	// Bands several rows tall survive the binary filter; single rows do not.
	for band := 4; band < height-8; band += 14 {
		for y := band; y < band+6 && y < height; y++ {
			fill(y, 3, 1, []int{5, 4, 4, 4, 6})
			fill(y, 40, 1, []int{4, 3, 3, 3, 4})
			fill(y, 3, 0, []int{6, 5, 5, 5, 7})
			if width > 30 {
				// A pattern whose trailing run touches the row end
				// exercises the j == max-1 edge of the machine.
				fill(y, width-18, 1, []int{4, 3, 3, 3, 5})
			}
		}
	}
	return bm
}

// TestGPUFinderRowScanParity pins the offload contract of the device row
// scan: for every scanned channel the record set is bit-identical to the CPU
// row walk's raw seek hits - same rows, same in-row order, same float64
// centre and module size once derived from the integer records.
func TestGPUFinderRowScanParity(t *testing.T) {
	const maxWidth = 331
	const maxHeight = 257
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	t.Logf("Vulkan adapter: %s", device.Info().AdapterName)
	input, err := device.NewBuffer(maxWidth * maxHeight * 4)
	if err != nil {
		_ = device.Close()
		t.Fatalf("allocate GPU row scan input: %v", err)
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
			t.Errorf("close GPU row scan input: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Errorf("close GPU row scan device: %v", err)
		}
	})

	tests := []struct {
		name   string
		bitmap *core.Bitmap
	}{
		{name: "noise", bitmap: gpuTestBitmap(maxWidth, maxHeight)},
		{name: "stripes", bitmap: stripeGPURowScanBitmap(311, 128)},
		{name: "narrow", bitmap: stripeGPURowScanBitmap(31, 64)},
	}
	const scanChannels = (1 << 0) | (1 << 1)
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			bm := test.bitmap
			if err := input.Upload(bm.Pix); err != nil {
				t.Fatalf("upload GPU row scan input: %v", err)
			}
			channels, hits, materialize, err := resident.Binarize(
				input, bm.Width, bm.Height, nil, false, scanChannels,
			)
			if err != nil {
				t.Fatalf("binarize with device row scan: %v", err)
			}
			if err := materialize(); err != nil {
				t.Fatalf("materialize device row scan masks: %v", err)
			}
			if hits == nil || !hits.valid {
				t.Fatal("device row scan returned no valid hits")
			}
			for channel := range 2 {
				want := cpuRowScanChannel(channels[channel])
				got := hits.channels[channel]
				if len(got) != len(want) {
					t.Fatalf(
						"channel %d device scan returned %d hits, CPU walk %d",
						channel, len(got), len(want),
					)
				}
				for index, hit := range got {
					ref := want[index]
					if hit.y != ref.y || hit.seq != ref.seq ||
						math.Float64bits(hit.center()) != math.Float64bits(ref.center) ||
						math.Float64bits(hit.moduleSize()) != math.Float64bits(ref.module) {
						t.Fatalf(
							"channel %d hit %d = (y %d seq %d center %v module %v), want (y %d seq %d center %v module %v)",
							channel, index,
							hit.y, hit.seq, hit.center(), hit.moduleSize(),
							ref.y, ref.seq, ref.center, ref.module,
						)
					}
				}
				if testing.Verbose() {
					t.Log(fmt.Sprintf("channel %d: %d hits bit-identical", channel, len(got)))
				}
			}
		})
	}
}
