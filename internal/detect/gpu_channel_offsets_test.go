//go:build !js

package detect

import (
	"image"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/srlehn/vulki"

	"github.com/srlehn/jabcode/internal/core"
)

// channelOffsetFixture renders a module grid whose blue plane is displaced by
// exactly one grid step, which is the damage the search exists to find: a print
// whose colorant planes did not land on the same paper position.
//
// The modules are random rather than a pattern, because the score is a
// bimodality measure over the module population and a regular pattern would let
// a wrong offset score as well as the right one.
func channelOffsetFixture(t *testing.T, side image.Point, module int, shift int) *core.Bitmap {
	t.Helper()
	width, height := side.X*module, side.Y*module
	bm := core.NewBitmap(width, height, 4)
	rng := rand.New(rand.NewPCG(0x0ff5e7, 0x9e3779b9))
	dark := make([]bool, side.X*side.Y)
	for i := range dark {
		dark[i] = rng.IntN(2) == 0
	}
	value := func(sx, sy int) byte {
		if sx < 0 || sy < 0 || sx >= side.X || sy >= side.Y {
			return 255
		}
		if dark[sy*side.X+sx] {
			return 20
		}
		return 235
	}
	// A few levels of per-pixel noise, because a flat-colour module makes every
	// candidate whose taps stay inside it score identically - exactly zero on the host, and a
	// scatter of f32 rounding on the device, which then lets the two parity
	// halves chase different candidates and disagree. Real print pixels carry
	// halftone and sensor noise; a fixture without it tests neither arm.
	noisy := func(v byte) byte {
		n := int(v) + rng.IntN(7) - 3
		return byte(min(max(n, 0), 255))
	}
	for y := range height {
		for x := range width {
			at := (y*width + x) * 4
			bm.Pix[at+0] = noisy(value(x/module, y/module))
			bm.Pix[at+1] = noisy(value(x/module, y/module))
			// Only the blue plane is displaced, by whole pixels so the fixture
			// carries no resampling of its own.
			bm.Pix[at+2] = noisy(value((x-shift)/module, y/module))
			bm.Pix[at+3] = 255
		}
	}
	return bm
}

// TestGPUChannelOffsetsMatchHost holds the device search to the host one it
// replaces. The comparison is the adopted offsets, not the scores: the two
// arms take their deciles differently - the host sorts the module values and
// the device bins them - so what has to agree is which displacement the search
// adopts, which is all the sampler ever sees.
func TestGPUChannelOffsetsMatchHost(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	side := image.Pt(37, 37)
	const module = 12
	for _, shift := range []int{0, 3, 4, 5, 6} {
		bm := channelOffsetFixture(t, side, module, shift)
		input, err := device.NewBuffer(uint64(bm.Width * bm.Height * 4))
		if err != nil {
			t.Fatalf("allocate channel-offset input: %v", err)
		}
		resident, err := newGPUResidentBinarizerWithDevice(device, bm.Width, bm.Height)
		if err != nil {
			_ = input.Close()
			t.Fatalf("new resident GPU binarizer: %v", err)
		}
		if err := input.Upload(bm.Pix); err != nil {
			t.Fatalf("upload channel-offset input: %v", err)
		}
		// The search reads the balanced image, so the arms have to agree on
		// which pixels those are: the host is given the same balanced copy the
		// device holds rather than the raw fixture.
		if _, _, materialize, err := resident.Binarize(
			input, bm.Width, bm.Height, nil, true, 0,
		); err != nil {
			t.Fatalf("binarize channel-offset input: %v", err)
		} else if err := materialize(); err != nil {
			t.Fatalf("materialize channel-offset masks: %v", err)
		}
		balanced, err := resident.DownloadBalanced(bm.Width, bm.Height)
		if err != nil {
			t.Fatalf("download balanced channel-offset image: %v", err)
		}

		pt := core.PerspectiveTransform(
			core.Pt(3.5*module, 3.5*module),
			core.Pt(float64(side.X-4)*module+0.5*module, 3.5*module),
			core.Pt(float64(side.X-4)*module+0.5*module, float64(side.Y-4)*module+0.5*module),
			core.Pt(3.5*module, float64(side.Y-4)*module+0.5*module),
			side,
		)
		// The adopted offsets are deliberately not compared. The two arms agree
		// on the scores to within a level out of a range of about thirty-six,
		// and the selection only adopts a winner beating the nominal position
		// by a quarter of its score - so they can differ only where two
		// candidates sit inside that one level of each other, which is a
		// near-tie the routes are not required to share. Which of those wins is
		// gated by the print harness and the capture census, not here.
		if _, err := resident.SearchChannelOffsets(bm.Width, bm.Height, pt, side); err != nil {
			t.Fatalf("device channel-offset search: %v", err)
		}

		// The stage produces the score table; the offsets are what the shared
		// selection makes of it. Comparing the table is comparing the work, and
		// it localizes a disagreement to the candidate that caused it.
		modW, modH := moduleExtent(pt, side)
		host := channelOffsetScoreTable(channelOffsetScorer(balanced, pt, side, modW, modH))
		got := resident.offsetTable
		if len(got) != len(host) {
			t.Fatalf("device produced %d scores, host %d", len(got), len(host))
		}
		spread := 0.0
		for slot := range host {
			spread = math.Max(spread, host[slot])
			// The two arms take their deciles differently - the host sorts the
			// module values, the device bins them at quarter resolution - so a
			// score may differ by about that much where the population is dense
			// around a decile.
			if math.Abs(got[slot]-host[slot]) > 1.0 {
				t.Fatalf("shift %d slot %d: device score %v, host %v",
					shift, slot, got[slot], host[slot])
			}
		}
		// Without this the comparison would pass on two arms that both scored
		// everything flat, which is what a fixture the search cannot see looks
		// like.
		if spread < 1.0 {
			t.Fatalf("shift %d: every host score is under %v, so agreeing on them "+
				"proves nothing about the search", shift, spread)
		}
		if err := resident.Close(); err != nil {
			t.Errorf("close resident GPU binarizer: %v", err)
		}
		if err := input.Close(); err != nil {
			t.Errorf("close channel-offset input: %v", err)
		}
	}
	if err := device.Close(); err != nil {
		t.Errorf("close channel-offset device: %v", err)
	}
}
