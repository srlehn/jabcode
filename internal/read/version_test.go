//go:build jabharness

package read

import (
	"bytes"
	"fmt"
	"image"
	"math/rand"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/detect"
	"github.com/srlehn/jabcode/internal/encode"
)

// TestVersionDetection measures side-version detection on large and
// rectangular symbols under mild degradation: the literature reports the
// reference decoder reading under 10% of such codes because the side version
// is finder distance divided by the finder-estimated module size, whose error
// scales with the module count between finders. Each row reports whether the
// finder chain survived, whether the computed side size matches the encoder's
// ground truth, and whether the full decode read the payload.
//
//	go test -tags jabharness -run TestVersionDetection -v .
func TestVersionDetection(t *testing.T) {
	payload := []byte("version detection gate: large and rectangular symbols under mild degradation 0123456789")
	versions := []image.Point{
		{X: 8, Y: 8}, {X: 16, Y: 16}, {X: 24, Y: 24}, {X: 32, Y: 32},
		{X: 32, Y: 8}, {X: 8, Y: 32}, {X: 24, Y: 12},
	}
	moduleSizes := []int{12, 6, 4, 3}
	degradations := []struct {
		name  string
		level float64
		apply func(image.Image, float64, *rand.Rand) image.Image
	}{
		{"identity", 0, func(s image.Image, _ float64, _ *rand.Rand) image.Image { return s }},
		{"blur-r", 1, boxBlurDeg},
		{"blur-r", 2, boxBlurDeg},
		{"noise-sd", 10, gaussianNoise},
		{"jpeg-q", 60, jpegRecompress},
	}
	const seed = 1

	var report bytes.Buffer
	fmt.Fprintf(&report, "%-8s %4s %-12s %-10s %-14s %s\n", "version", "px", "degradation", "finders", "side", "decode")
	sideWrong, sideTotal := 0, 0
	for _, v := range versions {
		want := image.Pt(17+v.X*4, 17+v.Y*4)
		for _, msz := range moduleSizes {
			r, err := encode.Render(encode.Config{
				Colors: 8, ModuleSize: msz, SymbolNumber: 1,
				SymbolVersions: []image.Point{v},
			}, payload)
			if err != nil {
				t.Fatalf("encode %dx%d: %v", v.X, v.Y, err)
			}
			for _, d := range degradations {
				rng := rand.New(rand.NewSource(seed))
				img := d.apply(r.Image, d.level, rng)

				bm := core.BitmapFromImage(img)
				detect.BalanceRGB(bm)
				ch := detect.BinarizerRGB(bm, nil)
				det := &detect.PrimaryDetector{BM: bm, Ch: ch, Mode: detect.IntensiveDetect}

				finders, side := "no", "-"
				if det.LocateFinders() {
					finders = "yes"
					ss := detect.CalculateSideSize(bm, det.FPs)
					if ss.X == -1 || ss.Y == -1 {
						if quad, ok := det.SelectFinderQuadByGeometry(); ok {
							fps := det.FPs
							copy(fps, quad[:])
							ss = detect.CalculateSideSize(bm, fps)
						}
					}
					sideTotal++
					if ss == want {
						side = "ok"
					} else {
						side = fmt.Sprintf("WRONG %dx%d", ss.X, ss.Y)
						sideWrong++
					}
				}
				decode := "failed"
				if out, err := Decode(img); err == nil && bytes.Equal(out, payload) {
					decode = "ok"
				} else if err == nil {
					decode = "CORRUPT"
				}
				fmt.Fprintf(&report, "%-8s %4d %-12s %-10s %-14s %s\n",
					fmt.Sprintf("%dx%d", v.X, v.Y), msz,
					fmt.Sprintf("%s %g", d.name, d.level), finders, side, decode)
			}
		}
	}
	t.Logf("version detection results (side wrong on %d of %d finder-successful rows):\n%s",
		sideWrong, sideTotal, report.String())
}
