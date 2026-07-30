package detect

import (
	"image"
	"image/draw"
	"testing"

	"github.com/srlehn/jabcode/internal/core"
	"github.com/srlehn/jabcode/internal/encode"
)

// This diagnostic pins the first stage that breaks half-turn equivariance.
// Any change here must be judged against the capture tables because symmetric
// threshold grids have already changed correct side-version selection.
func TestLocalThresholdGridHalfTurnDivergenceIsPinned(t *testing.T) {
	src := finderHalfTurnFixture(t)
	rotated := rotateNRGBAHalfTurn(src)
	srcBM, srcCh := prepareHalfTurnFixture(src)
	rotatedBM, rotatedCh := prepareHalfTurnFixture(rotated)

	if got := bitmapHalfTurnMismatches(srcBM, rotatedBM); got != 0 {
		t.Fatalf("balanced source differs at %d bytes", got)
	}
	want := [3]int{1416, 1642, 1640}
	for channel := range 3 {
		if got := bitmapHalfTurnMismatches(srcCh[channel], rotatedCh[channel]); got != want[channel] {
			t.Errorf("channel %d differs at %d pixels, want %d", channel, got, want[channel])
		}
	}
}

func TestFinderModuleSizeHalfTurnCorrespondence(t *testing.T) {
	src := finderHalfTurnFixture(t)
	rotated := rotateNRGBAHalfTurn(src)
	srcFPs := locateHalfTurnFixture(t, src)
	rotatedFPs := locateHalfTurnFixture(t, rotated)

	w, h := float64(src.Bounds().Dx()), float64(src.Bounds().Dy())
	for typ := range 4 {
		rotatedFPs[typ].Center.X = w - rotatedFPs[typ].Center.X
		rotatedFPs[typ].Center.Y = h - rotatedFPs[typ].Center.Y
		if srcFPs[typ].Typ != typ || rotatedFPs[typ].Typ != typ {
			t.Fatalf("type %d correspondence changed: source=%d rotated=%d",
				typ, srcFPs[typ].Typ, rotatedFPs[typ].Typ)
		}
	}

	wantSource := [4]float64{
		5.407407407407407,
		5.703703703703703,
		5.777777777777778,
		5.666666666666667,
	}
	wantRotated := [4]float64{
		5.916666666666667,
		5.555555555555554,
		5.722222222222222,
		5.777777777777778,
	}
	for typ := range 4 {
		if absFloat(srcFPs[typ].ModuleSize-wantSource[typ]) > 1e-12 ||
			absFloat(rotatedFPs[typ].ModuleSize-wantRotated[typ]) > 1e-12 {
			t.Errorf("type %d module correspondence changed: source=%.15f rotated=%.15f",
				typ, srcFPs[typ].ModuleSize, rotatedFPs[typ].ModuleSize)
		}
	}
}

// The normalized one-pixel centre gap is the separate middle-run precharge
// defect. Correcting it changes version-table rows in both directions, so this
// test keeps the boundary visible until that behavior change is handled whole.
func TestVerticalCrossCheckHalfTurnDivergenceIsPinned(t *testing.T) {
	src := finderHalfTurnFixture(t)
	rotated := rotateNRGBAHalfTurn(src)
	_, srcCh := prepareHalfTurnFixture(src)
	_, rotatedCh := prepareHalfTurnFixture(rotated)

	srcY, rotatedY := 658.0, 19.0
	var srcModule, rotatedModule float64
	if !crossCheckPatternVertical(srcCh[0], 12, 657, &srcY, &srcModule, 3) {
		t.Fatal("source vertical cross-check failed")
	}
	if !crossCheckPatternVertical(rotatedCh[0], 12, 21, &rotatedY, &rotatedModule, 3) {
		t.Fatal("rotated vertical cross-check failed")
	}
	if srcModule != 6 || rotatedModule != 6 || srcY != 657 || rotatedY != 22 ||
		srcY-(float64(src.Bounds().Dy())-rotatedY) != 1 {
		t.Fatalf("vertical cross-check boundary changed: source=(%.3f,%.3f) rotated=(%.3f,%.3f)",
			srcY, srcModule, rotatedY, rotatedModule)
	}
}

func TestAxisAndDirectionalZeroAcceptSameBlurredFP1(t *testing.T) {
	src := finderHalfTurnFixture(t)
	bm, ch := prepareHalfTurnFixture(src)
	axis := locateHalfTurnFixture(t, src)[fp1]

	d := &PrimaryDetector{BM: bm, Ch: ch, Mode: IntensiveDetect}
	d.Stats.Passes = append(d.Stats.Passes, FinderPassStats{})
	state := newPrimaryFamilyScan()
	d.scanDirectionalFamily(newScanDirection(0), 1, &state)
	for i := range state.total {
		got := state.fps[i]
		if got.Typ != fp1 {
			continue
		}
		if max(absFloat(got.Center.X-axis.Center.X), absFloat(got.Center.Y-axis.Center.Y)) <= axis.ModuleSize {
			return
		}
	}
	t.Fatalf("directional zero scan did not accept axis FP1 %+v", axis)
}

func finderHalfTurnFixture(t *testing.T) *image.NRGBA {
	t.Helper()
	r, err := encode.Render(encode.Config{
		Colors: 8, ModuleSize: 3, SymbolNumber: 1,
		SymbolVersions: []image.Point{{X: 24, Y: 24}},
	}, []byte("version detection gate: large and rectangular symbols under mild degradation 0123456789"))
	if err != nil {
		t.Fatal(err)
	}
	return UpscaleNRGBA(boxBlurRadiusOne(r.Image), 2)
}

func boxBlurRadiusOne(src image.Image) *image.NRGBA {
	b := src.Bounds()
	in := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(in, in.Bounds(), src, b.Min, draw.Src)
	tmp := image.NewNRGBA(in.Bounds())
	out := image.NewNRGBA(in.Bounds())
	for y := range b.Dy() {
		for x := range b.Dx() {
			for c := range 3 {
				sum := 0
				for k := -1; k <= 1; k++ {
					sx := min(max(x+k, 0), b.Dx()-1)
					sum += int(in.Pix[in.PixOffset(sx, y)+c])
				}
				tmp.Pix[tmp.PixOffset(x, y)+c] = byte((sum + 1) / 3)
			}
			tmp.Pix[tmp.PixOffset(x, y)+3] = 255
		}
	}
	for y := range b.Dy() {
		for x := range b.Dx() {
			for c := range 3 {
				sum := 0
				for k := -1; k <= 1; k++ {
					sy := min(max(y+k, 0), b.Dy()-1)
					sum += int(tmp.Pix[tmp.PixOffset(x, sy)+c])
				}
				out.Pix[out.PixOffset(x, y)+c] = byte((sum + 1) / 3)
			}
			out.Pix[out.PixOffset(x, y)+3] = 255
		}
	}
	return out
}

func rotateNRGBAHalfTurn(src *image.NRGBA) *image.NRGBA {
	b := src.Bounds()
	dst := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := range b.Dy() {
		for x := range b.Dx() {
			s := src.PixOffset(b.Min.X+x, b.Min.Y+y)
			d := dst.PixOffset(b.Dx()-1-x, b.Dy()-1-y)
			copy(dst.Pix[d:d+4], src.Pix[s:s+4])
		}
	}
	return dst
}

func prepareHalfTurnFixture(img image.Image) (*core.Bitmap, [3]*core.Bitmap) {
	bm := core.BitmapFromImage(img)
	BalanceRGB(bm)
	return bm, BinarizerRGB(bm, nil)
}

func locateHalfTurnFixture(t *testing.T, img image.Image) [4]FinderPattern {
	t.Helper()
	bm, ch := prepareHalfTurnFixture(img)
	d := &PrimaryDetector{BM: bm, Ch: ch, Mode: IntensiveDetect}
	if status := d.findPrimarySymbol(); status != core.Success {
		t.Fatalf("finder status %d", status)
	}
	var out [4]FinderPattern
	copy(out[:], d.FPs[:4])
	return out
}

func bitmapHalfTurnMismatches(src, rotated *core.Bitmap) int {
	if src.Width != rotated.Width || src.Height != rotated.Height || src.Channels != rotated.Channels {
		return -1
	}
	mismatches := 0
	for y := range src.Height {
		for x := range src.Width {
			for channel := range src.Channels {
				s := (y*src.Width+x)*src.Channels + channel
				r := ((src.Height-1-y)*src.Width+(src.Width-1-x))*src.Channels + channel
				if src.Pix[s] != rotated.Pix[r] {
					mismatches++
				}
			}
		}
	}
	return mismatches
}

func absFloat(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}
