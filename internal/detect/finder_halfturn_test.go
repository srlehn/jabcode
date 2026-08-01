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
		6,
		6,
		6,
		6,
	}
	wantRotated := [4]float64{
		6.333333333333334,
		6,
		6,
		5.966666666666667,
	}
	for typ := range 4 {
		if absFloat(srcFPs[typ].ModuleSize-wantSource[typ]) > 1e-12 ||
			absFloat(rotatedFPs[typ].ModuleSize-wantRotated[typ]) > 1e-12 {
			t.Errorf("type %d module correspondence changed: source=%v rotated=%v",
				typ, srcFPs[typ].ModuleSize, rotatedFPs[typ].ModuleSize)
		}
	}
}

// Corresponding source and rotated seeds measure the same 6-pixel module from
// exactly reversed run windows, so their refined centres must land on the same
// physical point. They did not while the vertical walk charged the centre pixel
// to the run before the middle one: that left a whole pixel of normalized gap.
// The gap is now zero and this pins it there, since the walk is the only stage
// between the reversed windows and the centre.
func TestVerticalCrossCheckHalfTurnIsEquivariant(t *testing.T) {
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
	if srcModule != 6 || rotatedModule != 6 || srcY != 656.5 || rotatedY != 21.5 ||
		srcY-(float64(src.Bounds().Dy())-rotatedY) != 0 {
		t.Fatalf("vertical cross-check equivariance changed: source=(%v,%v) rotated=(%v,%v)",
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

func TestFinderModuleSizeExcludesBlurredDiagonalBias(t *testing.T) {
	src := finderHalfTurnFixture(t)
	_, ch := prepareHalfTurnFixture(src)
	fps := locateHalfTurnFixture(t, src)
	for typ, fp := range fps {
		if fp.ModuleSize != 6 {
			t.Errorf("fp%d module size = %v, want the 6-pixel symbol-axis scale", typ, fp.ModuleSize)
		}
	}

	fp := fps[fp0]
	base := newScanDirection(0)
	for _, probe := range []scanDirection{base, base.perpendicular()} {
		centre := fp.Center
		var module float64
		if !crossCheckPatternAlong(ch[1], probe, 12, &centre, &module, 3, nil) || module != 6 {
			t.Fatalf("symbol-axis module size = %v, want 6", module)
		}
	}
	centre := fp.Center
	var diagonalModule float64
	var window walkWindow
	diagonal := base.turn(45)
	if !crossCheckAlong(ch[1], diagonal, 12, diagPxPerRun(diagonal),
		&centre, &diagonalModule, 3, &window) {
		t.Fatal("blurred finder diagonal did not confirm")
	}
	if window.runs != [5]int{13, 6, 4, 6, 15} || absFloat(diagonalModule-16.0/3.0) > 1e-12 {
		t.Fatalf("diagonal runs = %v, module size = %v; want [13 6 4 6 15], 16/3",
			window.runs, diagonalModule)
	}
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
