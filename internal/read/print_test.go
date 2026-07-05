//go:build jabharness

package read

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/srlehn/jabcode/internal/encode"
)

// inkDensity holds the optical density each colorant lays into the R, G and B
// capture channels (rows R,G,B; columns C,M,Y,K). Reflectance composes as
// paper * 10^-density (Beer-Bouguer), so densities of overlaid inks add. The
// off-diagonal structure follows the measured interference: the red channel
// is nearly clean, green picks up cyan, blue picks up magenta.
var inkDensity = [3][4]float64{
	{1.40, 0.08, 0.01, 1.45},
	{0.35, 1.30, 0.06, 1.45},
	{0.10, 0.35, 1.10, 1.40},
}

// halftone screen angles per colorant plane (C, M, Y, K), the classic
// clustered-dot rosette arrangement, in radians.
var screenAngles = [4]float64{15 * math.Pi / 180, 75 * math.Pi / 180, 0, 45 * math.Pi / 180}

// printProcess is a forward model of colour printing followed by capture-side
// paper reflectance: ink separation with full grey-component replacement, a
// driver tone ramp (real drivers never lay exact 0% or 100%, which is what
// makes halftone structure appear on saturated colours at all), dot gain,
// per-colorant halftoning (clustered-dot AM at the classic screen angles, or
// FM error diffusion), laser tracking dots, per-plane smear and
// misregistration, tonal drift, and Beer-Bouguer recomposition over a paper
// tint. Every length knob is a module fraction so rows stay scale-adaptive.
type printProcess struct {
	modulePx    int           // set from the row's module size
	cells       float64       // AM halftone cells per module (ignored under fm)
	fm          bool          // Floyd-Steinberg error diffusion instead of AM
	dotGain     float64       // tone value increase at 50% coverage
	misregister [4][2]float64 // per-plane (dx, dy) in modules
	smear       float64       // per-plane blur radius in modules (ink spread)
	yellowDots  bool          // laser anticounterfeiting tracking dots
	drift       [4]float64    // per-plane density scale; 0 means nominal 1.0
	paper       [3]float64    // paper reflectance per channel; zero means default stock
}

// apply prints src onto a paper page with a two-module margin and returns the
// captured reflectance image.
func (p printProcess) apply(src image.Image, rng *rand.Rand) image.Image {
	margin := 2 * p.modulePx
	sb := src.Bounds()
	w, h := sb.Dx()+2*margin, sb.Dy()+2*margin
	page := image.NewNRGBA(image.Rect(0, 0, w, h))
	draw.Draw(page, page.Bounds(), image.White, image.Point{}, draw.Src)
	draw.Draw(page, image.Rect(margin, margin, margin+sb.Dx(), margin+sb.Dy()), src, sb.Min, draw.Src)

	// Ink separation with full grey-component replacement: the achromatic
	// component prints as K, as office devices render black regions.
	var planes [4][]float64
	for i := range planes {
		planes[i] = make([]float64, w*h)
	}
	for i := range w * h {
		r := float64(page.Pix[i*4+0]) / 255
		g := float64(page.Pix[i*4+1]) / 255
		b := float64(page.Pix[i*4+2]) / 255
		c, m, y := 1-r, 1-g, 1-b
		k := min(c, m, y)
		planes[0][i], planes[1][i], planes[2][i], planes[3][i] = c-k, m-k, y-k, k
	}

	pitch := float64(p.modulePx) / max(p.cells, 1)
	for i, plane := range planes {
		for j, a := range plane {
			// Driver tone ramp, then dot gain (strongest at midtone).
			a = 0.02 + 0.93*a
			a += p.dotGain * 4 * a * (1 - a)
			plane[j] = min(max(a, 0), 1)
		}
		if p.fm {
			fsDither(plane, w, h)
		} else {
			amScreen(plane, w, h, pitch, screenAngles[i])
		}
	}

	if p.yellowDots {
		trackingDots(planes[2], w, h, p.modulePx, rng)
	}

	if p.smear > 0 {
		r := max(int(p.smear*float64(p.modulePx)+0.5), 1)
		for i := range planes {
			planeBoxBlur(planes[i], w, h, r)
		}
	}

	var offs [4][2]float64
	drift := p.drift
	for i := range 4 {
		offs[i][0] = p.misregister[i][0] * float64(p.modulePx)
		offs[i][1] = p.misregister[i][1] * float64(p.modulePx)
		if drift[i] == 0 {
			drift[i] = 1
		}
	}
	paper := p.paper
	if paper == ([3]float64{}) {
		paper = [3]float64{0.97, 0.955, 0.92} // warm office stock
	}

	out := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			var dens [3]float64
			for i := range 4 {
				a := bilinearPlane(planes[i], w, h, float64(x)-offs[i][0], float64(y)-offs[i][1])
				if a <= 0 {
					continue
				}
				for k := range 3 {
					dens[k] += inkDensity[k][i] * drift[i] * a
				}
			}
			o := (y*w + x) * 4
			for k := range 3 {
				out.Pix[o+k] = clampByte(255 * paper[k] * math.Pow(10, -dens[k]))
			}
			out.Pix[o+3] = 255
		}
	}
	return out
}

// amScreen thresholds a coverage plane against a clustered-dot spot function
// on a screen grid of the given pixel pitch rotated by angle: dots grow from
// the cell centres, solids keep pinholes at the cell corners.
func amScreen(a []float64, w, h int, pitch, angle float64) {
	sin, cos := math.Sincos(angle)
	for y := range h {
		for x := range w {
			u := (float64(x)*cos + float64(y)*sin) / pitch
			v := (-float64(x)*sin + float64(y)*cos) / pitch
			s := (math.Cos(2*math.Pi*u)+math.Cos(2*math.Pi*v))/4 + 0.5
			i := y*w + x
			if a[i] > s {
				a[i] = 1
			} else {
				a[i] = 0
			}
		}
	}
}

// fsDither binarizes a coverage plane by Floyd-Steinberg error diffusion (a
// stochastic FM screen).
func fsDither(a []float64, w, h int) {
	for y := range h {
		for x := range w {
			i := y*w + x
			old := a[i]
			var v float64
			if old >= 0.5 {
				v = 1
			}
			a[i] = v
			e := old - v
			if x+1 < w {
				a[i+1] += e * 7 / 16
			}
			if y+1 < h {
				if x > 0 {
					a[i+w-1] += e * 3 / 16
				}
				a[i+w] += e * 5 / 16
				if x+1 < w {
					a[i+w+1] += e * 1 / 16
				}
			}
		}
	}
}

// trackingDots stamps the sparse yellow anticounterfeiting dot grid that
// office colour lasers print across the whole page, jittered per dot.
func trackingDots(yPlane []float64, w, h, modulePx int, rng *rand.Rand) {
	pitch := 4 * modulePx
	r := max(modulePx/8, 1)
	for gy := 0; gy < h; gy += pitch {
		for gx := 0; gx < w; gx += pitch {
			cx := gx + rng.Intn(pitch)
			cy := gy + rng.Intn(pitch)
			for dy := -r; dy <= r; dy++ {
				for dx := -r; dx <= r; dx++ {
					x, y := cx+dx, cy+dy
					if x >= 0 && x < w && y >= 0 && y < h && dx*dx+dy*dy <= r*r {
						yPlane[y*w+x] = 1
					}
				}
			}
		}
	}
}

// planeBoxBlur applies a separable box blur of the given radius to a coverage
// plane, modelling ink dispersion into neighbouring area.
func planeBoxBlur(a []float64, w, h, r int) {
	win := float64(2*r + 1)
	tmp := make([]float64, len(a))
	for y := range h {
		var sum float64
		for k := -r; k <= r; k++ {
			sum += a[y*w+clampIdx(k, w)]
		}
		tmp[y*w] = sum / win
		for x := 1; x < w; x++ {
			sum += a[y*w+clampIdx(x+r, w)] - a[y*w+clampIdx(x-1-r, w)]
			tmp[y*w+x] = sum / win
		}
	}
	for x := range w {
		var sum float64
		for k := -r; k <= r; k++ {
			sum += tmp[clampIdx(k, h)*w+x]
		}
		a[x] = sum / win
		for y := 1; y < h; y++ {
			sum += tmp[clampIdx(y+r, h)*w+x] - tmp[clampIdx(y-1-r, h)*w+x]
			a[y*w+x] = sum / win
		}
	}
}

// bilinearPlane samples a coverage plane at a fractional position with edge
// clamping - the compositor's view of a misregistered colorant plane.
func bilinearPlane(a []float64, w, h int, x, y float64) float64 {
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	fx := x - float64(x0)
	fy := y - float64(y0)
	get := func(xi, yi int) float64 { return a[clampIdx(yi, h)*w+clampIdx(xi, w)] }
	top := get(x0, y0)*(1-fx) + get(x0+1, y0)*fx
	bot := get(x0, y0+1)*(1-fx) + get(x0+1, y0+1)*fx
	return top*(1-fy) + bot*fy
}

// printedPalette runs the palette colours through the same print process and
// returns the palette as it lands on paper: the module error rate must be
// measured against the print-shifted reference the decoder actually
// classifies with (via the embedded palette), not the ideal RGB corners,
// which sit outside the print gamut for every row. Tracking dots are
// disabled for the reference swatches.
func printedPalette(p printProcess, pal []byte) []byte {
	p.yellowDots = false
	n := len(pal) / 3
	sw := 8 * p.modulePx
	swatches := image.NewNRGBA(image.Rect(0, 0, n*sw, sw))
	for i := range n {
		c := color.NRGBA{R: pal[i*3], G: pal[i*3+1], B: pal[i*3+2], A: 255}
		draw.Draw(swatches, image.Rect(i*sw, 0, (i+1)*sw, sw), image.NewUniform(c), image.Point{}, draw.Src)
	}
	printed := p.apply(swatches, rand.New(rand.NewSource(1))).(*image.NRGBA)
	margin := 2 * p.modulePx
	out := make([]byte, len(pal))
	for i := range n {
		var sum [3]float64
		count := 0
		for y := margin + sw/4; y < margin+3*sw/4; y++ {
			for x := margin + i*sw + sw/4; x < margin+i*sw+3*sw/4; x++ {
				o := printed.PixOffset(x, y)
				for c := range 3 {
					sum[c] += float64(printed.Pix[o+c])
				}
				count++
			}
		}
		for c := range 3 {
			out[i*3+c] = clampByte(sum[c] / float64(count))
		}
	}
	return out
}

// TestPrintHarness runs the synthetic print-channel rows: each mechanism of
// the print physics in isolation, then printer-style combinations with a
// camera capture stack on top, reporting the stage reached and the module
// error rate like TestHarness (scored against the printed palette). These
// rows are mechanism probes for decoder work, not a print-robustness claim -
// that closes only on real prints. Set PRINTDUMP to a directory to save each
// row's degraded image for jabdiag.
//
//	go test -tags jabharness -run TestPrintHarness -v .
func TestPrintHarness(t *testing.T) {
	payload := []byte("PRINT harness: eight colours on office paper 0123456789")
	misreg := [4][2]float64{{0.20, 0.10}, {-0.15, 0.20}, {0.10, -0.20}, {-0.05, 0.05}}
	misregHeavy := [4][2]float64{{0.40, 0.20}, {-0.30, 0.40}, {0.20, -0.40}, {-0.10, 0.10}}
	laserDrift := [4]float64{0.75, 1.15, 0.80, 0.95}
	rows := []struct {
		name   string
		colors int
		px     int
		p      printProcess
		camera bool
	}{
		{"am-fine", 8, 12, printProcess{cells: 4}, false},
		{"am-coarse", 8, 12, printProcess{cells: 2}, false},
		{"fm", 8, 12, printProcess{fm: true}, false},
		{"dotgain", 8, 12, printProcess{cells: 4, dotGain: 0.20}, false},
		{"misreg", 8, 12, printProcess{cells: 4, misregister: misreg}, false},
		{"misreg-heavy", 8, 12, printProcess{cells: 4, misregister: misregHeavy}, false},
		{"smear", 8, 12, printProcess{cells: 4, smear: 0.25}, false},
		{"yellowdots", 8, 12, printProcess{cells: 4, yellowDots: true}, false},
		{"drift", 8, 12, printProcess{cells: 4, drift: laserDrift}, false},
		{"laser", 8, 12, printProcess{cells: 3, dotGain: 0.15, misregister: misreg,
			smear: 0.15, yellowDots: true, drift: laserDrift}, true},
		{"inkjet", 8, 12, printProcess{fm: true, dotGain: 0.10, smear: 0.30}, true},
		{"laser-6px", 8, 6, printProcess{cells: 3, dotGain: 0.15, misregister: misreg,
			smear: 0.15, yellowDots: true, drift: laserDrift}, true},
		{"inkjet-6px", 8, 6, printProcess{fm: true, dotGain: 0.10, smear: 0.30}, true},
		{"laser-4c", 4, 12, printProcess{cells: 3, dotGain: 0.15, misregister: misreg,
			smear: 0.15, yellowDots: true, drift: laserDrift}, true},
		{"am-fine-4c", 4, 12, printProcess{cells: 4}, false},
	}
	const seed = 1

	var report bytes.Buffer
	fmt.Fprintf(&report, "%-14s %3s %3s  %-12s %-9s %s\n", "print", "col", "px", "stage", "moduleBER", "berHD")
	for _, row := range rows {
		r, err := encode.Render(encode.Config{
			Colors: row.colors, ModuleSize: row.px, SymbolNumber: 1,
		}, payload)
		if err != nil {
			t.Fatalf("encode %s: %v", row.name, err)
		}
		gt := groundTruth{img: r.Image, Data: payload, matrix: r.Matrix, side: r.SideSize, Palette: r.Palette}
		rng := rand.New(rand.NewSource(seed))
		row.p.modulePx = row.px
		img := row.p.apply(gt.img, rng)
		if row.camera {
			img = boxBlurDeg(img, 1, rng)
			img = gaussianNoise(img, 5, rng)
			img = jpegRecompress(img, 80, rng)
		}
		if dir := os.Getenv("PRINTDUMP"); dir != "" {
			if f, err := os.Create(filepath.Join(dir, "print_"+row.name+".png")); err == nil {
				png.Encode(f, img)
				f.Close()
			}
		}
		gt.Palette = printedPalette(row.p, gt.Palette)
		res := runPipeline(img, gt)
		ber, berHD := "-", "-"
		if res.berValid {
			ber = fmt.Sprintf("%.3f", res.ber)
			berHD = fmt.Sprintf("%.3f", res.berHD)
		}
		fmt.Fprintf(&report, "%-14s %3d %3d  %-12s %-9s %s\n", row.name, row.colors, row.px, res.stage, ber, berHD)
	}
	t.Logf("print harness results:\n%s", report.String())
}
