//go:build jabharness

package jabcode

import (
	"bytes"
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	"math"
	"math/rand"
	"testing"
)

// TestHarness encodes known payloads, applies seeded degradations, runs each
// through the detector, and reports the stage reached (no-finders / no-sidesize /
// no-sample / sampled / decoded) plus, whenever sampling produced a module grid,
// the module / pre-LDPC error rate: each sampled module classified by nearest
// palette colour against the matrix the encoder rendered. That rate is invisible to
// a byte-level decode, where LDPC either hides the errors or fails outright.
// Build-tagged so the normal suite never runs it:
//
//	go test -tags jabharness -run TestHarness -v .
func TestHarness(t *testing.T) {
	payloads := [][]byte{
		[]byte("HARNESS round-trip 0123456789"),
		[]byte("The quick brown fox jumps over the lazy dog."),
	}
	degradations := []struct {
		name   string
		levels []float64
		apply  func(image.Image, float64, *rand.Rand) image.Image
	}{
		{"identity", []float64{0}, func(s image.Image, _ float64, _ *rand.Rand) image.Image { return s }},
		{"jpeg-q", []float64{90, 60, 30, 10}, jpegRecompress},
		{"blur-r", []float64{1, 2, 4, 8}, boxBlurDeg},
		{"colorcast", []float64{0.2, 0.5, 0.8}, colorCast},
		{"noise-sd", []float64{10, 25, 50}, gaussianNoise},
		{"lattice-p", []float64{3, 5, 8}, screenLattice},
	}
	const seed = 1

	var report bytes.Buffer
	fmt.Fprintf(&report, "%-14s %-26s %6s  %-12s %s\n", "degradation", "payload", "level", "stage", "moduleBER")
	for _, data := range payloads {
		gt := encodeGroundTruth(t, data)
		label := string(data)
		if len(label) > 24 {
			label = label[:24]
		}
		for _, d := range degradations {
			for _, lvl := range d.levels {
				rng := rand.New(rand.NewSource(seed))
				res := runPipeline(d.apply(gt.img, lvl, rng), gt)
				ber := "-"
				if res.berValid {
					ber = fmt.Sprintf("%.3f", res.ber)
				}
				fmt.Fprintf(&report, "%-14s %-26s %6s  %-12s %s\n", d.name, label, fmt.Sprintf("%g", lvl), res.stage, ber)
			}
		}
	}

	t.Logf("harness results:\n%s", report.String())
}

// groundTruth is an encoded code together with the per-module colour indices and
// palette the encoder actually rendered — the reference the harness scores against.
type groundTruth struct {
	img     image.Image
	data    []byte
	matrix  []byte // rendered module colour indices, row-major, side.X wide
	side    image.Point
	palette []byte // packed RGB triples
}

func encodeGroundTruth(t *testing.T, data []byte, opts ...Option) groundTruth {
	e := NewEncoder(opts...)
	img, err := e.Encode(data)
	if err != nil {
		t.Fatalf("encode %q: %v", data, err)
	}
	s := e.symbols[0]
	return groundTruth{img: img, data: data, matrix: s.matrix, side: s.sideSize, palette: e.palette}
}

// pipelineResult is the outcome of running one image through the detector: the
// furthest stage reached, and (when sampling succeeded and the grid matched the
// ground-truth side) the module / pre-LDPC error rate.
type pipelineResult struct {
	stage    string // no-finders | no-sidesize | no-sample | sampled | decoded
	berValid bool
	ber      float64
}

// runPipeline mirrors detectPrimary's finder-pattern path stage by stage so the
// failure can be attributed, then runs the full Decode to tell a clean decode from
// a sample that LDPC could not recover.
func runPipeline(img image.Image, gt groundTruth) pipelineResult {
	bm := bitmapFromImage(img)
	balanceRGB(bm)
	ch := binarizerRGB(bm, nil)
	d := &primaryDetector{bm: bm, ch: ch, mode: intensiveDetect}
	if !d.locateFinders() {
		return pipelineResult{stage: "no-finders"}
	}
	side := calculateSideSize(d.fps)
	if side.X == -1 || side.Y == -1 {
		return pipelineResult{stage: "no-sidesize"}
	}
	pt := getPerspectiveTransform(d.fps[0].center, d.fps[1].center, d.fps[2].center, d.fps[3].center, side)
	sampled := sampleSymbol(bm, pt, side)
	if sampled == nil {
		return pipelineResult{stage: "no-sample"}
	}
	res := pipelineResult{stage: "sampled"}
	res.ber, res.berValid = moduleBER(sampled, gt)
	if out, err := Decode(img); err == nil && bytes.Equal(out, gt.data) {
		res.stage = "decoded"
	}
	return res
}

// moduleBER classifies every sampled module by nearest palette colour and returns
// the fraction that differ from the rendered ground truth. It is valid only when
// the sampled grid matches the ground-truth side size (otherwise geometry, not
// classification, failed, and a per-module comparison is meaningless).
func moduleBER(sampled *bitmap, gt groundTruth) (float64, bool) {
	if sampled.width != gt.side.X || sampled.height != gt.side.Y {
		return 0, false
	}
	bpp := sampled.channels
	n := gt.side.X * gt.side.Y
	wrong := 0
	for i := range n {
		o := i * bpp
		if byte(nearestColor(gt.palette, sampled.pix[o], sampled.pix[o+1], sampled.pix[o+2])) != gt.matrix[i] {
			wrong++
		}
	}
	return float64(wrong) / float64(n), true
}

// nearestColor returns the index of the palette colour closest to (r,g,b) by
// squared Euclidean distance in RGB.
func nearestColor(pal []byte, r, g, b byte) int {
	best, bestD := 0, math.MaxFloat64
	for i := range len(pal) / 3 {
		dr := float64(r) - float64(pal[i*3])
		dg := float64(g) - float64(pal[i*3+1])
		db := float64(b) - float64(pal[i*3+2])
		if d := dr*dr + dg*dg + db*db; d < bestD {
			bestD, best = d, i
		}
	}
	return best
}

// --- degradations (image.Image -> image.Image, seeded) ---

func toNRGBA(src image.Image) *image.NRGBA {
	b := src.Bounds()
	dst := image.NewNRGBA(b)
	draw.Draw(dst, b, src, b.Min, draw.Src)
	return dst
}

func clampByte(v float64) byte { return byte(min(max(v, 0), 255) + 0.5) }

// jpegRecompress round-trips the image through JPEG at the given quality, injecting
// the 4:2:0 chroma subsampling and block artefacts of a real photo.
func jpegRecompress(src image.Image, quality float64, _ *rand.Rand) image.Image {
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, src, &jpeg.Options{Quality: int(quality)}); err != nil {
		return src
	}
	out, err := jpeg.Decode(&buf)
	if err != nil {
		return src
	}
	return out
}

// boxBlurDeg applies a separable box blur of the given radius (defocus / optical
// low-pass).
func boxBlurDeg(src image.Image, radius float64, _ *rand.Rand) image.Image {
	r := int(radius)
	if r < 1 {
		return src
	}
	in := toNRGBA(src)
	b := in.Bounds()
	w, h := b.Dx(), b.Dy()
	tmp := image.NewNRGBA(b)
	win := float64(2*r + 1)
	for y := range h { // horizontal
		for c := range 3 {
			var sum float64
			for k := -r; k <= r; k++ {
				sum += float64(in.Pix[y*in.Stride+clampIdx(k, w)*4+c])
			}
			tmp.Pix[y*tmp.Stride+0*4+c] = clampByte(sum / win)
			for x := 1; x < w; x++ {
				sum += float64(in.Pix[y*in.Stride+clampIdx(x+r, w)*4+c]) - float64(in.Pix[y*in.Stride+clampIdx(x-1-r, w)*4+c])
				tmp.Pix[y*tmp.Stride+x*4+c] = clampByte(sum / win)
			}
		}
		for x := range w {
			tmp.Pix[y*tmp.Stride+x*4+3] = 255
		}
	}
	out := image.NewNRGBA(b)
	for x := range w { // vertical
		for c := range 3 {
			var sum float64
			for k := -r; k <= r; k++ {
				sum += float64(tmp.Pix[clampIdx(k, h)*tmp.Stride+x*4+c])
			}
			out.Pix[0*out.Stride+x*4+c] = clampByte(sum / win)
			for y := 1; y < h; y++ {
				sum += float64(tmp.Pix[clampIdx(y+r, h)*tmp.Stride+x*4+c]) - float64(tmp.Pix[clampIdx(y-1-r, h)*tmp.Stride+x*4+c])
				out.Pix[y*out.Stride+x*4+c] = clampByte(sum / win)
			}
		}
		for y := range h {
			out.Pix[y*out.Stride+x*4+3] = 255
		}
	}
	return out
}

func clampIdx(i, n int) int { return min(max(i, 0), n-1) }

// colorCast applies a per-channel gain (a warm white-balance cast): strength in
// [0,1] scales red up and blue down, the cast that defeats raw-RGB-order finder
// classification.
func colorCast(src image.Image, strength float64, _ *rand.Rand) image.Image {
	in := toNRGBA(src)
	gain := [3]float64{1 + 0.4*strength, 1, 1 - 0.3*strength}
	for i := 0; i < len(in.Pix); i += 4 {
		for c := range 3 {
			in.Pix[i+c] = clampByte(float64(in.Pix[i+c]) * gain[c])
		}
	}
	return in
}

// gaussianNoise adds zero-mean Gaussian noise of the given standard deviation
// (sensor noise) to each colour channel.
func gaussianNoise(src image.Image, sd float64, rng *rand.Rand) image.Image {
	in := toNRGBA(src)
	for i := 0; i < len(in.Pix); i += 4 {
		for c := range 3 {
			in.Pix[i+c] = clampByte(float64(in.Pix[i+c]) + rng.NormFloat64()*sd)
		}
	}
	return in
}

// screenLattice multiplies the image by a separable periodic grid of the given
// pixel pitch, darkening inter-cell gaps to mimic a TFT's lit-diode / dark-gap
// lattice — the moiré source the descreen retry targets. Unlike the real capture
// (modules ≈ lattice), here the modules stay larger than the pitch, so this also
// checks that descreen *recovers* detection, not merely that it runs.
func screenLattice(src image.Image, pitch float64, _ *rand.Rand) image.Image {
	p := int(pitch)
	if p < 2 {
		return src
	}
	in := toNRGBA(src)
	b := in.Bounds()
	w, h := b.Dx(), b.Dy()
	lit := func(k int) float64 { // lit half of each cell, dark gap otherwise
		if k%p < (p+1)/2 {
			return 1.0
		}
		return 0.35
	}
	for y := range h {
		fy := lit(y)
		for x := range w {
			m := fy * lit(x)
			o := y*in.Stride + x*4
			for c := range 3 {
				in.Pix[o+c] = clampByte(float64(in.Pix[o+c]) * m)
			}
		}
	}
	return in
}
