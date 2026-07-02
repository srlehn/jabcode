package decode

import (
	"image"
	"image/draw"
	"math"
	"sort"
)

// coarseMaxDim bounds the longer side of the downscaled image the coarse orientation
// search runs on. Bounding the probe resolution makes the per-rung cost independent of
// the capture's megapixels - the dominant cost of a failed read on a large photo. The
// finder run-lengths survive the reduction as long as the symbol fills a reasonable
// fraction of the frame; a small, strongly-rotated symbol is an accepted miss here (the
// upright pass and later region-of-interest work cover the small-symbol case).
const coarseMaxDim = 512

// coarseFamilyTypes is how many of the four finder types must cross-check at a window rung
// before the coarse search treats it as a real orientation rather than a chance alignment
// in structured non-code clutter. A clean code's aligned orientation reaches all four, but
// the green+blue (FP0/FP3) branch is the weaker one and can drop a type at the downscaled
// scale, so the floor is three; structured noise produces none, so it still yields nil.
const coarseFamilyTypes = 3

// maxCoarseFamilies caps how many orientation families the full-resolution retry will try.
// The true orientation is straddled by two adjacent window rungs, so the two best-scoring
// families always bracket it; each expands to four 90-degree turns, bounding the
// full-resolution work to maxCoarseFamilies*4 decodes.
const maxCoarseFamilies = 2

// coarseOrientationRungs probes a 90-degree window of pre-rotations on a downscaled copy of
// img with a single raw finder pass and returns the orientations worth a full-resolution
// decode, or nil if none looks like a symbol. The discriminator is the cross-check survivor
// count: counter-rotating the image to within the finder survival band of upright makes the
// single-module 1:1:1:1:1 run-lengths integer on the raster again, so an aligned rung
// spikes cross survivors across the finder types while a wrong angle leaves them near zero.
// Raw run-length hits cannot discriminate orientation - they are rotation-robust by design
// - but cross survivors can. The probe is raw only, with no avg-RGB or descreen retry, so
// it stays a cheap orientation scan; the expensive retries are reserved for the returned
// full-resolution rungs.
//
// The finder arrangement is identical under a 90-degree turn, so the probe cannot tell the
// true orientation from its three rotational aliases - only a full data decode can - and a
// 90-degree window already determines the family. Each retained family is therefore
// expanded to its four 90-degree turns (90 is a whole number of 15-degree steps, so all
// four share the rung's small residual), one of which is the true orientation.
func coarseOrientationRungs(img image.Image) []float64 {
	return familiesToRungs(coarseProbeFamilies(img))
}

// coarseFamily is one probe rung's measurement: the pre-rotation angle and the
// finder cross-check evidence the raw pass produced there.
type coarseFamily struct {
	deg   float64
	types int // finder types with at least one cross-check survivor (0..4)
	sum   int // total cross-check survivors, the tie-break
}

// coarseProbeFamilies measures every coarseProbeAngles rung with a single raw
// finder pass on a downscaled copy of img, returning one unfiltered result per
// rung; familiesToRungs applies the retention policy.
func coarseProbeFamilies(img image.Image) []coarseFamily {
	small := downscaleToMax(img, coarseMaxDim)
	fams := make([]coarseFamily, 0, len(coarseProbeAngles))
	for _, deg := range coarseProbeAngles {
		var rot image.Image = small
		if deg != 0 {
			rot = rotateImage(small, deg)
		}
		bm := bitmapFromImage(rot)
		balanceRGB(bm)
		ch := binarizerRGB(bm, nil)
		d := &primaryDetector{bm: bm, ch: ch, mode: intensiveDetect}
		d.findPrimarySymbol()
		if len(d.stats.passes) == 0 {
			continue
		}
		types, sum := 0, 0
		for _, c := range d.stats.passes[0].crossSurvivors {
			if c > 0 {
				types++
			}
			sum += c
		}
		fams = append(fams, coarseFamily{deg, types, sum})
	}
	return fams
}

// familiesToRungs keeps the families with enough finder types, best first, capped
// at maxCoarseFamilies, and expands each to its four 90-degree turns.
func familiesToRungs(fams []coarseFamily) []float64 {
	var kept []coarseFamily
	for _, f := range fams {
		if f.types >= coarseFamilyTypes {
			kept = append(kept, f)
		}
	}
	sort.SliceStable(kept, func(i, j int) bool {
		if kept[i].types != kept[j].types {
			return kept[i].types > kept[j].types
		}
		return kept[i].sum > kept[j].sum
	})
	if len(kept) > maxCoarseFamilies {
		kept = kept[:maxCoarseFamilies]
	}
	var rungs []float64
	for _, f := range kept {
		rungs = append(rungs, f.deg, f.deg+90, f.deg+180, f.deg+270)
	}
	return rungs
}

// downscaleToMax returns src reduced so its longer side is at most maxDim, by averaging
// each destination pixel over the source box it covers (a box filter, which preserves the
// finder rings better than point sampling at the reduction the coarse search needs). An
// image already within the bound is returned as an NRGBA copy unchanged.
func downscaleToMax(src image.Image, maxDim int) *image.NRGBA {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	in := image.NewNRGBA(image.Rect(0, 0, w, h))
	draw.Draw(in, in.Bounds(), src, b.Min, draw.Src)
	if w <= maxDim && h <= maxDim {
		return in
	}
	scale := float64(maxDim) / float64(max(w, h))
	nw := max(int(math.Round(float64(w)*scale)), 1)
	nh := max(int(math.Round(float64(h)*scale)), 1)
	out := image.NewNRGBA(image.Rect(0, 0, nw, nh))
	for oy := range nh {
		sy0 := oy * h / nh
		sy1 := max((oy+1)*h/nh, sy0+1)
		for ox := range nw {
			sx0 := ox * w / nw
			sx1 := max((ox+1)*w/nw, sx0+1)
			var rs, gs, bs, n int
			for sy := sy0; sy < sy1; sy++ {
				row := sy * in.Stride
				for sx := sx0; sx < sx1; sx++ {
					o := row + sx*4
					rs += int(in.Pix[o+0])
					gs += int(in.Pix[o+1])
					bs += int(in.Pix[o+2])
					n++
				}
			}
			o := oy*out.Stride + ox*4
			out.Pix[o+0] = byte(rs / n)
			out.Pix[o+1] = byte(gs / n)
			out.Pix[o+2] = byte(bs / n)
			out.Pix[o+3] = 255
		}
	}
	return out
}
