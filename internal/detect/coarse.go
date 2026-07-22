package detect

import (
	"image"
	"image/draw"
	"math"
	"sort"
	"sync"

	"github.com/srlehn/jabcode/internal/core"
)

// CoarseMaxDim bounds the longer side of the downscaled image the coarse orientation
// search runs on. Bounding the probe resolution makes the per-rung cost independent of
// the capture's megapixels - the dominant cost of a failed read on a large photo. The
// finder run-lengths survive the reduction as long as the symbol fills a reasonable
// fraction of the frame; a small, strongly-rotated symbol is an accepted miss here (the
// upright pass and later region-of-interest work cover the small-symbol case).
const CoarseMaxDim = 512

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

// CoarseOrientationRungs probes a 90-degree window of pre-rotations on a downscaled copy of
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
func CoarseOrientationRungs(img image.Image) []float64 {
	return FamiliesToRungs(CoarseProbeFamilies(img))
}

// CoarseFamily is one probe rung's measurement: the pre-rotation angle and the
// finder cross-check evidence the raw pass produced there.
type CoarseFamily struct {
	Deg   float64
	Types int // finder types with at least one cross-check survivor (0..4)
	Sum   int // total cross-check survivors, the tie-break
}

// CoarseProbeTrace owns the bounded probe input and one fixed-order record per
// tested angle. It is produced only for detailed diagnostics.
type CoarseProbeTrace struct {
	Input  *image.NRGBA
	Angles []CoarseAngleTrace
}

// CoarseAngleTrace records the actual balanced and binarized probe canvas plus
// its raw finder-pass result.
type CoarseAngleTrace struct {
	Family   CoarseFamily
	Bitmap   *core.Bitmap
	Channels [3]*core.Bitmap
	Pass     FinderPassStats
}

// CoarseProbeFamilies measures every coarseProbeAngles rung with a single raw
// finder pass on a downscaled copy of img, returning one unfiltered result per
// rung; FamiliesToRungs applies the retention policy. The rungs only read the
// shared downscaled copy and each writes its own result slot, so they run
// concurrently while the returned order stays the fixed angle order.
func CoarseProbeFamilies(img image.Image) []CoarseFamily {
	return coarseProbeFamilies(img, CoarseMaxDim, nil)
}

// CoarseProbeFamiliesTraced is CoarseProbeFamilies with detailed observation
// of the same probe run.
func CoarseProbeFamiliesTraced(img image.Image) ([]CoarseFamily, CoarseProbeTrace) {
	var trace CoarseProbeTrace
	families := coarseProbeFamilies(img, CoarseMaxDim, &trace)
	return families, trace
}

// CoarseProbeFamiliesWithin is CoarseProbeFamilies with the probe copy's
// longer side bounded by maxDim instead of CoarseMaxDim. A dense symbol
// filling a large frame can hold too few probe pixels per module under the
// default bound for any cross-check to survive (roughly five per module are
// needed), even though the same symbol reads fine once an orientation is
// known; probing under a doubled bound doubles that per-module budget. The
// probe cost grows with the square of the bound, so callers escalate only
// after a cheaper probe retained nothing.
func CoarseProbeFamiliesWithin(img image.Image, maxDim int) []CoarseFamily {
	return coarseProbeFamilies(img, maxDim, nil)
}

// CoarseProbeFamiliesWithinTraced is CoarseProbeFamiliesWithin with detailed
// observation of the same probe run.
func CoarseProbeFamiliesWithinTraced(img image.Image, maxDim int) ([]CoarseFamily, CoarseProbeTrace) {
	var trace CoarseProbeTrace
	families := coarseProbeFamilies(img, maxDim, &trace)
	return families, trace
}

// coarseProbeFamiliesPrepared is the non-traced probe seam for resident
// preprocessors. Rotation and the bounded probe canvas remain shared CPU
// geometry, while the caller supplies the balanced pixels and masks used by
// each raw finder pass. Detailed traces continue through coarseProbeFamilies
// so they retain the established CPU-owned diagnostic images.
func coarseProbeFamiliesPrepared(
	img image.Image,
	maxDim int,
	prepare func(*core.Bitmap) ([3]*core.Bitmap, error),
) ([]CoarseFamily, error) {
	small := DownscaleToMax(img, maxDim)
	results := make([]*CoarseFamily, len(coarseProbeAngles))
	for idx, deg := range coarseProbeAngles {
		var bm *core.Bitmap
		if deg != 0 {
			bm = RotateToBitmap(small, deg)
		} else {
			bm = core.BitmapFromImage(small)
		}
		channels, err := prepare(bm)
		if err != nil {
			return nil, err
		}
		d := &PrimaryDetector{BM: bm, Ch: channels, Mode: IntensiveDetect}
		d.findPrimarySymbol()
		if len(d.Stats.Passes) == 0 {
			continue
		}
		types, sum := 0, 0
		for _, count := range d.Stats.Passes[0].CrossSurvivors {
			if count > 0 {
				types++
			}
			sum += count
		}
		results[idx] = &CoarseFamily{Deg: deg, Types: types, Sum: sum}
	}
	families := make([]CoarseFamily, 0, len(results))
	for _, family := range results {
		if family != nil {
			families = append(families, *family)
		}
	}
	return families, nil
}

// coarseProbeFamilies is CoarseProbeFamilies under a caller-chosen resolution
// bound.
func coarseProbeFamilies(img image.Image, maxDim int, trace *CoarseProbeTrace) []CoarseFamily {
	small := DownscaleToMax(img, maxDim)
	results := make([]*CoarseFamily, len(coarseProbeAngles))
	var angleTraces []CoarseAngleTrace
	if trace != nil {
		angleTraces = make([]CoarseAngleTrace, len(coarseProbeAngles))
		trace.Input = small
	}
	var wg sync.WaitGroup
	wg.Add(len(coarseProbeAngles))
	for idx, deg := range coarseProbeAngles {
		go func() {
			defer wg.Done()
			var bm *core.Bitmap
			if deg != 0 {
				bm = RotateToBitmap(small, deg)
			} else {
				bm = core.BitmapFromImage(small)
			}
			BalanceRGB(bm)
			ch := BinarizerRGB(bm, nil)
			d := &PrimaryDetector{BM: bm, Ch: ch, Mode: IntensiveDetect}
			d.findPrimarySymbol()
			if len(d.Stats.Passes) == 0 {
				if trace != nil {
					angleTraces[idx] = CoarseAngleTrace{Family: CoarseFamily{Deg: deg}, Bitmap: bm, Channels: ch}
				}
				return
			}
			types, sum := 0, 0
			for _, c := range d.Stats.Passes[0].CrossSurvivors {
				if c > 0 {
					types++
				}
				sum += c
			}
			family := CoarseFamily{deg, types, sum}
			results[idx] = &family
			if trace != nil {
				angleTraces[idx] = CoarseAngleTrace{Family: family, Bitmap: bm, Channels: ch, Pass: d.Stats.Passes[0]}
			}
		}()
	}
	wg.Wait()
	if trace != nil {
		trace.Angles = angleTraces
	}
	fams := make([]CoarseFamily, 0, len(coarseProbeAngles))
	for _, r := range results {
		if r != nil {
			fams = append(fams, *r)
		}
	}
	return fams
}

// FamiliesToRungs keeps the families with enough finder types, best first, capped
// at maxCoarseFamilies, and expands each to its four 90-degree turns.
func FamiliesToRungs(fams []CoarseFamily) []float64 {
	return familiesToRungs(fams, maxCoarseFamilies)
}

// FamiliesToRungsUncapped is FamiliesToRungs without the maxCoarseFamilies cut,
// in the same best-first ladder order. Fine-resolution probes lose the cap's
// premise: image-grid and lattice texture inflates cross survivors at wrong
// angles, so the two best-scoring families no longer reliably bracket the true
// orientation - on measured captures the true family sat at rank 3-4 behind
// texture families. Escalated probes therefore keep every family passing the
// types floor and let the decode ladder discriminate; the floor still bounds
// the set (at most six families).
func FamiliesToRungsUncapped(fams []CoarseFamily) []float64 {
	return familiesToRungs(fams, len(fams))
}

// familiesToRungs applies the retention policy: the types floor, best-first
// order, at most maxFamilies kept, each expanded to its four 90-degree turns.
func familiesToRungs(fams []CoarseFamily, maxFamilies int) []float64 {
	var kept []CoarseFamily
	for _, f := range fams {
		if f.Types >= coarseFamilyTypes {
			kept = append(kept, f)
		}
	}
	sort.SliceStable(kept, func(i, j int) bool {
		if kept[i].Types != kept[j].Types {
			return kept[i].Types > kept[j].Types
		}
		return kept[i].Sum > kept[j].Sum
	})
	if len(kept) > maxFamilies {
		kept = kept[:maxFamilies]
	}
	var rungs []float64
	for _, f := range kept {
		rungs = append(rungs, f.Deg, f.Deg+90, f.Deg+180, f.Deg+270)
	}
	return rungs
}

// zeroOriginNRGBA returns src itself when it already is a tightly packed
// zero-origin NRGBA image, or nil. Callers use it to skip a full-frame copy;
// the aliased image must be treated read-only.
func zeroOriginNRGBA(src image.Image) *image.NRGBA {
	n, ok := src.(*image.NRGBA)
	if !ok || n.Rect.Min != (image.Point{}) || n.Stride != n.Rect.Dx()*4 {
		return nil
	}
	return n
}

// DownscaleToMax returns src reduced so its longer side is at most maxDim, by averaging
// each destination pixel over the source box it covers (a box filter, which preserves the
// finder rings better than point sampling at the reduction the coarse search needs). An
// image already within the bound is returned unreduced - aliased when it already is a
// zero-origin NRGBA, as an NRGBA copy otherwise - so treat the result read-only.
func DownscaleToMax(src image.Image, maxDim int) *image.NRGBA {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	in := zeroOriginNRGBA(src)
	if in == nil {
		in = image.NewNRGBA(image.Rect(0, 0, w, h))
		draw.Draw(in, in.Bounds(), src, b.Min, draw.Src)
	}
	if w <= maxDim && h <= maxDim {
		return in
	}
	scale := float64(maxDim) / float64(max(w, h))
	nw := max(int(math.Round(float64(w)*scale)), 1)
	nh := max(int(math.Round(float64(h)*scale)), 1)
	return downscaleNRGBA(in, nw, nh)
}

// HalveNRGBA returns in box-filtered to half size, rounding odd sides up - the
// constructor for successive resolution-pyramid levels. Each destination pixel
// averages the 2x2 source box it covers, so every halving is also a mild
// low-pass: coarse levels arrive pre-smoothed, which is why they can decode
// captures whose full-resolution noise defeats detection.
func HalveNRGBA(in *image.NRGBA) *image.NRGBA {
	w, h := in.Rect.Dx(), in.Rect.Dy()
	return downscaleNRGBA(in, max((w+1)/2, 1), max((h+1)/2, 1))
}

// downscaleNRGBA box-filters in down to nw x nh, averaging each destination
// pixel over the source box it covers.
func downscaleNRGBA(in *image.NRGBA, nw, nh int) *image.NRGBA {
	w, h := in.Rect.Dx(), in.Rect.Dy()
	base := in.PixOffset(in.Rect.Min.X, in.Rect.Min.Y)
	out := image.NewNRGBA(image.Rect(0, 0, nw, nh))
	core.ParallelChunks(nh, 8, func(lo, hi int) {
		for oy := lo; oy < hi; oy++ {
			sy0 := oy * h / nh
			sy1 := max((oy+1)*h/nh, sy0+1)
			for ox := range nw {
				sx0 := ox * w / nw
				sx1 := max((ox+1)*w/nw, sx0+1)
				var rs, gs, bs, n int
				for sy := sy0; sy < sy1; sy++ {
					row := base + sy*in.Stride
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
	})
	return out
}
