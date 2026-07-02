package decode

import (
	"image"
	"image/draw"
	"io"
	"math"
	"sort"
)

// roiMaxDim bounds the longer side of the downscaled copy the region-of-interest
// search analyses. Like coarseMaxDim it keeps the per-image cost independent of the
// capture's megapixels; the proposer only needs the module texture to survive the
// reduction, not full resolution, and it maps the chosen boxes back to
// full-resolution coordinates.
const roiMaxDim = 512

// roiGrid is the number of tiles along the longer side of the working image. The
// tile is the unit the joint score is computed over, derived from the image size
// rather than a fixed pixel count, so the proposer stays scale-adaptive.
const roiGrid = 32

// roiThreshold keeps a tile when its joint score is at least this fraction of the
// peak tile score. It is the floor separating the symbol's dense, colourful texture
// from background clutter; a starting value, refined by measurement, not a copied
// constant tied to any pixel size.
const roiThreshold = 0.20

// roiAnnexThreshold is the lower level of a two-level (hysteresis) threshold: a
// component must contain a tile at roiThreshold to count, but extends through
// connected tiles down to this fraction of the peak. A rotated symbol's corner
// covers only a sliver of its tiles, which score below roiThreshold but far above
// background (measured on the dev capture: corner tiles 0.05-0.20 of peak, noise
// under 0.01), so annexing them keeps the corner inside the component's box.
const roiAnnexThreshold = 0.05

// roiCandidate is a proposed region likely to hold a symbol, in full-resolution
// pixel coordinates, with the joint score that ranked it.
type roiCandidate struct {
	bounds     image.Rectangle
	score      float64 // summed joint tile score over the region, the ranking key
	chromaVar  float64 // mean tile chroma variance over the region
	gradEnergy float64 // mean tile gradient energy over the region
	tiles      int
}

// proposeROIs ranks regions of img by how much they look like a JAB symbol: a dense
// patch that is both high in local chroma variance (many different saturated
// colours, unlike a flat coloured UI bar) and high in gradient energy (a fine module
// grid, unlike a plain background or document). The two features are combined
// multiplicatively so a region must satisfy both at once: flat coloured chrome scores
// high chroma but near-zero variance, and a plain rectangle scores low gradient, so
// each drops out of the product. It returns at most maxN candidates, best first, or
// nil if nothing stands out. It reads img only; it never modifies it or the decode.
func proposeROIs(img image.Image, maxN int) []roiCandidate {
	m := buildROITileMap(img)
	peak := m.peak()
	if peak == 0 {
		return nil
	}
	thr := roiThreshold * peak
	weak := make([]bool, len(m.score))
	for i := range m.score {
		weak[i] = m.score[i] >= roiAnnexThreshold*peak
	}

	fb := img.Bounds()
	sx := float64(fb.Dx()) / float64(m.w)
	sy := float64(fb.Dy()) / float64(m.h)
	var cands []roiCandidate
	for _, comp := range labelComponents(weak, m.gx, m.gy) {
		seeded := false
		for _, i := range comp {
			if m.score[i] >= thr {
				seeded = true
				break
			}
		}
		if !seeded {
			continue
		}
		minTx, minTy, maxTx, maxTy := m.gx, m.gy, -1, -1
		var sumScore, sumChroma, sumGrad float64
		for _, i := range comp {
			tx, ty := i%m.gx, i/m.gx
			minTx, minTy = min(minTx, tx), min(minTy, ty)
			maxTx, maxTy = max(maxTx, tx), max(maxTy, ty)
			sumScore += m.score[i]
			sumChroma += m.chroma[i]
			sumGrad += m.grad[i]
		}
		// Pad one tile so a symbol whose border sits mid-tile is not clipped, then
		// map the work-pixel extents back to full-resolution coordinates.
		x0 := clamp((minTx-1)*m.tile, 0, m.w)
		y0 := clamp((minTy-1)*m.tile, 0, m.h)
		x1 := clamp((maxTx+2)*m.tile, 0, m.w)
		y1 := clamp((maxTy+2)*m.tile, 0, m.h)
		n := len(comp)
		cands = append(cands, roiCandidate{
			bounds: image.Rect(
				fb.Min.X+int(float64(x0)*sx), fb.Min.Y+int(float64(y0)*sy),
				fb.Min.X+int(float64(x1)*sx), fb.Min.Y+int(float64(y1)*sy)),
			score:      sumScore,
			chromaVar:  sumChroma / float64(n),
			gradEnergy: sumGrad / float64(n),
			tiles:      n,
		})
	}
	sort.SliceStable(cands, func(i, j int) bool { return cands[i].score > cands[j].score })
	if len(cands) > maxN {
		cands = cands[:maxN]
	}
	return cands
}

// roiTileMap is the per-tile joint-score grid proposeROIs thresholds: the
// max-normalized chroma-variance and gradient-energy features and their product
// over the gx by gy tile grid of the w by h downscaled working image.
type roiTileMap struct {
	score, chroma, grad []float64
	gx, gy, tile        int
	w, h                int
}

func buildROITileMap(img image.Image) roiTileMap {
	small := downscaleToMax(img, roiMaxDim)
	w, h := small.Bounds().Dx(), small.Bounds().Dy()
	if w < 2 || h < 2 {
		return roiTileMap{}
	}
	tile := max(max(w, h)/roiGrid, 1)
	gx, gy := (w+tile-1)/tile, (h+tile-1)/tile

	chroma := make([]float64, gx*gy)
	grad := make([]float64, gx*gy)
	for ty := range gy {
		for tx := range gx {
			cv, ge := tileFeatures(small, tx*tile, ty*tile, tile, w, h)
			chroma[ty*gx+tx] = cv
			grad[ty*gx+tx] = ge
		}
	}
	normalizeMax(chroma)
	normalizeMax(grad)

	score := make([]float64, gx*gy)
	for i := range score {
		score[i] = chroma[i] * grad[i]
	}
	return roiTileMap{score: score, chroma: chroma, grad: grad, gx: gx, gy: gy, tile: tile, w: w, h: h}
}

func (m roiTileMap) peak() float64 {
	var p float64
	for _, s := range m.score {
		p = max(p, s)
	}
	return p
}

// tileFeatures returns the chroma variance and mean gradient energy over the tile
// whose top-left work-pixel is (x0, y0), clipped to the w x h working image. Chroma
// is the (R-G, G-B) opponent pair, whose variance across the tile is large when the
// tile holds many different hues and near zero on a uniform colour; gradient energy
// is the mean absolute luma step to the right and down neighbours.
func tileFeatures(im *image.NRGBA, x0, y0, tile, w, h int) (chromaVar, gradEnergy float64) {
	x1, y1 := min(x0+tile, w), min(y0+tile, h)
	var n int
	var sCr, sCb, sCr2, sCb2, gSum float64
	var gN int
	for y := y0; y < y1; y++ {
		row := y * im.Stride
		for x := x0; x < x1; x++ {
			o := row + x*4
			r, g, b := float64(im.Pix[o]), float64(im.Pix[o+1]), float64(im.Pix[o+2])
			cr, cb := r-g, g-b
			sCr, sCb = sCr+cr, sCb+cb
			sCr2, sCb2 = sCr2+cr*cr, sCb2+cb*cb
			n++
			lum := luma(r, g, b)
			if x+1 < w {
				p := row + (x+1)*4
				gSum += math.Abs(lum - luma(float64(im.Pix[p]), float64(im.Pix[p+1]), float64(im.Pix[p+2])))
				gN++
			}
			if y+1 < h {
				p := (y+1)*im.Stride + x*4
				gSum += math.Abs(lum - luma(float64(im.Pix[p]), float64(im.Pix[p+1]), float64(im.Pix[p+2])))
				gN++
			}
		}
	}
	if n == 0 {
		return 0, 0
	}
	fn := float64(n)
	chromaVar = (sCr2/fn - (sCr/fn)*(sCr/fn)) + (sCb2/fn - (sCb/fn)*(sCb/fn))
	if gN > 0 {
		gradEnergy = gSum / float64(gN)
	}
	return chromaVar, gradEnergy
}

func luma(r, g, b float64) float64 { return 0.299*r + 0.587*g + 0.114*b }

// normalizeMax scales v into [0, 1] by its maximum, leaving a zero entry at zero so a
// feature-less tile stays feature-less in the product. An all-zero slice is left as is.
func normalizeMax(v []float64) {
	var mx float64
	for _, x := range v {
		mx = max(mx, x)
	}
	if mx == 0 {
		return
	}
	for i := range v {
		v[i] /= mx
	}
}

// labelComponents groups the set tiles of mask into 8-connected components, each
// returned as a list of tile indices into the gx by gy grid.
func labelComponents(mask []bool, gx, gy int) [][]int {
	seen := make([]bool, len(mask))
	var comps [][]int
	var stack []int
	for start := range mask {
		if !mask[start] || seen[start] {
			continue
		}
		seen[start] = true
		stack = append(stack[:0], start)
		var comp []int
		for len(stack) > 0 {
			i := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			comp = append(comp, i)
			cx, cy := i%gx, i/gx
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					if dx == 0 && dy == 0 {
						continue
					}
					nx, ny := cx+dx, cy+dy
					if nx < 0 || ny < 0 || nx >= gx || ny >= gy {
						continue
					}
					j := ny*gx + nx
					if mask[j] && !seen[j] {
						seen[j] = true
						stack = append(stack, j)
					}
				}
			}
		}
		comps = append(comps, comp)
	}
	return comps
}

func clamp(v, lo, hi int) int { return min(max(v, lo), hi) }

// cropImage returns the r portion of img as a standalone image, clipped to img's
// bounds. The per-region decode retry probes orientation on such crops, so the
// probe's downscale works at the region's scale rather than the whole frame's.
func cropImage(img image.Image, r image.Rectangle) *image.NRGBA {
	r = r.Intersect(img.Bounds())
	out := image.NewNRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	draw.Draw(out, out.Bounds(), img, r.Min, draw.Src)
	return out
}

// diagROIProposals dumps the ranked region-of-interest proposals for img, the
// measurement vehicle for whether the joint chroma/gradient score isolates the
// symbol before the proposer is wired into the decode path, and returns them for
// the follow-on per-region diagnostics.
func diagROIProposals(w io.Writer, img image.Image) []roiCandidate {
	rois := proposeROIs(img, 6)
	diagLogf(w, "ROI proposals (joint chroma-variance x gradient-energy): %d", len(rois))
	for i, r := range rois {
		diagLogf(w, "  ROI %d score=%.3f chromaVar=%.3f grad=%.3f tiles=%d box=(%d,%d)-(%d,%d)",
			i, r.score, r.chromaVar, r.gradEnergy, r.tiles,
			r.bounds.Min.X, r.bounds.Min.Y, r.bounds.Max.X, r.bounds.Max.Y)
	}
	diagROITileMap(w, img)
	return rois
}

// diagROITileMap prints the joint-score grid as an ASCII map banded relative to
// the peak tile, showing which tiles pass roiThreshold ('#') and how far below it
// the near-miss tiles sit - the evidence for whether a clipped symbol corner is a
// sub-threshold sliver (recoverable by a lower annex threshold) or scores zero.
func diagROITileMap(w io.Writer, img image.Image) {
	m := buildROITileMap(img)
	peak := m.peak()
	if peak == 0 {
		diagLogf(w, "ROI tile map: flat image, no score peak")
		return
	}
	diagLogf(w, "ROI tile map (%dx%d tiles of %d work-px, work %dx%d, peak=%.4f):",
		m.gx, m.gy, m.tile, m.w, m.h, peak)
	diagLogf(w, "  '#' >= %.2f*peak (roiThreshold)  '+' >= %.2f (roiAnnexThreshold)  '.' >= 0.01  ' ' below",
		roiThreshold, roiAnnexThreshold)
	for ty := range m.gy {
		row := make([]byte, m.gx)
		for tx := range m.gx {
			switch s := m.score[ty*m.gx+tx] / peak; {
			case s >= roiThreshold:
				row[tx] = '#'
			case s >= roiAnnexThreshold:
				row[tx] = '+'
			case s >= 0.01:
				row[tx] = '.'
			default:
				row[tx] = ' '
			}
		}
		diagLogf(w, "  |%s|", row)
	}
}
